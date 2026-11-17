// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread-cow.h"

#include <cstdio>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/common/ptr-compr.h"
#include "src/common/assert-scope.h"
#include "src/execution/green-thread-scheduler.h"
#include "src/execution/heap-ref-table.h"
#include "src/execution/isolate.h"
#include "src/handles/handles.h"
#include "src/heap/gc-callbacks.h"

// Provide the out-of-line definitions of HandleScope::HandleScope() and the
// slot store primitives used below. In V8's shared-library build these are
// exported from the V8 dylib; in a static embedding (Node.js) they must be
// emitted within this TU, which is the only caller that needs them out-of-line.
#include "src/handles/handles-inl.h"
#include "src/heap/heap.h"
#include "src/objects/heap-object-inl.h"
#include "src/objects/compressed-slots-inl.h"
#include "src/objects/slots-inl.h"

namespace v8 {
namespace internal {

// Thread-local counter tracking how many COW-enabled green threads are active.
std::atomic<uint32_t> GreenThreadCow::g_cow_active_count{0};

// Counts CowCloneForGeneratedCode entries past the fast-path guards, i.e.
// real TurboFan kArchStoreWithWriteBarrier stores that consulted the COW
// hook with COW active. Exposed for JIT→COW boundary verification.
std::atomic<uint64_t> GreenThreadCow::g_cow_generated_code_calls{0};

// Re-entry guard: prevents infinite recursion when cloning triggers another
// write barrier. CloneObjectInternal allocates a new heap object, which may
// fire internal V8 stores that re-enter COWWriteBarrierCheck. Returning
// unchanged on nested entry (depth > 0) breaks the cycle safely — the outer
// call owns the clone; the inner call defers.
static thread_local int64_t g_cow_nesting_depth = 0;

// Isolates on which at least one COW green thread has run, plus a mutex
// guarding the set. The set is consulted by the JIT code generator to decide
// whether to emit the COW write-barrier callout; it is set once per isolate and
// never cleared, because bodies compiled in the window between spawnAll calls
// must still emit the callout even though the active count is zero then.
namespace {
std::mutex g_cow_capable_isolates_mutex;
std::unordered_set<Isolate*> g_cow_capable_isolates;
}  // namespace

// ================================================================
// GC root integration
//
// A green thread's copy-on-write heap view is a set of GT-local clones that
// are reachable only through this module's bookkeeping. On their own those
// clones are invisible to V8's GC, so the first collection during a live green
// thread could either collect a copy the body is still using or
// compact-move it while the body holds a redirected reference to the old
// address. To make the views GC-safe we register a Heap::GCRootsProvider that
// roots every active view during the mark phase: both the main-heap original
// (so the snapshot a GT is reading stays alive) and the GT-local copy (so it
// stays alive and is relocated if compaction moves it). The per-GT stack and
// register rewrite on relocation is handled inside HeapRefTable::VisitAsRoots.
//
// The provider is wired to the COW active count so it is registered on the
// heap exactly while at least one green thread's view is active, and each live
// view is listed for the provider to walk. In the Model A single-execution-
// thread design GC runs on the same thread that drives green-thread lifecycle,
// so these accesses never race; the registry mutex is retained for future
// multi-threaded variants.
// ================================================================

namespace {
// Currently COW-enabled green threads, walked by the GC roots provider.
std::mutex g_cow_green_threads_mutex;
std::vector<GreenThread*> g_cow_green_threads;

// Heap::GCRootsProvider that roots every active COW green-thread view during
// the mark phase. Registered globally on the heap via Heap::AddGlobalGCRootsProvider.
class GreenThreadGCRootsProvider final : public GCRootsProvider {
 public:
  void Iterate(RootVisitor* visitor) override {
    std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
    for (GreenThread* gt : g_cow_green_threads) {
      if (gt == nullptr || !gt->cow_enabled()) continue;
      HeapRefTable* table = gt->cow_ref_table();
      if (table != nullptr) table->VisitAsRoots(visitor);
    }
  }
};

// The provider outlives any single green thread; keep it in static storage so
// the heap always holds the same, stable pointer.
GreenThreadGCRootsProvider* g_cow_gc_root_provider = nullptr;
}  // namespace

// Record this green thread as a live COW view the GC must root. Must be called
// with the same mutex as g_cow_green_threads and before the active count is
// incremented, so a concurrent-in-time GC never walks a view that lacks a
// table or is not yet tracked.
static void RecordCOWGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
  g_cow_green_threads.push_back(gt);
}

// Stop rooting this green thread's view. Mirrors RecordCOWGreenThread.
static void UnrecordCOWGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> guard(g_cow_green_threads_mutex);
  for (auto it = g_cow_green_threads.begin(); it != g_cow_green_threads.end();
       ++it) {
    if (*it == gt) {
      g_cow_green_threads.erase(it);
      return;
    }
  }
}

// Register the GC roots provider on the heap the first time COW becomes active
// (active count 0 -> 1). Balanced by UnregisterCOWRootsProvider on the last
// teardown (1 -> 0) so the provider lives exactly while a view is active.
static void RegisterCOWRootsProvider(Heap* heap) {
  if (g_cow_gc_root_provider == nullptr) {
    g_cow_gc_root_provider = new GreenThreadGCRootsProvider();
  }
  heap->AddGlobalGCRootsProvider(g_cow_gc_root_provider);
}

// Remove the GC roots provider from the heap. Safe to call even if it was never
// registered; RemoveGlobalGCRootsProvider DCHECKs, so guard the null case.
static void UnregisterCOWRootsProvider(Heap* heap) {
  if (g_cow_gc_root_provider != nullptr) {
    heap->RemoveGlobalGCRootsProvider(g_cow_gc_root_provider);
  }
}

void GreenThreadCow::SetIsolateCowCapable(Isolate* isolate) {
  if (isolate == nullptr) return;
  std::lock_guard<std::mutex> lock(g_cow_capable_isolates_mutex);
  g_cow_capable_isolates.insert(isolate);
}

bool GreenThreadCow::IsolateCowCapable(Isolate* isolate) {
  if (isolate == nullptr) return false;
  std::lock_guard<std::mutex> lock(g_cow_capable_isolates_mutex);
  return g_cow_capable_isolates.count(isolate) != 0;
}

void GreenThreadCow::Initialize(GreenThread* gt) {
  // Once any green thread runs on an isolate, mark it COW-capable so compile
  // units generated for it (including by the background compiler after this
  // call returns, while the active count is zero) keep the COW callout wired in.
  SetIsolateCowCapable(gt->isolate());
  HeapRefTable* table = new HeapRefTable();
  table->set_owner(gt);
  gt->set_cow_ref_table(table);
  gt->set_cow_enabled(true);
  gt->clear_cow_dirty_objects();
  gt->set_cow_dirty_flag(false);

  // Root this view during GC so its clones survive collection. Record the view
  // and register the global roots provider on the first transition (0 -> 1),
  // before bumping the active count, so a GC never walks a view lacking a
  // table or that is not yet tracked.
  uint32_t previous_count =
      g_cow_active_count.load(std::memory_order_relaxed);
  RecordCOWGreenThread(gt);
  if (previous_count == 0) {
    RegisterCOWRootsProvider(gt->isolate()->heap());
  }
  g_cow_active_count.fetch_add(1, std::memory_order_release);
}

void GreenThreadCow::Teardown(GreenThread* gt) {
  // Stop rooting this view before dropping the active count. On the last
  // teardown (count 1 -> 0) this also unregisters the global roots provider,
  // keeping the provider registered exactly while a view is active.
  uint32_t previous_count =
      g_cow_active_count.load(std::memory_order_relaxed);
  UnrecordCOWGreenThread(gt);
  g_cow_active_count.fetch_sub(1, std::memory_order_acq_rel);
  if (previous_count == 1) {
    UnregisterCOWRootsProvider(gt->isolate()->heap());
  }

  gt->set_cow_enabled(false);
  gt->clear_cow_dirty_objects();
  gt->set_cow_dirty_flag(false);
  HeapRefTable* table = gt->cow_ref_table();
  gt->set_cow_ref_table(nullptr);
  if (table != nullptr) {
    table->set_owner(nullptr);
    delete table;
  }
}

Address GreenThreadCow::Clone(Isolate* isolate, GreenThread* gt,
                              Address obj_ptr) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (!table) return obj_ptr;
  return table->CloneObject(isolate, obj_ptr);
}

// Helper: get the COW-enabled green thread for this execution context.
static GreenThread* CowEnabledThread() {
  GreenThread* gt = CurrentGreenThread();
  if (!gt || !gt->cow_enabled()) return nullptr;
  if (!gt->cow_ref_table()) return nullptr;
  return gt;
}

Address GreenThreadCow::CowCheckBeforeStore(Isolate* isolate, Address obj_ptr) {
  // Guard against recursive COW operations (e.g., during snapshot
  // deserialization: clone → allocate → write barrier → clone → crash).
  // If we're already inside a clone operation, return unchanged.
  if (g_cow_nesting_depth++ > 0) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  GreenThread* gt = CowEnabledThread();
  if (!gt) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }
  if (!isolate) isolate = gt->isolate();

  // Never clone from inside the collector or during snapshot deserialization.
  // Snapshot deserialization triggers the write barrier for every object read
  // from the snapshot; cloning in that context would try to allocate while
  // the deserializer is actively building the heap.
  if (isolate && (isolate->heap()->gc_state() != Heap::NOT_IN_GC ||
                  !isolate->heap()->deserialization_complete())) {
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  HeapRefTable* table = gt->cow_ref_table();
  if (!table) { g_cow_nesting_depth--; return obj_ptr; }

  // If this object is already cloned, return the copy immediately.
  Address copy = kNullAddress;
  if (table->FindCopy(obj_ptr, &copy)) {
    g_cow_nesting_depth--;
    return copy;
  }

  // First write to this object: clone it for the GT's isolated heap view.
  // The C++ store path (WriteToField → CombinedWriteBarrier) runs under
  // DisallowHeapAllocation — we must explicitly re-allow it here. Allocation
  // also requires safepoints, so both scopes are necessary.
  if (!isolate) { g_cow_nesting_depth--; return obj_ptr; }
  AllowHeapAllocation allow_alloc;
  AllowSafepoints allow_safepoints;
  // Cloning allocates, and the clone path hands out GC-safe handles for every
  // object it touches. Handle creation is forbidden while the isolate's handle
  // scope is sealed, which is the case for every write performed by running JS
  // (Execution::Invoke seals the scope around the JS entry stub). Opening a
  // scope here deliberately unseals it for the duration of the clone — the same
  // pattern V8's own runtime functions use when called from generated code
  // (e.g. Runtime_DefineKeyedOwnIC_Miss opens a HandleScope). The scope is
  // left again before returning, releasing the temporary handles; the clone
  // itself stays reachable through the reference table and the rewritten slot.
  HandleScope handle_scope(isolate);
  copy = table->CloneObject(isolate, obj_ptr);

  if (copy == obj_ptr) {
    // Not cloneable: write through with original.
    g_cow_nesting_depth--;
    return obj_ptr;
  }

  // Record the copy mapping so future accesses get redirected.
  // Also queue the object for stack/register reference redirection.
  table->RecordCopy(obj_ptr, copy);
  gt->add_cow_dirty_object(obj_ptr);
  gt->set_cow_dirty_flag(true);

  g_cow_nesting_depth--;
  return copy;
}

Address GreenThreadCow::COWCheckAndRedirect(Address obj_ptr) {
  GreenThread* gt = CowEnabledThread();
  if (!gt) return obj_ptr;

  HeapRefTable* table = gt->cow_ref_table();
  Address copy = kNullAddress;
  if (!table->FindCopy(obj_ptr, &copy)) return obj_ptr;

  gt->add_cow_dirty_object(obj_ptr);
  gt->set_cow_dirty_flag(true);
  return copy;
}

void GreenThreadCow::RedirectReferences(GreenThread* gt, Address old_ptr,
                                        Address new_ptr) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (table) table->RedirectReferences(gt, old_ptr, new_ptr);
}

void GreenThreadCow::VisitAsRoots(GreenThread* gt, RootVisitor* visitor) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  if (table) table->VisitAsRoots(visitor);
}

void GreenThreadCow::FlushGreenThreadDirtySet(GreenThread* gt) {
  if (!gt) return;
  if (!gt->cow_dirty_flag()) return;

  HeapRefTable* table = gt->cow_ref_table();
  if (!table) {
    gt->set_cow_dirty_flag(false);
    return;
  }

  gt->set_cow_dirty_flag(false);

  // Collect pending objects (the set is swapped so we can iterate safely).
  std::unordered_set<Address> pending;
  pending.swap(gt->cow_dirty_objects_ref());

  // One stack/register scan per batch of writes.
  // At flush time the clone already happened during CowCheckBeforeStore,
  // so we only need to redirect references.
  for (Address main_ptr : pending) {
    Address copy = kNullAddress;
    if (!table->FindCopy(main_ptr, &copy)) continue;
    table->RedirectReferences(gt, main_ptr, copy);
  }
}

int GreenThreadCow::ClonedObjectCount(GreenThread* gt) {
  HeapRefTable* table = gt ? gt->cow_ref_table() : nullptr;
  return table ? table->ClonedCount() : 0;
}

// --- COW runtime helper for TurboFan-generated code ---

// Number of times CowCloneForGeneratedCode was reached with COW active.
// Used by tests to verify the JIT→COW boundary is crossed.
uint64_t GreenThreadCow::CowCloneForGeneratedCodeCallCount() {
  return g_cow_generated_code_calls.load(std::memory_order_relaxed);
}

// Reset the JIT→COW boundary counter (test isolation).
void GreenThreadCow::ResetCowCloneForGeneratedCodeCallCount() {
  g_cow_generated_code_calls.store(0, std::memory_order_relaxed);
}

// TurboFan-generated code calls this to clone a heap object pointer
// for the current green thread's isolated heap view.
//
// Takes a tagged pointer (with kHeapObjectTag for uncompressed or
// compressed form). Decompresses if needed, calls CowCheckBeforeStore,
// then re-tags/re-compresses the result.
//
// Returns the (possibly cloned) tagged pointer address.
Address GreenThreadCow::CowCloneForGeneratedCode(Address tagged_addr) {
  // Fast path: no COW-enabled threads.
  if (V8_LIKELY(!ActiveCount())) return tagged_addr;

  GreenThread* gt = CurrentGreenThread();
  if (!gt) return tagged_addr;

  Isolate* isolate = gt->isolate();
  if (!isolate) return tagged_addr;

  // Smi store operands need no cloning -- they are not heap objects and,
  // like the interpreter path, untagging them as a heap address would feed a
  // garbage pointer into CowCheckBeforeStore. Return the operand unchanged
  // before it is counted as a real JIT->COW clone.
  if (HAS_SMI_TAG(tagged_addr)) return tagged_addr;

  // Reached the JIT→COW boundary: a TurboFan kArchStoreWithWriteBarrier
  // store with COW active. Count it for verification.
  g_cow_generated_code_calls.fetch_add(1, std::memory_order_relaxed);

  // Decompress the tagged address to get the raw heap object address. In
  // this configuration store operands arrive as full (decompressed) tagged
  // addresses; DecompressTagged is identity there, so both forms are handled.
  Address obj_ptr;
#if V8_COMPRESS_POINTERS
  obj_ptr = V8HeapCompressionScheme::DecompressTagged(
      static_cast<Tagged_t>(tagged_addr));
#else
  obj_ptr = tagged_addr;
#endif
  // DecompressTagged preserves the tag bits; strip the heap object tag to
  // get the untagged address the clone table and CloneObject expect. If the
  // store operand arrived untagged (no tag bits), pass it through unchanged.
  if ((obj_ptr & kHeapObjectTagMask) == kHeapObjectTag) {
    obj_ptr -= kHeapObjectTag;
  }

  // Check and clone if needed.
  Address copy_addr = CowCheckBeforeStore(isolate, obj_ptr);

  // Re-tag AND re-compress the result. The operand lives in a general purpose
  // register, and a compressed build stores it with `stur` (32-bit) into the
  // tagged slot, so what must come back is the compressed form Tagged_t, not
  // the full address: handing back a full address truncates to the low 32
  // bits, which are not cage-relative in the shared-cage configuration and
  // therefore decompress to an invalid address on the next load. If nothing was
  // cloned, return the original operand unchanged, preserving its form.
  if (copy_addr == obj_ptr) return tagged_addr;
#if V8_COMPRESS_POINTERS
  return static_cast<Address>(
      V8HeapCompressionScheme::CompressObject(copy_addr + kHeapObjectTag));
#else
  return copy_addr + kHeapObjectTag;
#endif
}

// --- COW write barrier hook (implemented here to avoid header cycles) ---

// Provenance filter: determines whether the host object is a user-level
// mutable container worthy of COW isolation.
//
// V8 internal objects (maps, strings, descriptors, SharedFunctionInfo,
// contexts, allocation sites, etc.) are never cloned by COW. They are
// per-isolate, not per-application-object, and their internal structure
// (unaligned slots, snapshot-allocated backing stores) is not safe to
// deep-clone.  The only user objects COW can isolate are plain JS objects and
// JS arrays — the same types HeapRefTable::IsCowCloneableRootType accepts for a
// deep copy. Every other type (functions, proxies, Maps, Sets, ...) is shared
// and routed through its own native mutation path rather than a green-thread
// copy.
//
// This check is performed on the HOST (the containing object), not the
// VALUE (what is being stored).  V8 internals frequently write to their
// own structures during compilation and GC; filtering on the host skips
// all of those safely while preserving isolation for user writes.
//
// Safety: for test-mode fake addresses (very low numbers), skip the
// check entirely — the address may not point to valid memory, and the
// subsequent CowCheckBeforeStore handles test mode separately.
inline bool IsCowProvenanceHost(Tagged<HeapObject> host) {
  // Test-mode fake objects have very low addresses that fall outside
  // the actual heap.  Dereferencing their map would crash.  Skip the
  // provenance check in that case — the full COW flow handles test
  // mode via null-isolate and deserialization guards.
  // Real V8 heap addresses are typically > 0x100000000 (64-bit).
  Address host_addr = host.address();
  if (host_addr < 0x100000000ULL) return true;  // trust test fake objects

  uint16_t it = host->map()->instance_type();
  switch (it) {
    case JS_OBJECT_TYPE:
    case JS_ARRAY_TYPE:
      return true;
    default:
      return false;
  }
}

// Fast-path COW check for the write barrier path.
//
// Called from CombinedWriteBarrierInternal after the store has completed
// but before GC bookkeeping. If the host is a user COW-tracked object and
// the value needs cloning, rewrites the slot with the copy and queues the
// object for stack/register reference redirection.
//
// The slot rewrite is done via raw address write to avoid calling the
// protected Tagged<HeapObject> constructor directly.
void GreenThreadCow::COWWriteBarrierCheck(Tagged<HeapObject> host,
                                          Tagged<HeapObject> value,
                                          HeapObjectSlot slot) {
  GreenThread* gt = CurrentGreenThread();

  // Smi (small-integer) values are not heap objects and cannot be cloned.
  // Because ComputeWriteBarrierModeForObject forces an UPDATE_WRITE_BARRIER
  // for young hosts while a green thread is active, Smi stores reach this
  // barrier exactly like object stores. Untagging a Smi below as a heap
  // address (value.ptr() - kHeapObjectTag) yields a garbage pointer that
  // CowCheckBeforeStore / CloneObject then dereference and write out of
  // bounds against -- the root cause of the green-thread load crash. This
  // guard short-circuits before any pointer is dereferenced. The check must
  // read the raw tag bits directly: Tagged<HeapObject> overrides IsSmi() to
  // compile-time "false" and IsHeapObject() to compile-time "true", so the
  // usual IsHeapObject(value) overload is always-true here and useless.
  if (HAS_SMI_TAG(value.ptr())) return;

  if (!gt || !gt->cow_enabled()) return;

  // Provenance filter: skip COW for V8 internal host objects.
  // This prevents COW from firing on internal V8 structures
  // (maps, descriptors, SharedFunctionInfo, FixedArray, etc.)
  // during JIT compilation, GC, snapshot deserialization, etc.
  if (!IsCowProvenanceHost(host)) return;

  // Untag the value to get the raw heap address.
  Address value_addr = value.ptr() - kHeapObjectTag;

  // Fast path: if COW is globally inactive, skip entirely.
  if (V8_LIKELY(!GreenThreadCow::ActiveCount())) return;

  // CowCheckBeforeStore eagerly clones on first write and returns the copy.
  // If the object is not cloned yet, it returns the original address.
  Address copy_addr = CowCheckBeforeStore(gt->isolate(), value_addr);

  // If a copy was produced, rewrite the slot with the copy.
  // Note: slot stores the COPY address (not the original). FromAddress
  // takes an untagged, tag-aligned address and applies the tag itself.
  if (copy_addr != value_addr) {
    slot.StoreHeapObject(HeapObject::FromAddress(copy_addr));
  }
}

}  // namespace internal
}  // namespace v8

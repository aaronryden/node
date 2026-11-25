// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_EXECUTION_GREEN_THREAD_COW_H_
#define V8_EXECUTION_GREEN_THREAD_COW_H_

#include <atomic>
#include <cstdint>

#include "src/common/globals.h"
#include "src/execution/green-thread.h"
#include "src/objects/objects.h"

#if V8_ENABLE_GREEN_THREADS
#include "src/base/platform/mutex.h"
#endif  // V8_ENABLE_GREEN_THREADS

namespace v8 {

namespace internal {

class GreenThread;
class RootVisitor;

// GreenThreadCow — copy-on-write support for green thread heap isolation.
//
// This module handles:
// 1. Writing to a captured object triggers a deep clone into the GT's view
// 2. Reading from a captured object uses the clone if one exists
// 3. Managing the reference table to track original <-> view mappings
// 4. Redirecting stack/register references after cloning
// 5. Visiting cloned objects as GC roots
//
// In Model A execution, COW operations happen on the embedding thread.
class GreenThreadCow {
 public:
  // Initialize COW state for a green thread (called when the GT starts).
  static void Initialize(GreenThread* gt);

  // Tear down COW state (called when the GT terminates).
  static void Teardown(GreenThread* gt);

  // Mark an isolate as COW-capable: at least one green thread has run on it.
  // The flag is persistent for the isolate's lifetime (never cleared), so any
  // body compiled after a green-thread run still emits the COW write-barrier
  // callout even though the active count has since dropped back to zero. The
  // alternative (gating on the live active count at codegen time) misses bodies
  // that a background compiler emits in the window between two spawnAll calls,
  // where the count is zero and the callout would be omitted. Snapshot builtins
  // are compiled before any green thread ever runs, so they stay unaffected.
  static void SetIsolateCowCapable(Isolate* isolate);
  static bool IsolateCowCapable(Isolate* isolate);

  // Enable/disable concurrent young-gen allocation on an isolate's green heap.
  // When true, the young-gen LinearAllocationArea's bump-pointer switches to an
  // atomic CAS so multiple OS workers can reserve disjoint pointers from the
  // same green V8 isolate's heap (GT_SHARED_HEAP=1). Only the green-threads
  // bridge calls this; the default (false) leaves the fast path byte-for-byte
  // unchanged.
#if V8_ENABLE_GREEN_THREADS
  static void SetGreenHeapConcurrentAllocationEnabled(v8::Isolate* isolate, bool enabled);

  // Set whether concurrent JIT compilation is suspended for the green isolate.
  //
  // True-parallel body execution (GT_M2_PARALLEL=1) runs the same body on
  // multiple OS workers with no CoW view active, so GreenThreadCOWSuspendedConcurrentGC()
  // (ActiveCount() != 0) is false and V8's concurrent Turbofan/Maglev/batch
  // compilers would compile against the heap as the workers rewrite it in place
  // -- torn SharedFunctionInfo / feedback-vector / Map reads that land on a dead
  // pointer (EXC_BAD_ACCESS at 0xbeeddead) on the next deopt. The green runtime
  // sets this true around the concurrent body window and false after the workers
  // join, forcing those compilation paths to run synchronously on the mutator
  // thread where the heap view is always consistent. Complements the young-gen
  // allocation CAS gate (SetGreenHeapConcurrentAllocationEnabled), which makes
  // the heap pointers themselves disjoint; this makes the compiled code and the
  // IC/map state those pointers feed consistent.
  static void SetConcurrentCompilationSuspended(bool suspended);
#endif  // V8_ENABLE_GREEN_THREADS

  // Clone an object into the GT's heap view.
  // If gt is null, returns the original pointer unchanged.
  static Address Clone(Isolate* isolate, GreenThread* gt, Address obj_ptr);

  // Write barrier: called before a store to a potential COW object.
  // If the target hasn't been cloned, clones it and returns the copy address.
  // If it's already cloned, returns the copy address (dirty set is queued).
  // If COW is not enabled, returns the original pointer.
  static Address CowCheckBeforeStore(Isolate* isolate, Address obj_ptr);

  // Read barrier: called before a load from a potential COW object.
  // Returns the copy address if one exists, otherwise the original.
  static Address COWCheckAndRedirect(Address obj_ptr);

  // Redirect all GT stack/register references from old_ptr to new_ptr.
  // Called after cloning to keep the GT's stack/registers consistent.
  static void RedirectReferences(GreenThread* gt, Address old_ptr,
                                 Address new_ptr);

  // Scan all copied objects and visit them as GC roots.
  // Used during V8's GC mark phase.
  static void VisitAsRoots(GreenThread* gt, RootVisitor* visitor);

  // Flush the dirty set: redirect all queued writes to point at their copies.
  // Called at yield points and context switches.
  static void FlushGreenThreadDirtySet(GreenThread* gt);

  // Get the number of cloned objects for the given GT.
  static int ClonedObjectCount(GreenThread* gt);

  // Check if COW cloning is active in any green thread.
  static uint32_t ActiveCount() {
    return g_cow_active_count.load(std::memory_order_acquire);
  }

#if V8_ENABLE_GREEN_THREADS
  // Recursive mutex serializing access to V8 shared-writer internal state
  // (Map slack-tracking, JSDispatchTable code/tiering entries, feedback
  // vectors) while a CoW view is active. Recursive re-entrancy avoids
  // self-deadlock. Process-global (not per-isolate) matches the process-global
  // g_cow_active_count and one green isolate per process.
  static base::RecursiveMutex& SharedStateMutex();
  // True during an opt-in TRUE-CPU-PARALLEL body dispatch (see
  // GreenThreadCowSharedWriterScope). When set, the whole-body writer scope at
  // GreenJSEntry becomes a no-op so N allocation-light, already-optimized bodies
  // run concurrently on the OS worker pool; the narrow per-op writer scopes
  // (Map/feedback/JSDispatchTable) still lock for any body that compiles or
  // allocates. The embedding sets/clears this ONLY around dispatches it knows to
  // be safe (the parallel-execution discriminator), so allocation-heavy
  // concurrent suites stay serialized and correct.
  static void SetParallelBodyDispatchActive(bool active);
  static bool ParallelBodyDispatchActive();
#endif  // V8_ENABLE_GREEN_THREADS

  // --- All-mutator stop-the-world GC handshake (invariant I3) ---
  //
  // PollForGCRequest is called from a mutator safepoint (GreenScheduler::Yield,
  // step 1) and -- in the forced-GC test -- from the worker entry boundary
  // (item 2). If a stop-the-world GC has been requested (gc_requested) and one
  // is already in progress (gc_in_progress), it publishes this green thread's
  // current roots (saved registers, stack, HeapRefTable) via the canonical
  // RedirectWordsLocked machinery, records the "published + parked" flag, and
  // suspends the calling worker on its per-GT condition variable until the
  // coordinator resumes it. Returns true iff the worker parked (it must not
  // continue executing); false if there is nothing to do. A plain
  // `if (gc_requested && gc_in_progress) { publish(); park(); return true; }`.
  //
  // RequestAllMutatorsStop is the coordinator entry: it release-stores
  // gc_requested, snapshots the active COW green threads under the same mutex
  // the GC roots provider walks, and writes the snapshot to *out_suspended /
  // *out_n. The quiescence it establishes REPLACES the old
  // GreenThreadCOWSuspendedConcurrentGC suppression. Collect the heap while
  // gc_requested stays set, then call ResumeAllMutators to clear the request,
  // notify every parked worker, and restore the concurrent-GC gate.
  //
  // gc_in_progress is a separate atomic guard so a second coordinator blocks
  // and the bytecode poll cannot recurse into the publish path mid-collection.
#if V8_ENABLE_GREEN_THREADS
  static bool PollForGCRequest(GreenThread* gt, Isolate* isolate);
  static void RequestAllMutatorsStop(GreenThread*** out_suspended, int* out_n);
  static void ResumeAllMutators();
  // True while a coordinator is between RequestAllMutatorsStop and
  // ResumeAllMutators. Gated so the poll does not act outside a real GC.
  static bool GCInProgress();
  // Spin-join every active COW view until it has published + parked (gc_parked),
  // or until timeout_ms elapses (returns false). This is the quiescence barrier
  // that REPLACES GreenThreadCOWSuspendedConcurrentGC: the collector must not
  // start until every mutator is parked with roots published, so no worker is
  // mid-body mutating the state the GC walks.
  static bool WaitForAllMutatorsParked(int timeout_ms);
#endif  // V8_ENABLE_GREEN_THREADS

  // Number of times CowCloneForGeneratedCode was reached with COW active
  // (i.e. a real TurboFan kArchStoreWithWriteBarrier store consulted the
  // COW hook). Tests snapshot this counter around JIT stores to prove the
  // JIT→COW boundary is reached; Reset zeros it between tests.
  static uint64_t CowCloneForGeneratedCodeCallCount();
  static void ResetCowCloneForGeneratedCodeCallCount();

  // COW write barrier hook: check and rewrite the slot with the COW copy.
  // Called from CombinedWriteBarrierInternal (in heap-write-barrier-inl.h).
  // The host parameter is used for provenance filtering: COW only fires
  // when the host is a user-level mutable container (JSObject, JSArray,
  // JSProxy). This prevents crashes caused by V8 internal writes
  // (to maps, descriptors, SharedFunctionInfo, etc.) triggering deep
  // clone attempts on uninitialised/unaligned internal structures.
#if V8_ENABLE_GREEN_THREADS
  // Called from TurboFan-generated code to clone a heap object pointer
  // for the current green thread's isolated heap view.
  // Takes a tagged pointer, returns a tagged copy address (or original
  // if not cloned). The caller is responsible for writing the result
  // back to the store slot.
  static Address CowCloneForGeneratedCode(Address tagged_addr);
#endif  // V8_ENABLE_GREEN_THREADS

  static void COWWriteBarrierCheck(Tagged<HeapObject> host,
                                   Tagged<HeapObject> value,
                                   HeapObjectSlot slot);

 private:
  static std::atomic<uint32_t> g_cow_active_count;

  // Backing for SetParallelBodyDispatchActive(). Process-global (one green
  // isolate per process), matching g_cow_active_count. Defined in green-thread-cow.cc.
  static std::atomic<bool> g_parallel_body_dispatch_;

  // Counts CowCloneForGeneratedCode entries past the fast-path guards.
  static std::atomic<uint64_t> g_cow_generated_code_calls;
};

#if V8_ENABLE_GREEN_THREADS
// Serializes access to V8 shared-writer internal state (Map slack-tracking,
// JSDispatchTable code/tiering entries, feedback vectors) so invariant I4 holds
// under true multi-worker concurrency. No-op when no CoW view is active;
// recursive-mutex backed so it can be taken re-entrantly.
class GreenThreadCowSharedWriterScope {
 public:
  GreenThreadCowSharedWriterScope() {
    // During an opt-in true-parallel dispatch the whole-body serialize is
    // skipped even though a CoW view is active (ActiveCount() > 0): the
    // dispatched bodies are known to touch no shared-writer state.
    if (GreenThreadCow::ParallelBodyDispatchActive()) {
      locked_ = false;
      mutex_ = nullptr;
    } else if (GreenThreadCow::ActiveCount() > 0) {
      mutex_ = &GreenThreadCow::SharedStateMutex();
      mutex_->Lock();
      locked_ = true;
    } else {
      locked_ = false;
      mutex_ = nullptr;
    }
  }

  ~GreenThreadCowSharedWriterScope() {
    if (locked_) mutex_->Unlock();
  }

  GreenThreadCowSharedWriterScope(const GreenThreadCowSharedWriterScope&) =
      delete;
  GreenThreadCowSharedWriterScope& operator=(
      const GreenThreadCowSharedWriterScope&) = delete;

 private:
  bool locked_;
  base::RecursiveMutex* mutex_;
};
#endif  // V8_ENABLE_GREEN_THREADS

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_GREEN_THREAD_COW_H_

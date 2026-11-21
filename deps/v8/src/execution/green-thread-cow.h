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

  // Counts CowCloneForGeneratedCode entries past the fast-path guards.
  static std::atomic<uint64_t> g_cow_generated_code_calls;
};

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_GREEN_THREAD_COW_H_

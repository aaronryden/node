// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread-cow.h"
#include "src/execution/green-thread-scheduler.h"
#include "src/execution/isolate.h"

#include <cstdlib>
#include <pthread.h>

namespace v8 {
namespace internal {

// Run the green thread's entry function on its own stack.
//
// The entry function is a state machine driven by iteration_count_.
// iteration_count==1 → Phase 1 (READ + YIELD)
// iteration_count>=2 → Phase 2 (WRITE + complete)
//
// RunGreenThread calls the entry once. If the entry yields, Yield()
// calls the next thread's entry directly (recursive call on the same
// stack). When Yield() returns, the entry function just returns.
// The scheduler later calls RunGreenThread again for the same thread
// with a higher iteration_count to run Phase 2.
//
// @param gt The green thread to execute.
// @sideEffects Modifies the green thread's iteration_count and state.
void RunGreenThread(GreenThread* gt) {
  if (!gt || gt->pc_raw() == kNullAddress) {
    gt->set_state(GreenThreadState::kComplete);
    return;
  }

  Isolate* isolate = gt->isolate();
  SetCurrentGreenThread(gt);
  if (::getenv("GT_COW_DEBUG"))
    ::fprintf(stderr, "[GTDBG-ENTRY] RunGreenThread set CurrentGreenThread=%p os=%ld\n", (void*)gt, (long)pthread_self());
  gt->increment_iteration_count();

  // Entry function runs to completion (or yields, which calls
  // entry(next) recursively and returns).
  typedef void (*EntryFunc)(Isolate*, void*);
  EntryFunc entry = reinterpret_cast<EntryFunc>(gt->pc_raw());
  entry(isolate, gt);
}

}  // namespace internal
}  // namespace v8

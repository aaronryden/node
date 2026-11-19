// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <cstdio>

#include "green_threads_gc.h"

// This is the single translation unit that includes V8's heap/incremental-
// marking internals, so it alone pulls V8's tracing/trace-event.h (via
// gc-tracer.h) and never node's tracing/trace_event.h. See green_threads_gc.h
// for the full rationale for keeping these headers out of green_threads.cc.
#include "src/execution/isolate.h"
#include "src/heap/heap.h"
#include "src/heap/incremental-marking.h"

namespace node {
namespace vi = v8::internal;

// Constructing GreenGCPause snapshots the current incremental-marking state and
// stops marking; the destructor resumes it only if it had been running. This
// freezes the green heap across the green body's re-parse + execution (see
// RelocateFunction in green_threads.cc) so a factory allocation inside
// Compiler::Compile cannot relocate/move the JSFunction the compiler holds in a
// DirectHandle / EnsureClosureFeedbackCellArray, which would leave a dangling
// pointer that aborts in SharedFunctionInfo::GetActiveBytecodeArray. Green
// bodies only ever run on the embedding thread, so no worker is waiting to
// resume marking once this scope exits.
GreenGCPause::GreenGCPause(v8::Isolate* iso)
    : heap_(reinterpret_cast<vi::Isolate*>(iso)->heap()),
      was_running_(heap_ != nullptr &&
                    heap_->incremental_marking()->Stop()) {}

GreenGCPause::~GreenGCPause() {
  if (heap_ != nullptr && was_running_) {
    heap_->StartIncrementalMarkingOnInterrupt();
  }
}

}  // namespace node

// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GREEN_THREADS_GC_H_
#define GREEN_THREADS_GC_H_

#include <v8.h>

namespace node {

// A forward declaration lives here only so green_threads.cc can name
// GreenGCPause without itself pulling V8's heap/incremental-marking internals.
// The class body + definitions live in green_threads_gc.cc, the single
// translation unit that is allowed to include V8's heap headers.
//
// Why the split: green_threads.cc must include V8 internal headers (objects.h,
// green-thread-cow.h, ...) and it also needs node's per_process::v8_platform,
// which is pulled in via env-inl.h -> node_internals.h -> tracing/trace_event.h.
// V8's own heap.h transitively drags in V8's tracing/trace-event.h (via
// gc-tracer.h). That header and node's tracing/trace_event.h declare the same
// symbols (CategoryGroupEnabledFlags and the TRACE_EVENT* macros) but under
// different include guards, so having both in one translation unit collides.
// Moving GreenGCPause into green_threads_gc.cc keeps heap.h entirely out of
// green_threads.cc and therefore keeps V8's trace-event.h out of it as well.
class GreenGCPause final {
 public:
  // Suspend the isolate's incremental marking for the enclosing scope and
  // resume it when the scope exits. See green_threads_gc.cc for the full
  // rationale (the green-heap fork has no compile-time GC defer, so a factory
  // allocation inside Compiler::Compile could relocate the JSFunction the
  // compiler holds in a non-rooting handle).
  explicit GreenGCPause(v8::Isolate* iso);
  ~GreenGCPause();

  GreenGCPause(const GreenGCPause&) = delete;
  GreenGCPause& operator=(const GreenGCPause&) = delete;

 private:
  v8::internal::Heap* heap_;
  bool was_running_;
};

}  // namespace node

#endif  // GREEN_THREADS_GC_H_

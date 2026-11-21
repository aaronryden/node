// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef GREEN_THREADS_GC_H_
#define GREEN_THREADS_GC_H_

#include <functional>
#include <string>
#include <vector>

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

// Discover the names of the variables the green body `emb_fn` captures from its
// enclosing lexical scopes (its free variables), innermost-first.
//
// Defined in green_threads_gc.cc -- not green_threads.cc -- because it walks the
// internal scope-info / context chain via js-function.h, which drags V8's
// heap.h + V8 tracing/trace-event.h in. Keeping that here (the single TU allowed
// to include V8's heap headers) leaves heap.h, and therefore V8's trace-event.h,
// out of green_threads.cc, which must keep node's tracing/trace_event.h (see
// GreenGCPause above). Returns plain std::string names only; all V8-internal
// work -- reinterpretting the public handle, the context walk, and String
// decoding -- happens entirely inside green_threads_gc.cc.
std::vector<std::string> CaptureGreenClosureVars(v8::Isolate* emb_isolate,
                                                 v8::Local<v8::Function> emb_fn);

// Read the current values of the free variables `CaptureGreenClosureVars`
// discovered for the green body `emb_fn` -- by those exact names -- from the
// embedding closure-context chain, invoking `yield(name, value)` for each as a
// public `v8::Local<v8::Value>` on the embedding isolate.
//
// A closure's free variables are the inlined locals of its context; the slot
// index `var` of such a name (ScopeInfo::ContextInlinedLocalName) is exactly the
// context-slot index (Context::Get) holding its bound value, so the value is
// read directly with the internal Context::Get and bridged to a public handle
// with Utils::Convert (the only internal->public bridge, no type check). The
// value is yielded while the enclosing HandleScope keeps it rooted -- the caller
// MUST consume it within the callback (e.g. structured-capture it) because no
// Local is returned past this point. Captures carrying std::string + heap-owned
// GreenIR pointers between the embedding and green passes keep no handle across
// the isolate boundary.
void CaptureGreenClosureValues(
    v8::Isolate* emb_isolate, v8::Local<v8::Function> emb_fn,
    const std::function<void(const std::string&, v8::Local<v8::Value>)>& yield);

// Forced GC modes for the M2 forced-GC acceptance test (item 2). The knob is
// read from the GT_M2_FORCE_GC environment variable so the existing probe path
// is byte-for-byte unchanged when it is unset.
enum class M2ForcedGCMode {
  kNone,     // unset: never force a collection (default, unchanged behavior)
  kYoung,    // force one young (all-available) collection
  kFull,     // force one full collection
  kBoth,     // force young then full (the 4-workers-active acceptance gate)
};

// Parse the GT_M2_FORCE_GC env var ("young"|"full"|"both"; unset -> kNone).
M2ForcedGCMode M2ParseForcedGCEnv();

// Force a young and/or full collection on the green isolate, gated by `mode`.
//
// The caller (M2Probe) has already invoked GreenThreadCow::RequestAllMutatorsStop
// (which release-stores gc_requested and snapshots the active COW views) and, in
// the forced-GC test, the worker threads are parked at their entry safepoint.
// That quiescence is what makes a synchronous full collection safe here: the
// collector walks only published, reported roots and no mutator is mid-body.
//
// young|both -> Heap::CollectAllAvailableGarbage(kTesting);
// full|both  -> Heap::CollectAllGarbage(kNoFlags, kTesting, kGCCallbackFlagForced).
// Collecting on the green isolate's own heap (via vi::Isolate::heap()) keeps this
// in the single TU that owns V8's heap headers, so heap.h stays out of
// green_threads.cc (see the GreenGCPause rationale above).
void M2ForceGC(v8::Isolate* green_isolate, M2ForcedGCMode mode);

}  // namespace node

#endif  // GREEN_THREADS_GC_H_

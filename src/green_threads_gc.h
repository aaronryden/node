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

// Read the current values of the free variables
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

// Whether `emb_fn` is a JavaScript async function (async fn / async arrow /
// async generator / async concise method). Reads the SharedFunctionInfo's
// FunctionKind so async bodies can be routed to the embedding realm (which owns
// Node's async I/O bindings) without running them first -- see green_threads_gc.cc.
bool GTIsAsyncFunction(v8::Isolate* emb_isolate,
                       v8::Local<v8::Function> emb_fn);

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

// Enter / leave a GC-free allocation window on the green isolate. While the
// window is entered the heap's always_allocate counter is non-zero, so the young
// scavenger, mark-compact, and incremental marking all skip and the heap merely
// grows instead of collecting. That makes every fast-path map / dictionary-
// fallback / heap-object-field read a JIT body performs mid-body observe a
// consistent, non-reclaimed heap -- which is the fix for the parallel-allocation
// SIGSEGV (a concurrent collection reclaimed a live pointer mid-body). Like
// M2ForceGC, these live here because they touch Heap's always_allocate counter,
// which needs heap.h; keeping heap.h out of green_threads.cc avoids the V8
// trace-event.h vs node tracing/trace_event.h collision.
void GreenEnterAlwaysAllocation(v8::Isolate* green_isolate);
void GreenLeaveAlwaysAllocation(v8::Isolate* green_isolate);

// Debug-only snapshot of the executing OS thread's green-thread execution-state
// context (reads the per-worker ThreadLocalTop block). Fields: `is_green` =
// whether the isolate is in green_thread_mode_; `main_iso` = the shared
// main-thread LocalIsolate; `main_heap` = the shared main-thread LocalHeap;
// `tblock` = this thread's ThreadLocalTop block. Lets an embedding-thread run
// (warmup) and a worker run be compared A/B on the very execution-state base a
// misaligned read points back to. Declared here (not in green_threads.cc) only
// because it needs the full vi::Isolate type, which is kept out of green_threads.cc
// to avoid V8's tracing/trace-event.h colliding with node's tracing/trace_event.h.
struct GTDebugCtx {
  bool is_green;
  void* main_iso;
  void* main_heap;
  void* tblock;
};
GTDebugCtx GTGreenThreadDebugCtx(v8::Isolate* iso);

// The current isolate for the executing OS thread (vi::Isolate::TryGetCurrent),
// or null. Declared here (not in green_threads.cc) because it needs the full
// vi::Isolate type; lets GreenJSEntry confirm the body runs on data->isolate
// (green) rather than some other isolate.
void* GTGreenThreadCurrentIsolate();

// Rich execution-state snapshot for the executing OS thread: current isolate
// (TryGetCurrent), whether it is a green-thread isolate, the current
// PerIsolateThreadData and the isolate it is bound to (and whether that binding
// is green), the thread-local-top base, and whether that base is the central
// IsolateData block (is_central) rather than the per-worker block. Lets GreenJSEntry
// confirm the body executes on the green isolate's per-worker execution state
// instead of the central/embedding block.
struct GTExecSnap {
  void* cur_iso;  // vi::Isolate::TryGetCurrent()
  int cur_green;  // cur_iso != null && cur_iso->is_green_thread_mode()
  void* cur_ptd;  // cur_iso ? cur_iso->CurrentPerIsolateThreadData() : null
  void* ptd_iso;  // cur_ptd ? cur_ptd->isolate() : null
  int ptd_green;  // ptd_iso != null && ptd_iso->is_green_thread_mode()
  void* tlt;  // cur_iso ? cur_iso->thread_local_top() : null
  int central;  // tlt points at the central IsolateData thread_local_top_
};
GTExecSnap GTGreenThreadExecSnapshot();

// Env-gated (GT_GT_DUMP) snapshot of the executing OS thread's ThreadLocalTop
// entry-state -- the fields a Maglev body reads on first JS entry (c_entry_fp_,
// context_, topmost_script_having_context_, exception_, handler_, js_entry_sp_,
// central/secondary stack limits, is_on_central_stack_flag_) plus this isolate's
// root register base (isolate_root()) and young-gen heap top. Lets the warmup
// (central) block and a fresh per-worker ExecutionState block be compared A/B in
// ONE run -- which is exactly what exposes why a shared-heap worker body that
// allocates + keyed-stores crashes while the identical warmup run does not.
// Declared here (not green_threads.cc) because it reads vi::Isolate /
// vi::ThreadLocalTop internals.
void GTGreenThreadEntryDump(v8::Isolate* iso);
}  // namespace node

#endif  // GREEN_THREADS_GC_H_

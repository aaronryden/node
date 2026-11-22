// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Green threads integration for Node.js.
//
// This native bridge exposes the V8 green-thread scheduler (cooperative,
// single-OS-thread "Model A" with recursive entry calls) together with its
// copy-on-write (COW) heap isolation to user JavaScript through
// `require('green_threads')`.
//
// Execution model
// ---------------
// Green threads run cooperatively on the embedding thread via recursive entry
// calls. Each green thread has an entry point `void(Isolate*, void*)` that is
// a state machine driven by `GreenThread::iteration_count()`:
//
//   * iteration_count == 1  -> the green thread runs its body once.
//   * iteration_count >= 2  -> a no-op re-entry; the body already completed at
//                              iteration 1, so the scheduler only re-enters a
//                              thread after it yields.
//
// `GreenScheduler::Yield()` saves the current green thread, runs the next one
// recursively on the same C++ stack, and returns when control comes back.
// `GreenScheduler::ScheduleLoop()` drains every scheduled green thread to
// completion. Because context switching happens by recursion (never by
// setjmp/longjmp), RAII `HandleScope`/`Context::Scope` nesting naturally tracks
// which green thread is on top.
//
// Copy-on-write
// -------------
// Each green thread owns a `HeapRefTable` mapping main-heap object addresses to
// its own private copies. The mapping is populated lazily: the first time a
// green thread stores a reference to a shared heap object into one of its own
// objects, the COW write barrier (wired into `CombinedWriteBarrierInternal`)
// clones the referenced object for that green thread and records the clone in
// its table. Objects the main thread never writes to stay shared, and main-heap
// objects mutated inside a green thread are invisible to the main thread.
//
// The public helpers `addrOf` and `cowCloneOf` let JavaScript observe this:
//   * `addrOf(obj)`       returns the untagged heap address of `obj`.
//   * `cowCloneOf(obj)`   returns the current green thread's private copy of
//                         `obj` (or undefined if it has not cloned it yet).

#include <cstdint>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <pthread.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "node_binding.h"
#include "util.h"

#include "v8.h"

#include "src/common/ptr-compr.h"
#include "src/execution/green-thread.h"
#include "src/execution/green-thread-cow.h"
#include "src/execution/green-thread-scheduler.h"
#include "src/execution/heap-ref-table.h"

// NOTE: green_threads.cc deliberately does NOT include V8 object headers
// (contexts.h, scope-info.h, js-function.h, string.h). Those pull V8's
// heap.h -> tracing/trace-event.h, which collides with node's tracing/trace_event.h
// (pulled below via env-inl.h -> node_internals.h). The internal closure
// scope-walk that needs them now lives in green_threads_gc.cc (the single TU
// allowed to include V8's heap headers), exposed here as the public-API
// CaptureGreenClosureVars(). See green_threads_gc.h for the full rationale.

// GreenGCPause (GC suspension across the green-compile + run window) is defined
// in green_threads_gc.cc, a dedicated V8-context translation unit. Declaring it
// here would pull V8's tracing/trace-event.h (via heap.h -> gc-tracer.h) into
// the same TU as node's tracing/trace_event.h (pulled below via env-inl.h for
// per_process::v8_platform); those two headers define the same symbols under
// different include guards and collide. See green_threads_gc.h for details.
#include "green_threads_gc.h"

// Environment / SnapshotData (and the inline accessors used below) live here so
// the green isolate can share the embedding isolate's V8 snapshot blob.
#include "env.h"
#include "env-inl.h"
#include "node_internals.h"
#include "node_snapshot_builder.h"
#include "node_v8_platform-inl.h"

namespace node {
// Forward declaration of the green `g` shell installer, defined below in node
// scope. LiftEmbeddingGlobals (in the anonymous namespace) calls it, so this
// lives here in node scope to match the definition's linkage/scope.
static void InstallGreenGlobalsShell(v8::Local<v8::Context> green_ctx);

namespace {

namespace vi = v8::internal;

using v8::Array;
using v8::BigInt;
using v8::Context;
using v8::Function;
using v8::FunctionCallbackInfo;
using v8::HandleScope;
using v8::EscapableHandleScope;
using v8::FunctionTemplate;
using v8::Isolate;
using v8::Local;
using v8::MaybeLocal;
using v8::Boolean;
using v8::Number;
using v8::Object;
using v8::Persistent;
using v8::Script;
using v8::String;
using v8::Undefined;
using v8::Value;

using vi::Address;  // v8::internal::Address (untagged heap pointer)

// Per-green-thread JavaScript invocation state. A single instance is shared
// between the scheduler's C++ entry point and the running spawnAll call.
struct GreenCallData {
  Isolate* isolate = nullptr;
  Persistent<Context> context_p;  // The context the body runs in.
  Persistent<Function> fn_p;      // The user's green-thread body.
  // Isolate that owns fn_p's handle (the embedding isolate in every mode). The
  // body is the embedding Function itself, so its Persistent can only be read
  // back on that isolate -- GreenJSEntry recovers it there before switching to
  // the green isolate/context for execution.
  Isolate* fn_isolate = nullptr;
  // Arguments passed to the body. Each is stored as its own Persistent<Value>
  // cell, wrapped in a unique_ptr so the argument vector can grow (a
  // Persistent<Value> is non-copyable and cannot live in a std::vector by
  // value). Storing each argument as a rooted Persistent cell -- rather than as
  // a transient Local -- is what keeps green-isolate argument values alive from
  // the copy that re-creates them in RunTasks until GreenJSEntry hands them to
  // the body, across the whole ScheduleLoop run.
  std::vector<std::unique_ptr<Persistent<Value>>> args_p;
  Persistent<Value> result_p;   // Body return value.
  // First thrown body's captured exception, rooted on the executing (green)
  // isolate. Captured in GreenJSEntry -- while the TryCatch is still in scope,
  // before its Exit restores the isolate's pending state -- so RunTasks can
  // carry it onto the embedding isolate and re-throw it. Clearing the pending
  // state after capture is what stops the task's exception leaking to a sibling
  // task on the shared green heap (and stops the isolate reporting it as
  // "Uncaught").
  Persistent<Value> exc_p;
  // Disjoint per-worker allocation marker (a small rooted object on the green
  // heap) used by the forced-GC acceptance test (item 2): it is rooted for the
  // whole probe so the forced GC -- which runs while this worker is parked --
  // cannot reclaim it, and post_gc_consistent verifies it survived intact.
  Persistent<Value> marker_p;
  bool threw = false;
  // Thrown-exception metadata. When a body throws a non-array Error-like value,
  // GreenJSEntry reads its resolved `name`/`message` here -- the real prototype
  // is intact at throw time, whereas the structured copy carries no prototype
  // and drops an Error's non-enumerable `message`, so the exact error class is
  // reconstructed faithfully on the embedding isolate in RunTasks. `exc_p` still
  // roots the raw thrown value, used for the non-error copy path.
  bool exc_is_error = false;
  std::string exc_name;
  std::string exc_msg;
  vi::GreenThread* gt = nullptr;  // Underlying green thread descriptor.
};

// Registry from green thread to its call data. The scheduler entry point is
// keyed by the GreenThread pointer the scheduler hands back on every call.
std::unordered_map<vi::GreenThread*, GreenCallData*> g_gt_registry;

// Returns the green address (untagged heap pointer) of a heap object handle,
// matching the key form the COW write barrier stores in each HeapRefTable
// (`Tagged<HeapObject>::ptr() - kHeapObjectTag`).
//
// In this build pointers are uncompressed and V8_ENABLE_DIRECT_HANDLE is off,
// so Local<Object>::operator->() returns a pointer to the handle's internal
// Address slot rather than to the object. The slot holds the tagged heap
// pointer (object address + kHeapObjectTag); dereference it and strip the tag
// to recover the exact untagged address the barrier keys on.
inline Address UntaggedObjectAddress(Local<Object> obj) {
  Address tagged = *reinterpret_cast<Address*>(obj.operator->());
  return tagged - static_cast<Address>(vi::kHeapObjectTag);
}

// ====================================================================
// Shared-green-heap mode (P0)
// --------------------------------------------------------------------
// When the env var GT_SHARED_HEAP is set, green threads execute on a
// dedicated second V8 isolate with its own heap, so a failure inside a
// green body cannot corrupt Node's main isolate; the embedding isolate
// keeps its own context and event loop. Task bodies are re-parsed onto
// the green isolate (so the running code is green Code), task arguments
// are copied into the green heap (so bodies mutate private copies and
// the shared `parent` in the harness is not corrupted), and results are
// copied back out. Every path here is opt-in: the default is the existing
// single-isolate behavior and is left byte-for-byte unchanged.
// ====================================================================

static bool GreenHeapEnabled() {
  return ::getenv("GT_SHARED_HEAP") != nullptr;
}

// Green isolate (created lazily, tied to the process lifetime). It joins
// Node's own isolate group so it shares the string table, and gets its own
// array-buffer allocator so green-heap corruption stays contained.
static Isolate* g_green_isolate = nullptr;

// Green context (created lazily). A fresh context starts with the standard
// builtins; embedding-defined helper functions (e.g. `fib`) are lifted by
// re-parsing their source, skipping anything that is not real JS (native
// bindings synthesize a non-compilable source, which we detect and skip).
static Persistent<Context> g_green_ctx;

static Isolate* GetGreenIsolate(Isolate* embedding) {
  if (g_green_isolate != nullptr) return g_green_isolate;

  // A raw v8::Isolate::New bypasses Node's platform registry: Isolate::Init ->
  // NodePlatform::GetForegroundTaskRunner -> ForIsolate hits a null-pointer
  // assertion because the isolate was never registered on the platform.
  // Reproduce Node's own isolate-creation ordering instead (see NewIsolate /
  // RAIIIsolateWithoutEntering in util.cc): allocate inside the embedding
  // isolate's group, register on the platform BEFORE Initialize (so the new
  // isolate can reach the platform during its own init), then Initialize.
  Isolate* isolate = Isolate::Allocate(embedding->GetGroup());
  if (isolate == nullptr) return nullptr;

  per_process::v8_platform.Platform()->RegisterIsolate(
      isolate, uv_default_loop());

  Isolate::CreateParams params;
  params.array_buffer_allocator =
      v8::ArrayBuffer::Allocator::NewDefaultAllocator();
  // The process runs with a shared read-only heap, and V8 requires every
  // isolate in the process to be initialized from the *same* snapshot blob --
  // Node copies first_params.snapshot_blob onto every isolate for exactly this
  // reason (see NewIsolate in environment.cc). A green isolate created with no
  // snapshot has no read-only-heap builtins or interpreter bytecode, so the
  // first body re-parse after any prior run produced a JSFunction whose
  // SharedFunctionInfo resolved to a bogus pointer inside PostInstantiation
  // (SIGABRT in SharedFunctionInfo::GetActiveBytecodeArray). Copy the embedding
  // isolate's V8 snapshot so the green isolate shares the readonly heap exactly
  // like every other Node isolate. SnapshotData is owned by IsolateData for the
  // whole process lifetime, matching the green isolate's own lifetime.
  {
    Environment* env = Environment::GetCurrent(embedding);
    const SnapshotData* snap =
        (env != nullptr) ? env->isolate_data()->snapshot_data() : nullptr;
    if (snap != nullptr && snap->v8_snapshot_blob_data.raw_size > 0) {
      params.snapshot_blob = &snap->v8_snapshot_blob_data;
    }
  }
  // A snapshotized isolate also needs its api_external_references populated:
  // during Isolate::Initialize V8 deserializes the read-only heap and, for every
  // embedded reference, checks the isolate's external-reference table is present
  // (Check failed: isolate->api_external_references() != nullptr) before it can
  // walk microtask-queue/global setup. Node sets this on every isolate from the
  // process-wide binding registry (SnapshotBuilder::CollectExternalReferences).
  // Node registers every binding it needs at startup, so by the time the green
  // isolate is created lazily on first spawn the table is fully populated and is
  // a superset of the entries the snapshot requires -- exactly what Node owns on
  // the main isolate.
  {
    const std::vector<intptr_t>& ext =
        node::SnapshotBuilder::CollectExternalReferences();
    if (!ext.empty()) {
      params.external_references = ext.data();
    }
  }
  // Green threads run cooperatively on the embedding (single) thread via
  // recursive Model-A entry, so the green heap marks and sweeps incrementally
  // on that same thread (no concurrent worker).
  v8::CppHeapCreateParams green_heap_params{{}};
  green_heap_params.marking_support = cppgc::Heap::MarkingType::kIncremental;
  green_heap_params.sweeping_support = cppgc::Heap::SweepingType::kIncremental;
  params.cpp_heap = v8::CppHeap::Create(per_process::v8_platform.Platform(),
                                        green_heap_params)
                        .release();
  // This fork's isolated green heap drives GC from plain allocation pressure
  // with no compile-time GC defer, so a factory allocation inside
  // Compiler::Compile can relocate/collect a JSFunction the compiler holds in
  // a non-rooting DirectHandle. Generous headroom stops compiling a body from
  // tripping the allocation limit; the GreenGCPause below pauses incremental
  // marking across the compile+run window to handle an in-progress cycle.
  params.constraints.set_max_old_generation_size_in_bytes(256 * 1024 * 1024);
  params.constraints.set_max_young_generation_size_in_bytes(32 * 1024 * 1024);
  Isolate::Initialize(isolate, params);

  g_green_isolate = isolate;
  // Several OS worker threads will Enter/Exit this isolate concurrently, so mark
  // it green-thread mode to route each worker's EntryStackItem chain onto its own
  // PerIsolateThreadData (see Isolate::Enter/Exit).
  g_green_isolate->SetGreenThreadMode(true);
  return isolate;
}

// Process-exit hook: release the green isolate before V8::Dispose() runs.
//
// The green isolate shares the embedding isolate's isolate group (so it shares
// the string table and other group-wide state). IsolateGroup::ReleaseDefault,
// reached from V8::Dispose() at teardown, asserts that the default group still
// has exactly one reference -- the main isolate's. Because the green isolate
// adds a second reference, it must be disposed before V8::Dispose() or that
// check fails with "group->reference_count_.load() == 1". V8 documents that
// V8::Dispose() must only be called after every isolate has been deleted, and
// Environment::AtExit callbacks run at FreeEnvironment, before
// TearDownOncePerProcess()/V8::Dispose().
static void DisposeGreenIsolate(void* /*arg*/) {
  if (g_green_isolate != nullptr) {
    if (!g_green_ctx.IsEmpty()) {
      g_green_ctx.Reset();
    }
    // Tear the green isolate down through the platform, not via a raw
    // Isolate::Dispose(). Isolate::Dispose() leaves the isolate registered on
    // the platform, so its PerIsolatePlatformData is destroyed at
    // NodePlatform::Shutdown()'s per_isolate_.clear() with flush_tasks_ still
    // set, tripping the ~PerIsolatePlatformData CHECK(!flush_tasks_). Routing
    // through DisposeIsolate runs UnregisterIsolate -> Shutdown, which nulls
    // flush_tasks_ first; Deinitialize + Isolate::Free replace the old Dispose.
    // The group-reference-count constraint still holds: this AtExit callback
    // runs at FreeEnvironment, before TearDownOncePerProcess()/V8::Dispose().
    per_process::v8_platform.Platform()->DisposeIsolate(g_green_isolate);
    g_green_isolate = nullptr;
  }
}

// GreenGCPause is defined in green_threads_gc.cc; see that file for the full
// rationale. It suspends the green isolate's incremental marking for the
// enclosing scope so the green heap cannot relocate/move the JSFunction the
// compiler holds in a non-rooting DirectHandle across the re-parse + run window.

static void LiftEmbeddingGlobals(Isolate* embedding,
                                  Local<Context> embedding_ctx,
                                  Local<Context> green_ctx) {
  // Split into two isolated passes because the embedding global cannot be read
  // while the green isolate is current (and vice versa): pass 1 reads each
  // embedding-global function's name + source into plain C++ while the
  // embedding isolate is current, and pass 2 re-parses and installs each one
  // under its original name on the green context while the green isolate is
  // current. The captured name/source strings are the only data carried
  // between the passes. Native bindings are skipped: the fresh green context
  // already has every standard builtin, and their [native code] source is not
  // compilable.
  struct FnSpec {
    std::string name;
    std::string src;
  };
  std::vector<FnSpec> fns;
  {
    v8::Isolate::Scope isolate_scope(embedding);
    HandleScope scope(embedding);
    Local<Context> emb_ctx = embedding->GetCurrentContext();
    Local<Object> emb_global = embedding_ctx->Global();
    Local<Array> keys = emb_global->GetPropertyNames(emb_ctx).ToLocalChecked();
    for (uint32_t i = 0; i < keys->Length(); i++) {
      Local<String> name_key =
          Local<String>::Cast(keys->Get(emb_ctx, i).ToLocalChecked());
      Local<Value> value;
      if (!emb_global->Get(emb_ctx, name_key).ToLocal(&value)) continue;
      if (!value->IsFunction()) continue;
      String::Utf8Value name_utf8(embedding, name_key);
      std::string name(*name_utf8, name_utf8.length());
      Local<String> fn_src =
          Local<Function>::Cast(value)->ToString(emb_ctx).ToLocalChecked();
      String::Utf8Value src_utf8(embedding, fn_src);
      std::string src(*src_utf8, src_utf8.length());
      if (src.find("[native code]") != std::string::npos) continue;
      fns.emplace_back(FnSpec{std::move(name), std::move(src)});
    }
  }
  {
    v8::Isolate::Scope isolate_scope(g_green_isolate);
    HandleScope scope(g_green_isolate);
    Context::Scope ctx_scope(green_ctx);
    // Suspend green-isolate marking across the re-parse + run so GC cannot move
    // a JSFunction the compiler roots only in a DirectHandle (see GreenGCPause).
    GreenGCPause gc(g_green_isolate);
    for (const FnSpec& spec : fns) {
      std::string program = "(function(){ return " + spec.src + "; })();";
      MaybeLocal<Script> compiled = Script::Compile(
          green_ctx, String::NewFromUtf8(g_green_isolate, program.c_str(),
                                         v8::NewStringType::kNormal)
                          .ToLocalChecked());
      if (compiled.IsEmpty()) continue;
      MaybeLocal<Value> lifted = compiled.ToLocalChecked()->Run(green_ctx);
      if (lifted.IsEmpty()) continue;
      Local<String> name_key = String::NewFromUtf8(
          g_green_isolate, spec.name.c_str(), v8::NewStringType::kNormal,
          spec.name.length()).ToLocalChecked();
      green_ctx->Global()->Set(green_ctx, name_key, lifted.ToLocalChecked()).Check();
    }
  }
  // Provide the green `g` shell so a body that references the native binding as
  // a free variable resolves it (native/function-bearing bindings are skipped by
  // the closure lift -- see LiftCapturedValues -- because moving them across
  // isolates would share an embedding SFI, an F1 cross-heap hazard).
  InstallGreenGlobalsShell(green_ctx);
}

// Ensure the green context exists, creating it (and lifting embedding globals
// into it) on first use. The context is stored in the durable persistent
// g_green_ctx, so no returned handle can dangle: callers reference the context
// through a Persistent only. This function owns the create+lift window and roots
// the context in g_green_ctx before its own scope ends.
static void EnsureGreenContext(Isolate* embedding, Local<Context> embedding_ctx) {
  if (!g_green_ctx.IsEmpty()) return;
  v8::Isolate::Scope isolate_scope(g_green_isolate);
  HandleScope scope(g_green_isolate);
  Local<Context> ctx = Context::New(g_green_isolate);
  Context::Scope ctx_scope(ctx);
  LiftEmbeddingGlobals(embedding, embedding_ctx, ctx);
  g_green_ctx.Reset(g_green_isolate, ctx);
}

static void LiftCapturedValues(Isolate* emb_isolate,
                               Local<Function> emb_fn,
                               Local<Context> green_ctx);

// Relocate an embedding-isolate function onto the green isolate by
// re-parsing its source, so a green body runs as green Code and free
// variables resolve against the lifted green global.
// Returns the relocated body rooted in a Persistent on the green isolate. The
// caller owns this durable handle; no caller handle scope is required because
// the function is rooted inside this function before its own scope ends.
static std::unique_ptr<Persistent<Function>> RelocateFunction(Isolate* emb_isolate,
                                                              Local<Function> emb_fn) {
  // Capture the source on the embedding isolate.
  std::string src;
  {
    v8::Isolate::Scope isolate_scope(emb_isolate);
    HandleScope scope(emb_isolate);
    Local<Context> emb_ctx = emb_isolate->GetCurrentContext();
    Local<String> fn_src = emb_fn->ToString(emb_ctx).ToLocalChecked();
    String::Utf8Value src_utf8(emb_isolate, fn_src);
    src.assign(*src_utf8, src_utf8.length());
  }
  std::string program = "(function(){ return " + src + "; })();";
  // Compile + run on the green isolate in green_ctx -- the persistent context
  // that owns the lifted embedding globals (not a fresh per-function context,
  // which used to give every body a different native context and left the green
  // isolate's current context torn for the second body dispatched on it). Free
  // variables in the body therefore resolve against those lifted globals. The
  // result is rooted in a Persistent before this scope ends, so no Local
  // dangles.
  GreenGCPause gc(g_green_isolate);
  v8::Isolate::Scope isolate_scope(g_green_isolate);
  HandleScope scope(g_green_isolate);
  Local<Context> green_ctx = Local<Context>::New(g_green_isolate, g_green_ctx);
  Context::Scope ctx_scope(green_ctx);
  // Lift captured closure values (module-scope `const`s like `vals`; native
  // bindings like `g` are skipped) onto the green global by name, so the
  // reparsed body's free variables resolve. Runs inside the ambient
  // GreenGCPause above; pass 1 reads the embedding closure context, pass 2
  // GreenRebuilds structured values onto green.
  LiftCapturedValues(emb_isolate, emb_fn, green_ctx);
  Local<Script> script = Script::Compile(
      green_ctx, String::NewFromUtf8(g_green_isolate, program.c_str(),
                                     v8::NewStringType::kNormal)
                    .ToLocalChecked())
                              .ToLocalChecked();
  Local<Function> relocated =
      Local<Function>::Cast(script->Run(green_ctx).ToLocalChecked());
  return std::make_unique<Persistent<Function>>(g_green_isolate, relocated);
}

// ====================================================================
// Value round-trip across isolates
// --------------------------------------------------------------------
// A value produced on the green isolate (a body's return) or consumed on it
// (a spawn argument) cannot keep a handle on the other isolate, so it is
// moved in two steps: GreenCapture reads the source-isolate value into an
// allocation-free-of-8 (plain C++) GreenIR tree, and GreenRebuild rebuilds it
// on the destination isolate with neither isolate required to be current while
// the other is read. Cyclic structures are not round-tripped (the scenarios in
// this workspace are acyclic); a repeated reference simply re-captures the
// subtree.
// ====================================================================
struct GreenIR {
  enum Kind { kPrim, kArray, kObject } kind = kPrim;
  uint64_t addr = 0;  // Source-isolate address (structured values only).
  bool prim_is_undef_null = false;
  bool prim_is_boolean = false;
  bool prim_is_number = false;
  bool prim_is_string = false;
  bool prim_is_bigint = false;
  bool prim_bool = false;
  double prim_num = 0.0;
  std::string prim_str;     // prim_is_string
  int bigint_sign = 0;      // prim_is_bigint
  std::vector<uint64_t> bigint_words;  // prim_is_bigint (base-2^64 magnitude)
  std::vector<GreenIR> children;  // kArray: one per element; kObject: [key,val,...] pairs
};

// Recursively read `value` (on the isolate that owns `ctx`) into a GreenIR
// tree, caching structured values by source address so a repeated reference to
// the same object reuses its captured subtree.
static GreenIR* GreenCapture(Isolate* src, Local<Context> ctx, Local<Value> value,
                             std::unordered_map<uint64_t, GreenIR*>& by_addr) {
  if (value->IsUndefined() || value->IsNull())
    return new GreenIR{.kind = GreenIR::kPrim, .prim_is_undef_null = true};
  if (value->IsBoolean())
    return new GreenIR{.kind = GreenIR::kPrim, .prim_is_boolean = true,
                       .prim_bool = value->IsTrue()};
  if (value->IsNumber())
    return new GreenIR{.kind = GreenIR::kPrim, .prim_is_number = true,
                       .prim_num = Local<Number>::Cast(value)->Value()};
  if (value->IsString()) {
    String::Utf8Value utf8(src, value);
    return new GreenIR{.kind = GreenIR::kPrim, .prim_is_string = true,
                       .prim_str = *utf8};
  }
  if (value->IsBigInt()) {
    Local<BigInt> bi = value.As<BigInt>();
    int sign_bit = 0;
    int word_count = bi->WordCount();
    std::vector<uint64_t> words(static_cast<size_t>(word_count));
    bi->ToWordsArray(&sign_bit, &word_count, words.data());
    return new GreenIR{.kind = GreenIR::kPrim, .prim_is_bigint = true,
                       .bigint_sign = sign_bit, .bigint_words = std::move(words)};
  }

  Address addr = UntaggedObjectAddress(Local<Object>::Cast(value));
  uint64_t key = static_cast<uint64_t>(addr);
  auto it = by_addr.find(key);
  if (it != by_addr.end()) return it->second;

  GreenIR* ir;
  if (value->IsArray()) {
    ir = new GreenIR{.kind = GreenIR::kArray, .addr = key};
    Local<Array> arr = Local<Array>::Cast(value);
    const uint32_t len = arr->Length();
    ir->children.reserve(len);
    for (uint32_t i = 0; i < len; i++) {
      ir->children.push_back(
          *GreenCapture(src, ctx, arr->Get(ctx, i).ToLocalChecked(), by_addr));
    }
  } else {
    ir = new GreenIR{.kind = GreenIR::kObject, .addr = key};
    Local<Object> obj = Local<Object>::Cast(value);
    Local<Array> keys = obj->GetPropertyNames(ctx).ToLocalChecked();
    const uint32_t len = keys->Length();
    ir->children.reserve(len * 2);
    for (uint32_t i = 0; i < len; i++) {
      Local<String> key_str =
          Local<String>::Cast(keys->Get(ctx, i).ToLocalChecked());
      String::Utf8Value key_utf8(src, key_str);
      std::string key_value(*key_utf8, key_utf8.length());
      ir->children.push_back(
          GreenIR{.kind = GreenIR::kPrim, .prim_is_string = true, .prim_str = key_value});
      ir->children.push_back(
          *GreenCapture(src, ctx, obj->Get(ctx, key_str).ToLocalChecked(), by_addr));
    }
  }
  by_addr[key] = ir;
  return ir;
}

// Recreate a value on `dst` from a GreenIR tree on `ctx`.
static Local<Value> GreenRebuild(Isolate* dst, const GreenIR* ir, Local<Context> ctx) {
  switch (ir->kind) {
    case GreenIR::kArray: {
      Local<Array> arr = Array::New(dst, static_cast<uint32_t>(ir->children.size()));
      for (size_t i = 0; i < ir->children.size(); i++) {
        // These writes target freshly created arrays, so Set cannot fail; the
        // check is only there to satisfy warn_unused_result and surface a real
        // failure as a thrown V8 error rather than a silent truncation.
        arr->Set(ctx, static_cast<uint32_t>(i),
                 GreenRebuild(dst, &ir->children[i], ctx)).Check();
      }
      return arr;
    }
    case GreenIR::kObject: {
      Local<Object> obj = Object::New(dst);
      for (size_t i = 0; i + 1 < ir->children.size(); i += 2) {
        const GreenIR& key_ir = ir->children[i];
        const GreenIR& val_ir = ir->children[i + 1];
        Local<String> key = String::NewFromUtf8(
            dst, key_ir.prim_str.c_str(), v8::NewStringType::kNormal).ToLocalChecked();
        obj->Set(ctx, key, GreenRebuild(dst, &val_ir, ctx)).Check();
      }
      return obj;
    }
    case GreenIR::kPrim: {
      if (ir->prim_is_undef_null) return Undefined(dst);
      if (ir->prim_is_boolean) return Boolean::New(dst, ir->prim_bool);
      if (ir->prim_is_number) return Number::New(dst, ir->prim_num);
      if (ir->prim_is_string)
        return String::NewFromUtf8(dst, ir->prim_str.c_str(),
                                   v8::NewStringType::kNormal).ToLocalChecked();
      if (ir->prim_is_bigint)
        return BigInt::NewFromWords(ctx, ir->bigint_sign,
                                    static_cast<int>(ir->bigint_words.size()),
                                    ir->bigint_words.data()).ToLocalChecked();
      return Undefined(dst);
    }
  }
  return Undefined(dst);
}

struct GreenValueCopier {
  Isolate* src;  // Isolate the source handle lives on.
  Isolate* dst;  // Isolate to rebuild the value on.
};

// Retrieve (creating it if necessary) the context to read a value on the given
// isolate. In shared-heap mode the green isolate owns a green context; other
// isolates use their current context. NOTE: this returns a handle only -- it
// does NOT make the context current on the isolate. Callers that build values
// on `iso` must enter it themselves (CopyValueAcross installs a Context::Scope),
// because single-arg V8 API constructors resolve their state from the isolate's
// current context. Isolate::Scope below only switches the current isolate so the
// returned handle belongs to the correct one.
static Local<Context> ContextFor(Isolate* iso) {
  if (iso == g_green_isolate) {
    v8::Isolate::Scope isolate_scope(iso);
    // No inner HandleScope: the Local is created in the caller's handle scope,
    // which stays open while the caller reads the context. Returning from an
    // inner scope would dangle the Local (the bug this removes).
    return Local<Context>::New(iso, g_green_ctx);
  }
  return iso->GetCurrentContext();
}

// Copy a value from `copier.src`'s heap to `copier.dst`'s heap. Reads the value
// into a GreenIR tree while the source isolate is current, then rebuilds it
// while the destination isolate is current.
// Returns the rebuilt value rooted in a Persistent on the destination isolate.
// Self-contained: the value is rooted here before this function's scope ends, so
// no caller handle scope is required and the durable handle is simply moved into
// the caller's storage.
static std::unique_ptr<Persistent<Value>> CopyValueAcross(const GreenValueCopier& copier,
                                                          Local<Value> value) {
  // Phase 1: read the value into a GreenIR tree while the source isolate is
  // current.
  {
    v8::Isolate::Scope isolate_src(copier.src);
    HandleScope scope_src(copier.src);
    Local<Context> ctx = ContextFor(copier.src);
    std::unordered_map<uint64_t, GreenIR*> by_addr;
    GreenIR* ir = GreenCapture(copier.src, ctx, value, by_addr);
    // Phase 2: rebuild on the destination isolate and root it in a Persistent.
    {
      v8::Isolate::Scope isolate_dst(copier.dst);
      HandleScope scope_dst(copier.dst);
      // Enter the destination context as current on the destination isolate
      // BEFORE any API constructor runs. GreenRebuild calls single-arg
      // constructors (Object::New, Undefined, Boolean, Number, String) that
      // resolve their backing state from the isolate's *current* context, not
      // from the `ctx` argument they are handed. With no context entered on the
      // green isolate those reads resolve a null/garbage context -- its map
      // word is 0xffffffffffffffff -- and SEGV. This is the crash behind an
      // object argument to spawnAll/spawn and the M2 concurrency probe
      // (stressBody's `parent` object), which copy values onto the green
      // isolate. `dst_ctx` is rooted in scope_dst above, so it stays live for
      // the whole rebuild and across this scope's teardown.
      Local<Context> dst_ctx = ContextFor(copier.dst);
      v8::Context::Scope ctx_scope(dst_ctx);
      Local<Value> rebuilt = GreenRebuild(copier.dst, ir, dst_ctx);
      delete ir;
      return std::make_unique<Persistent<Value>>(copier.dst, rebuilt);
    }
  }
}

// Rebuild a thrown Error on the embedding isolate as a real instance of its
// original class (Error, TypeError, ...). The class name and message are read
// from the original exception in GreenJSEntry (the structured copy carries no
// Error prototype and drops its non-enumerable message), so reconstruct from
// those. Unknown class names fall back to a plain Error carrying the message.
static Local<Value> ReconstructError(Isolate* isolate,
                                     const std::string& name,
                                     const std::string& msg) {
  Local<String> msg_str =
      v8::String::NewFromUtf8(isolate, msg.c_str(),
                              v8::NewStringType::kNormal).ToLocalChecked();
  if (name == "Error") return v8::Exception::Error(msg_str);
  if (name == "TypeError") return v8::Exception::TypeError(msg_str);
  if (name == "RangeError") return v8::Exception::RangeError(msg_str);
  if (name == "ReferenceError") return v8::Exception::ReferenceError(msg_str);
  if (name == "SyntaxError") return v8::Exception::SyntaxError(msg_str);
  return v8::Exception::Error(msg_str);
}
// Lift the embedding function's captured (free-variable) values onto the green
// global by name, so the body reparsed by RelocateFunction resolves them. This
// mirrors LiftEmbeddingGlobals: pass 1 runs on the embedding isolate and walks
// emb_fn's closure-context chain to read each captured value; pass 2 runs on the
// green isolate (inside the GreenGCPause RelocateFunction left suspended) and
// GreenRebuilds the structured values onto green. Native bindings are skipped:
// functions and bindings shaped like the `green_threads` exports can't be
// structured-captured and are resolved on green via the green `g` shell.
//
// Both the captured free-variable NAMES and VALUES come from the closure-context
// chain (JSFunction::context() -> Context::previous(), ScopeInfo::ContextInlined
// LocalName -> Context::Get at the same slot). That walk needs V8's js-function.h
// / scope-info.h / contexts.h, which drag heap.h + V8's tracing/trace-event.h in
// -- so the walk lives in green_threads_gc.cc (the TU allowed to include V8's
// heap headers, see green_threads_gc.h), which keeps green_threads.cc on the
// public API. CaptureGreenClosureValues returns the captured values by name,
// bridged to public Local<Value> on the embedding isolate, so green_threads.cc
// never touches a vi:: pointer. Captures carried between passes are plain C++
// (heap-owned GreenIR pointers keyed by source address), so no handle crosses the
// isolate boundary.
static void LiftCapturedValues(Isolate* emb_isolate,
                               Local<Function> emb_fn,
                               Local<Context> green_ctx) {
  // LiftSpec: one captured free variable, ready to install on the green global.
  struct LiftSpec {
    std::string name;  // install under this name on the green global.
    GreenIR* ir = nullptr;      // plain-data structured capture (heap-owned; freed in pass 2).
    std::string fn_src;         // captured JSFunction source, re-parsed into green code in pass 2.
    bool is_fn = false;         // true when `ir` is null and `fn_src` holds a function body.
  };
  std::vector<LiftSpec> specs;  // captured values, rebuilt/installed in pass 2.
  //
  // Pass 1 -- embedding isolate current. Capture each free var's value, read
  // directly from the closure-context slot (an internal read), structured into a
  // GreenIR tree. The value-local bridge to a public handle happens inside
  // CaptureGreenClosureValues (green_threads_gc.cc); here we only GreenCapture the
  // value while the embedding isolate is current. Values (numbers, arrays, plain
  // objects, ...) are structured-captured so a body that mutates one (e.g. `++count`)
  // writes its private green copy, not the embedding value -- preserving CoW isolation.
  {
    v8::Isolate::Scope isolate_scope(emb_isolate);
    HandleScope scope(emb_isolate);
    Local<Context> emb_ctx = emb_isolate->GetCurrentContext();
    // True when `value` (or any nested property) is a JSFunction: such a value
    // is a native/function-bearing binding (e.g. the `green_threads` exports
    // `g`, or a Node module namespace like `assert`/`performance`) whose function
    // members live on the embedding isolate and cannot be copied into green.
    auto contains_function = [&](Local<Value> value) -> bool {
      if (value->IsFunction()) return true;
      if (!value->IsObject()) return false;
      std::vector<Local<Value>> work;
      work.push_back(value);
      while (!work.empty()) {
        Local<Value> v = work.back();
        work.pop_back();
        if (!v->IsObject() || v->IsArray()) continue;
        Local<Array> keys =
            v.As<Object>()->GetPropertyNames(emb_ctx).ToLocalChecked();
        for (uint32_t i = 0; i < keys->Length(); i++) {
          // Resolve the property by NAME: the integer `i` is an indexed
          // lookup (`Object::Get(context, index)`) that misses named properties,
          // so it cannot see the exports' function methods and would wrongly
          // return false for `g`, letting GreenCapture choke on the binding.
          Local<Value> key = keys->Get(emb_ctx, i).ToLocalChecked();
          if (!key->IsString()) continue;
          v8::Local<v8::String> key_s = key.As<v8::String>();
          Local<Value> p = v.As<Object>()->Get(emb_ctx, key_s).ToLocalChecked();
          if (p->IsFunction()) return true;
          if (p->IsObject()) work.push_back(p);
        }
      }
      return false;
    };
    // CaptureGreenClosureValues yields each captured (name, value) pair while the
    // value is rooted in the scope above -- GreenCapture consumes it into a
    // heap-owned GreenIR within the callback, so no Local escapes this scope.
    CaptureGreenClosureValues(emb_isolate, emb_fn,
                              [&](const std::string& name, Local<Value> value) {
                                if (value->IsFunction()) {
                                  // A plain captured JSFunction carries no embedding
                                  // handle to move, so rather than skip it we re-parse
                                  // its source into fresh green code with its own SFI
                                  // (the same move LiftEmbeddingGlobals makes for
                                  // top-level functions). Re-parsing never shares an
                                  // embedding SFI -- which is what made the old code
                                  // skip functions in the first place -- so this is both
                                  // safe and faithful for functions without an external
                                  // closure. Its free variables resolve against the
                                  // green globals installed in pass 2.
                                  HandleScope hs(emb_isolate);
                                  Local<String> fn_src =
                                      value->ToString(emb_ctx).ToLocalChecked().As<String>();
                                  String::Utf8Value utf8(emb_isolate, fn_src);
                                  specs.push_back(LiftSpec{
                                      name, nullptr,
                                      std::string(*utf8, utf8.length()), true});
                                  return;
                                }
                                if (contains_function(value)) {
                                  // Object/array holding native/function members (the
                                  // `green_threads` binding, an `assert` or
                                  // `performance` namespace, ...) can't be structurally
                                  // copied. Resolve the free variable on green via the
                                  // matching pre-installed global -- the native-binding
                                  // shell (installed under `g`/`green_threads`/`gt`) or a
                                  // module shim -- of the same name. Left undefined if no
                                  // such green global exists: document the limitation.
                                  return;
                                }
                                std::unordered_map<uint64_t, GreenIR*> by_addr;  // fresh per name.
                                specs.push_back(LiftSpec{
                                    name, GreenCapture(emb_isolate, emb_ctx, value, by_addr), std::string{}, false});
                              });
  }
  //
  // Pass 2 -- green isolate current (RelocateFunction installed it inside its
  // GreenGCPause, so allocations here are GC-protected). Rebuild each structured
  // capture on green and install it under its original name, so the reparsed body
  // resolves the free variable against that green global. Each GreenIR is heap-owned
  // and deleted after rebuild.
  {
    v8::Isolate::Scope isolate_scope(g_green_isolate);
    HandleScope scope(g_green_isolate);
    Context::Scope ctx_scope(green_ctx);
    // 2a: plain-data structured captures first, so the function bodies re-parsed in
    // 2b resolve their captured data against these green globals.
    for (const LiftSpec& spec : specs) {
      if (spec.is_fn) continue;
      Local<Value> rebuilt = GreenRebuild(g_green_isolate, spec.ir, green_ctx);
      Local<String> name_key = String::NewFromUtf8(
          g_green_isolate, spec.name.c_str(), v8::NewStringType::kNormal,
          spec.name.length()).ToLocalChecked();
      green_ctx->Global()->Set(green_ctx, name_key, rebuilt).Check();
      delete spec.ir;
    }
    // 2b: re-parse each captured JSFunction into fresh green code and install it
    // under its name. The ambient GreenGCPause (from RelocateFunction) roots the new
    // JSFunction so GC cannot move it mid-install.
    for (const LiftSpec& spec : specs) {
      if (!spec.is_fn) continue;
      std::string program = "(function(){ return " + spec.fn_src + "; })();";
      MaybeLocal<Script> compiled = Script::Compile(
          green_ctx, String::NewFromUtf8(g_green_isolate, program.c_str(),
                                         v8::NewStringType::kNormal)
                         .ToLocalChecked());
      if (compiled.IsEmpty()) continue;
      MaybeLocal<Value> lifted = compiled.ToLocalChecked()->Run(green_ctx);
      if (lifted.IsEmpty() || !lifted.ToLocalChecked()->IsFunction()) continue;
      Local<String> name_key = String::NewFromUtf8(
          g_green_isolate, spec.name.c_str(), v8::NewStringType::kNormal,
          spec.name.length()).ToLocalChecked();
      green_ctx->Global()->Set(green_ctx, name_key, lifted.ToLocalChecked()).Check();
    }
  }
}

// Green scheduler entry point: drives a single green thread's JavaScript body.
//
// @param isolate The V8 internal isolate the green thread was created with.
// @param arg Pointer to the GreenThread the scheduler is currently running.
//
// Runs the body on the first iteration and records its return value, then marks
// the green thread complete. Later iterations are no-ops because the body has
// already finished. If the body throws, the exception is left pending on the
// isolate; because Model A drives entries recursively (no setjmp/longjmp), that
// pending exception propagates up through RunGreenThread and ScheduleLoop to the
// JS caller when RunTasks returns. This stripped V8 exposes no way to read the
// pending exception back, so we do not try to capture a distinct object.
void GreenJSEntry(vi::Isolate* vi_isolate, void* arg) {
  (void)vi_isolate;  // V8 API work below uses the captured public isolate.
  vi::GreenThread* gt = static_cast<vi::GreenThread*>(arg);
  auto it = g_gt_registry.find(gt);
  if (it == g_gt_registry.end()) return;
  GreenCallData* data = it->second;

  // Phase machine: iteration 1 runs the body; anything else has already run.
  // In the recursive Model A, ScheduleLoop calls RunGreenThread once per GT and
  // Yield() drives the rest via recursion, so the body always runs at iter 1.
  // CurrentGreenThread() is already set to this GT by the caller before entry.
  if (gt->iteration_count() >= 2) {
    return;
  }

  // Multiple OS workers now drive bodies concurrently inside ONE shared green
  // heap. Their C++ HandleScopes and Context::Enter/Exit all route through the
  // *shared* HandleScopeImplementer (blocks_ + entered-context stack), and
  // isolate->set_context() is per-isolate too -- so two workers can both Enter
  // the same context and the second Context::Exit aborts with "Cannot exit
  // non-entered context", corrupting heap handles down the line (confirmed
  // empirically: SIGSEGV / BPT / BUS with ~78% rate once serialization was
  // dropped). GreenThreadCowSharedWriterScope serializes the whole per-worker
  // body under the shared-state recursive mutex while a CoW view is active; it
  // is a no-op off-CoW, so serial/CONTROL and warmup are unaffected and the
  // bodies are re-entrant writers (they take this same mutex for their
  // Map/feedback/Code steps, so the recursive mutex handles nesting). Body
  // execution is pinned to a worker (yield cooperates only within the worker),
  // so correctness here is the precondition for unsynchronized parallelism,
  // which additionally requires per-worker HandleScope/entered-context state.
  vi::GreenThreadCowSharedWriterScope shared_writer_scope;

  // Enter the green thread's own isolate so its body runs on that heap.
  // The pool drives bodies on real OS workers, so absent this the current
  // isolate would be whichever worker last entered it; GreenJSEntry must
  // switch to data->isolate (the green isolate in shared-heap mode) for the
  // whole body execution. Constructed before the HandleScope so it is the last
  // to go, restoring the embedding isolate once the body has fully returned.
  v8::Isolate::Scope isolate_scope(data->isolate);
  HandleScope scope(data->isolate);
  Local<Context> ctx = Local<Context>::New(data->isolate, data->context_p);
  Context::Scope context_scope(ctx);

  // fn_p lives on fn_isolate, which is the same isolate as data->isolate in
  // every current mode (green in shared-heap / M2 modes, embedding otherwise),
  // so recovering it here is equivalent to reading it on data->isolate; the
  // field documents ownership and future-proofs a split ownership.
  Local<Function> fn = Local<Function>::New(data->fn_isolate, data->fn_p);
  Local<Value> recv = ctx->Global();
  std::vector<Local<Value>> argv;
  argv.reserve(data->args_p.size());
  for (auto& arg : data->args_p) {
    argv.push_back(Local<Value>::New(data->isolate, *arg));
  }

  Local<Value> out;
  v8::TryCatch body_tc(data->isolate);
  if (!fn->Call(ctx, recv, argv.size(), argv.data()).ToLocal(&out)) {
    // The body threw: TryCatch has already captured the exception and its
    // message. Root it in exc_p (on the executing isolate) while the TryCatch
    // is still in scope, then let it return. Its Exit() restores the isolate's
    // pending exception to what it was before the body ran -- null for a clean
    // task -- which clears the leak the previous code left: a sibling task on
    // the shared green heap would otherwise inherit this exception, and the
    // isolate would report it as "Uncaught" (caught exceptions are not reported
    // unless SetVerbose(true)).
    if (!body_tc.Exception().IsEmpty()) {
      data->exc_p.Reset(data->isolate, body_tc.Exception());
      // An Error carries `name` on its prototype and `message` as a
      // non-enumerable own prop. The structured copy (no prototype, dropped
      // attributes) cannot move those faithfully, so capture the RESOLVED
      // values here -- while the real object still has its prototype -- and
      // rebuild the exact error class on the embedding isolate in RunTasks.
      // Non-array Error-like objects only; arrays/primitives copy through
      // exc_p as before.
      Local<Value> exc = body_tc.Exception();
      if (exc->IsObject() && !exc->IsArray()) {
        Local<String> name_key =
            String::NewFromUtf8(data->isolate, "name",
                                v8::NewStringType::kNormal)
                .ToLocalChecked();
        Local<Value> name_v;
        if (Local<Object>::Cast(exc)
                ->Get(ctx, name_key)
                .ToLocal(&name_v) &&
            name_v->IsString()) {
          String::Utf8Value nv(data->isolate, name_v);
          data->exc_name.assign(*nv, nv.length());
          Local<String> msg_key =
              String::NewFromUtf8(data->isolate, "message",
                                  v8::NewStringType::kNormal)
                  .ToLocalChecked();
          Local<Value> msg_v;
          if (Local<Object>::Cast(exc)
                  ->Get(ctx, msg_key)
                  .ToLocal(&msg_v) &&
              msg_v->IsString()) {
            String::Utf8Value mv(data->isolate, msg_v);
            data->exc_msg.assign(*mv, mv.length());
            data->exc_is_error = true;
          }
        }
      }
    }
    data->threw = true;
    gt->set_state(vi::GreenThreadState::kComplete);
    return;
  }
  data->result_p.Reset(data->isolate, out);
  gt->set_state(vi::GreenThreadState::kComplete);
}

// Throws a TypeError and returns from the binding.
void ThrowTypeError(const Isolate* isolate, const char* message) {
  HandleScope scope(const_cast<Isolate*>(isolate));
  Local<Value> exception = v8::Exception::TypeError(String::NewFromUtf8(
      const_cast<Isolate*>(isolate), message,
      v8::NewStringType::kNormal).ToLocalChecked());
  const_cast<Isolate*>(isolate)->ThrowException(exception);
}

// Builds a green thread that runs a JavaScript body.
//
// Model A drives entries recursively on the embedding thread and never
// switches to the per-thread stack, so unlike v8's GreenScheduler::CreateThread
// we do not mmap a stack here -- only the descriptor, the entry point, and the
// COW view need setting up. See green-thread-hello-world.cc for the same shape.
//
// `Initialize` allocates the per-GT ref table, records the thread as its owner,
// marks COW on, and increments the active-thread count. The write barrier gates
// on that count, so this is what makes copy-on-write actually fire.
// `GreenThreadCow::Teardown` (called per thread after scheduling) balances it.
vi::GreenThread* CreateGreenThread(vi::Isolate* isolate) {
  vi::GreenThread* gt = new vi::GreenThread(isolate);
  gt->set_pc_raw(reinterpret_cast<Address>(&GreenJSEntry));
  gt->set_state(vi::GreenThreadState::kRunnable);
  vi::GreenThreadCow::Initialize(gt);
  return gt;
}

// Scheduler bound to the OS thread running a pool worker. Read by Yield() to
// drive that worker's recursive interleave. Null outside a pool worker (so
// yield() from the embedding thread is a no-op, matching the pre-pool contract).
static thread_local vi::GreenScheduler* g_current_pool_scheduler = nullptr;

// Real-OS-thread worker pool. Each worker owns its own GreenScheduler (per
// worker global runqueue + processor), so bodies are pinned to a worker and
// the recursive interleave that implements yield() never crosses an OS-thread
// boundary -- a body's continuation lives on the stack of the worker that
// started it. Bodies are distributed round-robin; a body that never yields runs
// straight through, so N workers give N-way parallelism for independent bodies.
class GreenThreadPool {
 public:
  void Run(std::vector<vi::GreenThread*> gts, vi::Isolate* vi_isolate_for_gt,
           int num_workers);
 private:
  static void WorkerEntry(GreenThreadPool* self, int worker_id);

  std::vector<std::unique_ptr<vi::GreenScheduler>> schedulers_;
  std::vector<std::thread> workers_;
};

void GreenThreadPool::WorkerEntry(GreenThreadPool* self, int worker_id) {
  vi::GreenScheduler* sched = self->schedulers_[worker_id].get();
  g_current_pool_scheduler = sched;
  sched->ScheduleLoop();
  g_current_pool_scheduler = nullptr;
}

void GreenThreadPool::Run(std::vector<vi::GreenThread*> gts,
                          vi::Isolate* vi_isolate_for_gt, int num_workers) {
  if (num_workers < 1) num_workers = 1;
  if (static_cast<int>(gts.size()) < num_workers)
    num_workers = static_cast<int>(gts.size());

  // One GreenScheduler per worker, each initialized to a single processor.
  schedulers_.reserve(num_workers);
  for (int i = 0; i < num_workers; i++) {
    schedulers_.push_back(std::make_unique<vi::GreenScheduler>(vi_isolate_for_gt));
    schedulers_.back()->Initialize(1);
  }

  // Round-robin partition: body i runs on worker (i % num_workers).
  for (uint32_t i = 0; i < gts.size(); i++) {
    schedulers_[i % num_workers]->EnqueueGlobal(gts[i]);
  }

  for (int i = 0; i < num_workers; i++)
    workers_.emplace_back(&GreenThreadPool::WorkerEntry, this, i);
  for (std::thread& w : workers_) w.join();

  for (auto& s : schedulers_) s->Shutdown();
}

// Shared core: schedule every task in `tasks` ([[fn, ...args]]) and run all of
// their green threads to completion. Returns the per-task results array.
//
// If a body throws, its exception is left pending on the isolate and *threw_out
// is set so the caller avoids returning a value and lets that exception surface.
// A malformed `tasks` argument throws before any green thread is scheduled.
Local<Array> RunTasks(Isolate* isolate, Local<Context> ctx, Local<Array> tasks,
                      bool* threw_out) {
  EscapableHandleScope scope(isolate);
  const uint32_t n = tasks->Length();
  *threw_out = false;

  vi::Isolate* vi_isolate = reinterpret_cast<vi::Isolate*>(isolate);

  // In shared-heap mode the green threads, their COW clones, and their run
  // state all live on the dedicated green isolate; vi_isolate_for_gt is the
  // one handed to CreateGreenThread (its COW clones onto that isolate's heap,
  // which is what keeps green bodies off the embedding heap). green_isolate_ret
  // is the public green isolate, used to copy results back out.
  vi::Isolate* vi_isolate_for_gt = vi_isolate;
  Isolate* green_isolate_ret = isolate;

  // Phase 1: validate every task before allocating any per-thread state, so a
  // malformed task aborts the whole call without leaving allocations behind.
  // Task bodies and their argument arrays are remembered here and reused in
  // Phase 2, avoiding a second lookup and re-narrowing.
  std::vector<Local<Function>> fns;
  std::vector<Local<Array>> task_arrays;
  fns.reserve(n);
  task_arrays.reserve(n);
  for (uint32_t i = 0; i < n; i++) {
    Local<Value> item;
    if (!tasks->Get(ctx, i).ToLocal(&item) || !item->IsArray()) {
      *threw_out = true;
      ThrowTypeError(isolate, "green_threads.spawnAll(tasks): "
                                   "each task must be an array [fn, ...args]");
      return scope.Escape(Array::New(isolate, 0));
    }
    Local<Array> task = Local<Array>::Cast(item);
    // Narrowing Value -> Function needs As<S>() here: this stripped V8's
    // ToLocal() performs same-type assignment only and cannot assign a
    // Local<Value> into a Local<Function>*.
    Local<Function> fn;
    if (!task->Get(ctx, 0).As<Function>().ToLocal(&fn) || !fn->IsFunction()) {
      *threw_out = true;
      ThrowTypeError(isolate, "green_threads.spawnAll(tasks): "
                                   "task[0] must be a function");
      return scope.Escape(Array::New(isolate, 0));
    }
    fns.push_back(fn);
    task_arrays.push_back(task);
  }

  // Phase 2: allocate per-thread state for the now-known-valid tasks.
  std::vector<GreenCallData*> datas;
  datas.reserve(n);
  if (GreenHeapEnabled()) {
    // Shared-heap mode: route each task onto the dedicated green isolate. The
    // body is the embedding Function itself; it is called on the green
    // context (see GreenJSEntry) but kept rooted on the embedding isolate so its
    // captured lexical scope (closures) survives -- reparsing its source into a
    // fresh green function would strip the closure and every scenario that
    // references captured state would throw. Arguments are copied in so bodies
    // mutate private copies and the shared `parent` in the harness is never
    // corrupted. COW clones land on the green heap too.
    green_isolate_ret = GetGreenIsolate(isolate);
    vi_isolate_for_gt = reinterpret_cast<vi::Isolate*>(green_isolate_ret);
    // Ensure the green context lives in the durable persistent g_green_ctx
    // (creating it + lifting embedding globals on first use). Bodies and
    // argument copies below each root their outputs in a Persistent, so nothing
    // dangles across the isolate switch.
    EnsureGreenContext(isolate, ctx);

    GreenValueCopier args_copier;
    args_copier.src = isolate;
    args_copier.dst = green_isolate_ret;
    for (uint32_t i = 0; i < n; i++) {
      Local<Array> task = task_arrays[i];
      GreenCallData* data = new GreenCallData();
      data->isolate = green_isolate_ret;
      // context_p is rooted from the durable persistent g_green_ctx -- no green
      // handle scope is needed.
      data->context_p.Reset(green_isolate_ret, g_green_ctx);
      // Relocate the body onto the green isolate. RelocateFunction returns a
      // durable Persistent rooted inside itself, so no caller handle scope is
      // needed; Reset copies that handle into data->fn_p (it is not a Persistent
      // move, which does not exist for this build). fn_isolate is green, so
      // GreenJSEntry recovers the handle on the green isolate.
      std::unique_ptr<Persistent<Function>> relocated =
          RelocateFunction(isolate, fns[i]);
      data->fn_isolate = green_isolate_ret;
      data->fn_p.Reset(green_isolate_ret, *relocated);
      const uint32_t argc = task->Length();
      for (uint32_t a = 1; a < argc; a++) {
        // The argument is read on the embedding isolate; its Local is rooted in
        // RunTasks's embedding scope, which stays open for the whole call.
        Local<Value> arg = task->Get(ctx, a).ToLocalChecked();
        // CopyValueAcross returns a durable Persistent rooted on the green
        // isolate; move that handle into the argument vector.
        data->args_p.push_back(CopyValueAcross(args_copier, arg));
      }
      datas.push_back(data);
    }
  } else {
    for (uint32_t i = 0; i < n; i++) {
      Local<Array> task = task_arrays[i];
      GreenCallData* data = new GreenCallData();
      data->isolate = isolate;
      data->fn_isolate = isolate;
      data->context_p.Reset(isolate, ctx);
      data->fn_p.Reset(isolate, fns[i]);
      const uint32_t argc = task->Length();
      for (uint32_t a = 1; a < argc; a++) {
        data->args_p.push_back(
            std::make_unique<Persistent<Value>>(isolate,
                                                task->Get(ctx, a).ToLocalChecked()));
      }
      datas.push_back(data);
    }
  }

  const uint32_t m = datas.size();

  // Create each green thread (which enables COW, allocates its ref table and
  // registers it as COW-active), register its call data, and collect its
  // descriptor. The worker pool below partitions them across real OS threads.
  std::vector<vi::GreenThread*> gts;
  gts.reserve(datas.size());
  for (GreenCallData* data : datas) {
    vi::GreenThread* gt = CreateGreenThread(vi_isolate_for_gt);
    data->gt = gt;
    g_gt_registry[gt] = data;
    gts.push_back(gt);
  }

  // Drive every green thread to completion on the real-OS-thread worker pool.
  //
  // Per-OS-thread attribution: CurrentGreenThread() routes to each worker's
  // PerIsolateThreadData only in green-thread mode; in serial mode it falls
  // back to the process-global slot. Bodies here are driven by the real OS
  // worker pool: off-CoW (ActiveCount()==0, i.e. the serial path) no CoW
  // write-barrier ever reads CurrentGreenThread, so one worker-at-a-time
  // attribution slot is sufficient and no flip is required. When a CoW view is
  // active (shared-heap) the GreenThreadCowSharedWriterScope body lock serializes
  // execution, so again only one body drives the scheduler at a time and the
  // per-worker green isolate (already in green_thread_mode) routes attribution
  // safely. Flipping the embedding isolate's green_thread_mode here would only
  // expose it to PerIsolateThreadData that is not green-mode initialized.
  Isolate* working_isolate = GreenHeapEnabled() ? green_isolate_ret : isolate;

  // Concurrent CAS bump-pointer reservation: N workers hand out disjoint
  // young-gen pointers from ONE green heap instead of racing the plain bump
  // write (which would hand two workers the same address). Restored to the
  // serial fast path once the workers join; the single-threaded result
  // copy-back + teardown below must not pay CAS overhead.
  const bool enable_cas = GreenHeapEnabled();
  if (enable_cas) vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(working_isolate, true);

  {
    int num_workers = static_cast<int>(std::thread::hardware_concurrency());
    if (num_workers < 1) num_workers = 1;
    // Override with GT_NUM_WORKERS (when set and > 0) to pin the green-thread
    // worker pool to a fixed number of OS threads, regardless of the host core
    // count. This lets a harness exercise real multi-core GT parallelism
    // deterministically (e.g. GT_NUM_WORKERS=4) instead of one worker per core.
    if (const char* wenv = ::getenv("GT_NUM_WORKERS")) {
      int requested = atoi(wenv);
      if (requested > 0) num_workers = requested;
    }
    GreenThreadPool pool;
    pool.Run(gts, vi_isolate_for_gt, num_workers);
  }

  if (enable_cas) vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(working_isolate, false);

  // Gather results. A body that threw had its exception captured and cleared
  // by GreenJSEntry (see there); we stop populating as soon as we notice so the
  // caller can propagate the original exception rather than construct a
  // replacement. Because we must not Set a return value once an exception is
  // pending, we surface the fact through *threw_out and, at the end, re-throw
  // the captured exception on this isolate.
  Local<Array> results = Array::New(isolate, m);
  bool threw = false;
  // The first thrown body's exception, copied onto the embedding isolate so it
  // survives the per-task teardown below and can be re-thrown at the API
  // boundary. Copied here (before teardown disposes the green exc_p), not in
  // GreenJSEntry, so it is rooted on the embedding isolate RunTasks already
  // owns. When the body threw an Error-like value, ReconstructError instead
  // rebuilds the exact error class from the name/message captured in
  // GreenJSEntry; recon_error flags that case and recon_name/recon_msg hold the
  // captured values.
  std::unique_ptr<Persistent<Value>> thrown_exc;
  bool recon_error = false;
  std::string recon_name;
  std::string recon_msg;
  if (GreenHeapEnabled()) {
    // Results live on the green heap. Hold the green Isolate::Scope +
    // HandleScope for the whole loop so green result handles can be created;
    // CopyValueAcross moves each one into the embedding isolate. The copied
    // value is rooted in a Persistent on the embedding isolate so it survives
    // the green scope's teardown before it is Set on results.
    v8::Isolate::Scope green_scope(green_isolate_ret);
    HandleScope green_handle_scope(green_isolate_ret);
    GreenValueCopier out_copier;
    out_copier.src = green_isolate_ret;
    out_copier.dst = isolate;
    for (uint32_t i = 0; i < m; i++) {
      if (datas[i]->threw) {
        threw = true;
        if (datas[i]->exc_is_error) {
          // Error-like throw: rebuild the exact class on the embedding isolate
          // from the captured name/message (the structured copy cannot carry an
          // Error's prototype or its non-enumerable message). No cross-isolate
          // copy is needed.
          recon_error = true;
          recon_name = datas[i]->exc_name;
          recon_msg = datas[i]->exc_msg;
        } else {
          // Carry the captured exception onto the embedding isolate before
          // teardown below disposes the green exc_p.
          Local<Value> green_exc =
              Local<Value>::New(green_isolate_ret, datas[i]->exc_p);
          if (!green_exc.IsEmpty()) {
            thrown_exc = CopyValueAcross(out_copier, green_exc);
          }
        }
        break;
      }
      Local<Value> green_result =
          Local<Value>::New(green_isolate_ret, datas[i]->result_p);
      // CopyValueAcross returns a durable Persistent rooted on the embedding
      // isolate; root the result row directly from it.
      std::unique_ptr<Persistent<Value>> emb_result_p =
          CopyValueAcross(out_copier, green_result);
      results->Set(ctx, i, Local<Value>::New(isolate, *emb_result_p)).Check();
    }
  } else {
    for (uint32_t i = 0; i < m; i++) {
      if (datas[i]->threw) {
        threw = true;
        if (datas[i]->exc_is_error) {
          recon_error = true;
          recon_name = datas[i]->exc_name;
          recon_msg = datas[i]->exc_msg;
        } else {
          // exc_p is already rooted on the embedding isolate in this mode.
          Local<Value> emb_exc = Local<Value>::New(isolate, datas[i]->exc_p);
          if (!emb_exc.IsEmpty()) {
            thrown_exc = std::make_unique<Persistent<Value>>(isolate, emb_exc);
          }
        }
        break;
      }
      results->Set(ctx, i, Local<Value>::New(isolate, datas[i]->result_p)).Check();
    }
  }
  if (threw) {
    *threw_out = true;
  }

  // Release green threads and call data.
  for (GreenCallData* data : datas) {
    vi::GreenThreadCow::Teardown(data->gt);
    delete data->gt;
    // clear() destroys each unique_ptr, which Disposes its rooted Persistent.
    data->args_p.clear();
    data->result_p.Reset();
    data->fn_p.Reset();
    g_gt_registry.erase(data->gt);
    delete data;
  }

  // Re-throw the first thrown body's exception on the embedding isolate so it
  // reaches the JS caller's try/catch. ThrowException leaves it pending;
  // Spawn/SpawnAll return without Set-ing a return value once *threw_out is
  // true, surfacing the exception verbatim instead of the swallowed undefined a
  // thrown body previously yielded. recon_error rebuilds a faithful Error
  // (recon_name/recon_msg captured in GreenJSEntry); thrown_exc was copied here
  // (before the teardown loop above disposed the green exc_p) for the
  // non-error copy path.
  if (recon_error) {
    isolate->ThrowException(
        ReconstructError(isolate, recon_name, recon_msg));
  } else if (thrown_exc) {
    isolate->ThrowException(Local<Value>::New(isolate, *thrown_exc));
  }
  return scope.Escape(results);
}

}  // namespace

// Public API
// ====================================================================

// green_threads.spawnAll(tasks)
// tasks: array of [fn, ...args] entries.
// Returns: array of each body's return value, in task order. Bodies run
// cooperatively to completion; a body that calls green_threads.yield() lets the
// others run in between.
void SpawnAll(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);

  if (!args[0]->IsArray()) {
    ThrowTypeError(isolate, "green_threads.spawnAll(tasks): "
                                "first argument must be an array of [fn, ...args]");
    return;
  }
  Local<Array> tasks = Local<Array>::Cast(args[0]);
  bool threw = false;
  Local<Array> results =
      RunTasks(isolate, isolate->GetCurrentContext(), tasks, &threw);
  // A body or task threw: the pending exception propagates to the JS caller, so
  // do not Set a return value (doing so would clear it).
  if (threw) return;
  args.GetReturnValue().Set(results);
}

// green_threads.spawn(fn, ...args)
// Convenience wrapper spawning a single green thread. Returns the body result.
void Spawn(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);

  if (!args[0]->IsFunction()) {
    ThrowTypeError(isolate, "green_threads.spawn(fn, ...args): "
                                "first argument must be a function");
    return;
  }

  const uint32_t argc = args.Length();
  Local<Context> ctx = isolate->GetCurrentContext();
  Local<Array> tasks = Array::New(isolate, 1);
  Local<Array> task = Array::New(isolate, argc);
  task->Set(ctx, 0, args[0]).Check();
  for (uint32_t i = 1; i < argc; i++) {
    task->Set(ctx, i, args[i]).Check();
  }
  tasks->Set(ctx, 0, task).Check();

  bool threw = false;
  Local<Array> results = RunTasks(isolate, ctx, tasks, &threw);
  // A body threw: the pending exception propagates to the JS caller.
  if (threw) return;
  if (results->Length() == 0) {
    args.GetReturnValue().SetUndefined();
  } else {
    args.GetReturnValue().Set(results->Get(ctx, 0).ToLocalChecked());
  }
}

// green_threads.__concurrent_entry(n) -> { count, concurrent, results_ok }
//
// M1 architecture probe: proves that `n` OS worker threads can Enter the SAME
// green isolate *simultaneously*, each holding a green entry frame. This
// exercises the green_thread_mode_ entry path in Isolate::Enter/Exit -- per
// worker keeps its own EntryStackItem chain in its PerIsolateThreadData, so the
// change that makes concurrent multi-mutator entry possible (see
// GT-OS-WORKER-POOL-PLAN.md).
//
// Each worker thread:
//   1. Enters the green isolate (v8::Isolate::Scope), exercising Enter/Exit in
//      green_thread_mode_. It records that GetCurrent()==green on this OS
//      thread, i.e. the green entry frame (the per-thread entry_stack_ chain)
//      is live.
//   2. Reaches a rendezvous barrier and waits until all n workers have arrived.
//      Serial (single-OS-thread) execution deadlocks at the barrier and the
//      native side times out -> `concurrent` is false. Only n real OS threads
//      arrive together, so `concurrent` is true.
//   3. Exits (green_thread_mode_ path), keeping its own chain intact.
//
// This probe proves simultaneous ENTRY only. The barrier and the entry-frame
// check touch no green-heap objects, so they cannot race the CoW machinery
// (which is built around a single active JS mutator). Concurrent general-JS
// execution -- heap access via inline caches, feedback vectors, and object
// migration -- is an M2 / all-mutator-GC concern (invariants I3/I4) and is
// deliberately not part of this bounded probe. Incremental marking is paused
// (GreenGCPause) across setup + probe as an extra guard.
void ConcurrentEntry(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);
  Local<Context> ctx = isolate->GetCurrentContext();

  if (args.Length() < 1 || !args[0]->IsNumber()) {
    ThrowTypeError(isolate,
                   "green_threads.__concurrent_entry(n): the first argument "
                   "must be a number");
    return;
  }
  const int n = static_cast<int>(args[0]->NumberValue(ctx).ToChecked());
  if (n < 1 || n > 64) {
    ThrowTypeError(isolate,
                   "green_threads.__concurrent_entry(n): n must be in [1, 64]");
    return;
  }

  // Create (and mark green-thread mode) the shared green isolate + context on
  // the embedding thread, before any worker runs.
  Isolate* green = GetGreenIsolate(isolate);
  EnsureGreenContext(isolate, ctx);

  // Shared rendezvous state, referenced by every worker through pointers.
  std::atomic<int> arrived{0};
  std::atomic<bool> timed_out{false};
  // char (not std::vector<bool>): vector<bool> is a bitfield whose
  // read-modify-write is not safe for concurrent access to distinct elements,
  // so each worker writes its own byte.
  std::vector<char> got_result(static_cast<size_t>(n), 0);
  std::vector<char> entry_live(static_cast<size_t>(n), 0);
  const long long deadline_ns =
      std::chrono::steady_clock::now().time_since_epoch().count() +
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::milliseconds(5000))
          .count();
  std::atomic<int>* arrived_p = &arrived;
  std::atomic<bool>* timed_out_p = &timed_out;
  std::vector<char>* got_p = &got_result;
  std::vector<char>* live_p = &entry_live;

  // Pause incremental marking across the whole probe as an extra guard so the
  // barrier cannot observe a concurrent collection. This probe holds no green
  // heap objects, so nothing else needs pausing.
  {
    v8::Isolate::Scope isolate_scope(green);
    GreenGCPause gc(green);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const int idx = i;
      workers.emplace_back([=]() {
        // (1) Enter the green isolate: green_thread_mode_ keeps this worker's
        // EntryStackItem chain on its own PerIsolateThreadData, so it cannot
        // clobber a sibling worker's.
        v8::Isolate::Scope scope(green);

        // The green entry frame is live on this OS thread: Current() is the
        // green isolate (its per-thread entry_stack_ chain was pushed by Enter).
        (*live_p)[static_cast<size_t>(idx)] =
            static_cast<char>(v8::Isolate::GetCurrent() == green);

        // (2) Rendezvous barrier: every worker must arrive before any proceeds.
        // Serial (single-OS-thread) execution deadlocks here -- worker 0 spins
        // waiting for threads that can never be scheduled -- so the timeout
        // fires and timed_out is set -> concurrent comes back false. Only n
        // real OS threads arrive together, so concurrent is true.
        (void)arrived_p->fetch_add(1, std::memory_order_seq_cst);
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(5000);
        while (arrived_p->load(std::memory_order_seq_cst) < n) {
          if (std::chrono::steady_clock::now() >= deadline) {
            timed_out_p->store(true, std::memory_order_seq_cst);
            return;
          }
          std::this_thread::yield();
        }
        (*got_p)[static_cast<size_t>(idx)] = 1;

        // scope destructor: Exit the green isolate (green_thread_mode_ path),
        // popping this worker's own entry_stack_ chain.
      });
    }
    for (std::thread& w : workers) w.join();
  }
  // Green scope exited + GC resumed; the embedding isolate is current again.

  // Build the report on the embedding isolate.
  bool all_reached = true;  // every worker reached the rendezvous
  bool all_live = true;     // every worker held a live green entry frame
  for (int i = 0; i < n; ++i) {
    if (!got_result[static_cast<size_t>(i)]) all_reached = false;
    if (!entry_live[static_cast<size_t>(i)]) all_live = false;
  }
  const bool concurrent = !timed_out.load(std::memory_order_seq_cst);
  Local<Object> report = Object::New(isolate);
  report->Set(ctx, String::NewFromUtf8(isolate, "count").ToLocalChecked(),
              Number::New(isolate, static_cast<double>(n)));
  report->Set(ctx, String::NewFromUtf8(isolate, "concurrent").ToLocalChecked(),
              Boolean::New(isolate, concurrent));
  report->Set(ctx,
              String::NewFromUtf8(isolate, "results_ok").ToLocalChecked(),
              Boolean::New(isolate, concurrent && all_reached && all_live));
  args.GetReturnValue().Set(report);
}

// Coarse green-mutator lock (M2a baseline). Serializes concurrent body execution
// so at most ONE OS worker mutates the green heap's shared state (bump-pointer
// allocator, cached Maps, and any GC) at a time. Concurrent object-literal
// creation races here: 8 threads create objects of the same shape, resolve the
// single cached Map, and mutate it via Map::InobjectSlackTrackingStep while their
// young-space bump pointers collide -> EXC_BAD_ACCESS in Factory/Map stores.
// True all-mutator parallelism (I3 stop-the-world GC + I4 serialized allocation)
// is the later milestone; this lock is the "mutex-protected" phase the plan
// anticipates and the safe correctness baseline N workers run behind. Kept
// process-global: there is exactly one green isolate per process.
static std::mutex g_green_mutator_lock;

// One unit of work for the M2Probe worker pool: a green thread and its call
// data, claimed by an OS worker that runs its body to completion. `idx` is the
// worker's position in the probe's result vector (also the id baked into its
// Item-2 forced-GC marker), so a worker records its success under its own slot.
struct Work {
  vi::GreenThread* gt;
  GreenCallData* data;
  int idx;
};

// A tiny thread-safe claim queue driving the M2Probe fan-out. Unlike
// GreenScheduler's global runqueue (accessed only from the embedding thread, so
// green-thread-scheduler.h declares it needs no sync), this pool is driven by n
// real OS threads that must claim work concurrently -- hence the mutex +
// condition variable. Each queued item is handed to exactly one worker via a
// monotonic claim index, so no item is double-claimed and none is lost. The work
// items themselves live in the caller's vector for the whole run; the queue only
// hands them out. The recursive GreenScheduler::Yield entry() is deliberately not
// used here -- each worker runs its claimed body to completion.
class M2ProbeWorkQueue {
 public:
  explicit M2ProbeWorkQueue(std::vector<Work>* works) : works_(works) {}

  // Block until the next unclaimed work item is available, then claim it by
  // index. Returns false once every item has been claimed (queue exhausted).
  bool WaitAndClaim(Work* out) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return next_ >= works_->size() || shutdown_; });
    if (next_ >= works_->size()) return false;
    *out = (*works_)[next_];
    ++next_;
    return true;
  }

  // Release any worker parked in WaitAndClaim during teardown.
  void Shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    shutdown_ = true;
    cv_.notify_all();
  }

 private:
  std::vector<Work>* works_;
  std::mutex mu_;
  std::condition_variable cv_;
  size_t next_ = 0;
  bool shutdown_ = false;
};

// green_threads.__m2_probe(n, fn0, ..., fn_{n-1}) -> { results: [..], all_ok: bool }
//
// M2 foundational probe: `n` OS worker threads run `n` self-contained allocating
// CoW bodies SIMULTANEOUSLY in ONE green isolate. This is the first slice of the
// all-mutator-GC work (GT-OS-WORKER-POOL-PLAN.md M2) and exercises two invariants
// under real concurrent JS execution:
//
//   I1 (worker-local GT attribution). Each worker sets its own GT as the current
//   green thread BEFORE running its body, so the CoW write barrier attributes
//   every store to THIS worker's Thread -- not the process-global slot a sibling
//   worker would clobber. Before M2 this slot was process-global (single
//   embedding thread); see SetCurrentGreenThread/CurrentGreenThread.
//
//   I4 (per-GT CoW ownership). Each body allocates disjoint objects, so the only
//   shared state is the green heap itself. A crash or wrong result here is a
//   multi-mutator race, not a test bug.
//
// Bodies are self-contained (no embedding-scope closure): closure capture cannot
// cross the isolate boundary, so a dependency-bearing body would fail for an
// orthogonal reason. Forced GC while workers are active (I3) is a later slice;
// this pass leaves allocation-driven (serial) GC alone so the I1/I4 signals are
// isolated.
void M2Probe(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);
  Local<Context> ctx = isolate->GetCurrentContext();

  if (args.Length() < 2 || !args[0]->IsNumber()) {
    ThrowTypeError(isolate,
                   "green_threads.__m2_probe(n, ...bodies): the first argument "
                   "must be a worker count");
    return;
  }
  const int n = static_cast<int>(args[0]->NumberValue(ctx).ToChecked());
  if (n < 1 || n > 64) {
    ThrowTypeError(isolate,
                   "green_threads.__m2_probe(n, ...): n must be in [1, 64]");
    return;
  }
  // A probe run is allowed either a full set of n distinct bodies or a single
  // body replicated across all n workers. The shared-heap probe only needs to
  // prove that n OS workers allocate concurrently from ONE green heap, so a
  // lone body is filled out to the full worker count (see the setup loop below).
  if (args.Length() - 1 < 1) {
    ThrowTypeError(isolate,
                   "green_threads.__m2_probe(n, ...): expected at least one "
                   "body function");
    return;
  }

  // 1. Shared green isolate + context. Relocate + register every body on the
  //    embedding thread (single writer, so g_gt_registry has no concurrent
  //    writes); only after all setup do workers run.
  Isolate* green = GetGreenIsolate(isolate);
  EnsureGreenContext(isolate, ctx);

   // Item 2 (acceptance gate for item 1): an env-gated forced-GC knob.
   // Unset = current behavior (byte-for-byte unchanged). "young"|"full"|"both"
   // forces the corresponding collection(s) while the N workers are active.
   const M2ForcedGCMode force_gc_mode = M2ParseForcedGCEnv();
  std::vector<Work> works;
  works.reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    // Use args[i+1] when present; otherwise replicate the first body so a
    // single-argument probe still spawns n independent workers on the same
    // green heap.
    const int body_arg =
        (static_cast<int>(args.Length()) - 1 > i) ? (i + 1) : 1;
    Local<Function> emb_fn = Local<Function>::Cast(args[body_arg]);
    std::unique_ptr<Persistent<Function>> relocated = RelocateFunction(isolate, emb_fn);

    GreenCallData* data = new GreenCallData();
    data->isolate = green;
    // RelocateFunction returns a green-isolate Persistent, so GreenJSEntry must
    // recover the handle on the green isolate here (unlike RunTasks, which roots
    // the embedding Function on the embedding isolate to preserve closures).
    data->fn_isolate = green;
    data->context_p.Reset(green, g_green_ctx);
    data->fn_p.Reset(green, *relocated);

    vi::GreenThread* gt = CreateGreenThread(reinterpret_cast<vi::Isolate*>(green));
    data->gt = gt;
    g_gt_registry[gt] = data;
    works.push_back(Work{gt, data, i});
     // Item 2: allocate a disjoint per-worker allocation marker on the green
     // heap, rooted via a Persistent handle so the forced GC (which runs while
     // this worker is parked) cannot reclaim it. post_gc_consistent verifies
     // it survived the collection intact. Only built when the knob is set.
     if (force_gc_mode != M2ForcedGCMode::kNone) {
       v8::Isolate::Scope marker_scope(green);
       v8::HandleScope marker_hs(green);
       Local<Context> marker_ctx = Local<Context>::New(green, data->context_p);
       Context::Scope marker_ctx_scope(marker_ctx);
       Local<Object> marker = Object::New(green);
       marker->Set(marker_ctx,
                   String::NewFromUtf8(green, "id").ToLocalChecked(),
                   v8::Integer::New(green, i)).Check();
       data->marker_p.Reset(green, marker);
     }
  }

  // Warm up each body ONCE on this (single embedding) thread before any worker
  // is spawned. First execution triggers JIT compilation: Compiler::Compile ->
  // JSFunction::UpdateCode writes shared Code objects and the Code::flags_
  // atomic. Concurrent workers hitting that exact path is the first M2 race
  // (observed: multiple OS threads EXC_BAD_ACCESS in UpdateCode/Code::flags),
  // because 8 threads cannot safely write Code state at once. After warmup the
  // optimized Code is already set, so a worker's Builtins_CompileLazy only
  // READS Code flags (safe concurrently) and never calls UpdateCode again.
  // Bodies here are pure (allocate-local, return), so the discarded warmup
  // run does not affect the concurrent results.
  for (Work& w : works) {
    vi::SetCurrentGreenThread(w.gt);
    GreenJSEntry(reinterpret_cast<vi::Isolate*>(green), w.gt);
  }

  // 2. Run n workers concurrently. `ok` is written by every worker, so each
  //    worker gets a private pointer into it (captured by value) to avoid the
  //    [=} capture making a per-worker copy.
  std::vector<char> ok(static_cast<size_t>(n), 0);
  char* ok_p = ok.data();
  // Item 4: enable the atomic CAS bump-pointer for the duration of the
  // concurrent fan-out so the N workers reserve disjoint young-gen pointers
  // from the SAME green heap instead of racing the plain bump write (top_ +=
  // bytes), which would hand two workers the same address. Reset to the serial
  // fast path once the workers join; the result-copy and teardown below are
  // single-threaded and must not pay CAS overhead. The serializer below
  // (g_green_mutator_lock) is released only when GT_M2_PARALLEL is set, so that
  // flag alone turns "serialized CAS" into "real parallel CAS"; the lock stays
  // the correctness baseline for GT_NUM_WORKERS==1 / legacy serial compare.
  const bool parallel = (::getenv("GT_M2_PARALLEL") != nullptr);
  vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(green, true);
  // Item 1 (I3) / item 2: announce the stop-the-world GC BEFORE spawning the
  // workers, so their entry safepoints observe the request and park before
  // allocating. This quiescence (not the old
  // GreenThreadCOWSuspendedConcurrentGC suppression) is what makes the
  // synchronous collection below safe: the collector walks only published,
  // reported, quiescent roots.
  if (force_gc_mode != M2ForcedGCMode::kNone) {
    vi::GreenThreadCow::RequestAllMutatorsStop(nullptr, nullptr);
  }
  {
    v8::Isolate::Scope green_scope(green);

    // Item 4: a thread-safe claim queue drives the fan-out. Unlike the
    // GreenScheduler global runqueue (accessed only from the embedding thread,
    // so it needs no sync), this pool is driven by n real OS threads that must
    // claim work concurrently, so it is mutex + condition-variable protected.
    // Each worker runs its claimed body to completion -- the recursive
    // GreenScheduler::Yield entry() is deliberately not used here (prohibited in
    // parallel mode).
    M2ProbeWorkQueue pool(&works);

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      workers.emplace_back([&]() {
        Work w;
        // Claim exactly one body; a worker exits without running only if the
        // queue is already drained (never happens here -- n bodies for n
        // workers -- but keeps the claim contract sound for lopsided pools).
        if (!pool.WaitAndClaim(&w)) return;
        // (a) Enter the green isolate: green_thread_mode_ keeps this worker's
        // entry chain on its own PerIsolateThreadData (M1).
        // Item 1 (I3) / item 2: the single per-worker safepoint on this path.
        // If a forced GC is in progress, PollForGCRequest publishes this
        // worker's roots and parks it here (before the mutator lock + body)
        // until the coordinator resumes it; otherwise it returns immediately.
        // Either way we fall through to the body, which runs only after the
        // coordinator has collected and resumed.
        if (force_gc_mode != M2ForcedGCMode::kNone) {
          vi::GreenThreadCow::PollForGCRequest(
              w.gt, reinterpret_cast<vi::Isolate*>(green));
        }
        // Item 4: when GT_M2_PARALLEL is set the CAS above already guarantees
        // disjoint young-gen pointers, so release the serializer and let the N
        // workers allocate -- and mutate shared-writer state (serialized by the
        // Item 3 GreenThreadCowSharedWriterScope) -- concurrently. Otherwise
        // hold the lock across the whole body so at most one OS worker mutates
        // the green heap at a time (the serial correctness baseline). The park
        // above happens before this lock, so a forced-GC worker never holds it
        // while parked at its safepoint.
        std::unique_lock<std::mutex> mutator_guard(
            g_green_mutator_lock, std::defer_lock);
        if (!parallel) mutator_guard.lock();
        // (a) Enter the green isolate: green_thread_mode_ keeps this worker's
        // entry chain on its own PerIsolateThreadData (M1). The isolate scope is
        // taken INSIDE the mutator lock on purpose: SetIsolateThreadLocals writes
        // the shared isolate_data_.execution_state_base_ mirror that the green JS
        // entry arm loads x25 (thread_local_top base) from, and a sibling worker
        // still draining out of its parked state (its scope would otherwise run
        // here, OUTSIDE the lock) clobbers that mirror mid-body, arming the
        // running worker's re-entries x25 to a not-yet-context-entered block ->
        // SEGV reading a freed context(). Holding the lock serializes every
        // mirror write to the one running worker's own block.
        v8::Isolate::Scope worker_scope(w.data->isolate);
        // (b) Worker-local attribution (I1): set BEFORE the body so the CoW
        // barrier attributes every store to THIS worker's GT.
        vi::SetCurrentGreenThread(w.gt);
        // (c) Run the body: GreenJSEntry switches to the green isolate and calls
        // the body, setting data->result_p / data->threw.
        GreenJSEntry(reinterpret_cast<vi::Isolate*>(green), w.gt);
        ok_p[w.idx] =
            static_cast<char>(!w.data->threw && !w.data->result_p.IsEmpty());
        if (mutator_guard.owns_lock()) mutator_guard.unlock();
      });
    }
    // Item 1 (I3) / item 2: quiescence barrier -- spin-join every spawned
    // worker until it has published + parked at its entry safepoint, so the
    // collection below runs only when no mutator is mid-body. This is the
    // step that REPLACES GreenThreadCOWSuspendedConcurrentGC: without it the
    // collector could start before the workers park and observe torn state.
    if (force_gc_mode != M2ForcedGCMode::kNone) {
      vi::GreenThreadCow::WaitForAllMutatorsParked(2000);
    }

  // Item 1 (I3) / item 2: run the forced young + full collection now that
  // every worker is parked at its entry safepoint (published + quiescent),
  // so no mutator is mid-body. This is the acceptance gate for the handshake.
  if (force_gc_mode != M2ForcedGCMode::kNone) {
    M2ForceGC(green, force_gc_mode);
  }
   // Resume the parked workers so the joined bodies can run and finish.
   if (force_gc_mode != M2ForcedGCMode::kNone) {
     vi::GreenThreadCow::ResumeAllMutators();
   }
   // Release any worker still parked in WaitAndClaim, then join the pool.
   pool.Shutdown();
   for (std::thread& wt : workers) wt.join();
  }

  vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(green, false);

  // 3. Copy results back to the embedding isolate.
  Local<Array> results = Array::New(isolate, n);
  bool all_ok = true;
  {
    v8::Isolate::Scope green_scope(green);
    HandleScope green_handle_scope(green);
    GreenValueCopier out_copier{green, isolate};  // green -> embedding
    for (int i = 0; i < n; ++i) {
      if (!ok_p[i]) {
        all_ok = false;
        continue;
      }
      Local<Value> green_result = Local<Value>::New(green, works[i].data->result_p);
      std::unique_ptr<Persistent<Value>> emb = CopyValueAcross(out_copier, green_result);
      results->Set(ctx, i, Local<Value>::New(isolate, *emb)).Check();
    }
  }

   // Item 2: every rooted per-worker marker must have survived the forced GC
   // (still allocated, still carrying its original id). This MUST be checked
   // before teardown below, which resets marker_p -- the assertion only holds
   // on the forced-GC path, so it is false when the knob is unset.
   bool post_gc_consistent = (force_gc_mode == M2ForcedGCMode::kNone);
   if (force_gc_mode != M2ForcedGCMode::kNone) {
     post_gc_consistent = true;
     for (int i = 0; i < n; ++i) {
       if (works[i].data->marker_p.IsEmpty()) {
         post_gc_consistent = false;
         break;
       }
       v8::Isolate::Scope marker_scope(green);
       v8::HandleScope marker_hs(green);
       Local<Context> marker_ctx =
           Local<Context>::New(green, works[i].data->context_p);
       // marker_p roots an Object on the green heap (see the allocation above):
       // { id: <worker index> }. The per-worker marker must survive the forced
       // collection intact -- still allocated AND still carrying its original id.
       // (This is the I3 acceptance assertion: a rooted marker must not be swept.
       // Only meaningful on the forced-GC path, so it is false when unset.)
       Local<Value> marker_val = works[i].data->marker_p.Get(green);
       if (marker_val.IsEmpty() || !marker_val->IsObject()) {
         post_gc_consistent = false;
         break;
       }
       Local<Value> id_val = marker_val.As<Object>()->Get(
           marker_ctx, String::NewFromUtf8(green, "id").ToLocalChecked())
         .ToLocalChecked();
       if (!id_val->IsNumber() ||
           id_val->NumberValue(marker_ctx).ToChecked() !=
               static_cast<double>(i)) {
         post_gc_consistent = false;
         break;
       }
     }
   }

  // 4. Teardown CoW + free per-worker state (mirrors RunTasks' cleanup).
  for (Work& w : works) {
    vi::GreenThreadCow::Teardown(w.gt);
    g_gt_registry.erase(w.gt);
    w.data->args_p.clear();
    w.data->result_p.Reset();
    w.data->fn_p.Reset();
     w.data->marker_p.Reset();
    delete w.data;
    delete w.gt;
  }


  Local<Object> report = Object::New(isolate);
  report->Set(ctx, String::NewFromUtf8(isolate, "results").ToLocalChecked(), results).Check();
  report->Set(ctx, String::NewFromUtf8(isolate, "all_ok").ToLocalChecked(),
              Boolean::New(isolate, all_ok)).Check();
   report->Set(ctx, String::NewFromUtf8(isolate, "post_gc_consistent").ToLocalChecked(),
               Boolean::New(isolate, post_gc_consistent)).Check();
  args.GetReturnValue().Set(report);
}
// Cooperative checkpoint: hands control to the next green thread and returns
// when scheduling returns here. No-op when called outside a running green
// thread (there is nothing to yield to).
void Yield(const FunctionCallbackInfo<Value>& args) {
  vi::GreenThread* gt = vi::CurrentGreenThread();
  if (gt == nullptr) return;  // Not inside a green thread; nothing to yield.
  // Cooperative interleave runs on the same OS worker that the body is pinned
  // to; g_current_pool_scheduler is null outside a pool worker.
  vi::GreenScheduler* scheduler = g_current_pool_scheduler;
  if (scheduler == nullptr) return;
  scheduler->Yield();
  args.GetReturnValue().SetUndefined();
}

// green_threads.addrOf(obj) -> BigInt
// Returns the untagged heap address of a heap object handle. Used to record the
// main-heap view of a shared object before a green thread copies it.
void AddrOf(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);
  if (args.Length() < 1 || !args[0]->IsObject()) {
    args.GetReturnValue().SetUndefined();
    return;
  }
  Address addr = UntaggedObjectAddress(Local<Object>::Cast(args[0]));
  args.GetReturnValue().Set(
      BigInt::NewFromUnsigned(isolate, static_cast<uint64_t>(addr)));
}

// green_threads.cowCloneOf(obj) -> BigInt | undefined
// Returns the current green thread's private copy of `obj` once it has been
// cloned on write, or undefined while `obj` remains shared. Lets JavaScript
// prove copy-on-write isolation by comparing shared vs. local addresses.
void CowCloneOf(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);
  if (args.Length() < 1 || !args[0]->IsObject()) {
    args.GetReturnValue().SetUndefined();
    return;
  }
  vi::GreenThread* gt = vi::CurrentGreenThread();
  if (gt == nullptr || !gt->cow_enabled() || !gt->cow_ref_table()) {
    args.GetReturnValue().SetUndefined();
    return;
  }
  Address addr = UntaggedObjectAddress(Local<Object>::Cast(args[0]));
  Address copy = vi::kNullAddress;
  if (gt->cow_ref_table()->FindCopy(addr, &copy)) {
    args.GetReturnValue().Set(
        BigInt::NewFromUnsigned(isolate, static_cast<uint64_t>(copy)));
  } else {
    args.GetReturnValue().SetUndefined();
  }
}

// Build the green `g` shell: a green-isolate-bound object exposing the same
// native methods as the embedding `green_threads` module, so a green body's free
// variable `g` resolves to a callable binding instead of throwing ReferenceError.
//
// Each method is a fresh JS function bound to the shared C++ callback, so when
// the body calls g.<method> while executing on the green isolate, args.GetIsolate()
// returns the green isolate and the callback operates on the current green
// thread's own state (CoW ref table, scheduler, ...). The object is installed on
// the green context's global, so it resolves as a free variable in any body
// reparsed onto this context.
// Green `performance.now()` shim (see InstallGreenGlobalsShell). The green
// isolate has no Node perf_hooks, so a captured `performance` namespace -- which
// carries the native `now` that cannot be copied across isolates -- resolves to
// this object instead, backed by a process-monotonic millisecond clock. Enough
// for the CPU-timing harnesses that run inside bodies.
static void GreenPerformanceNow(const FunctionCallbackInfo<Value>& args) {
  Isolate* iso = args.GetIsolate();
  std::chrono::duration<double, std::milli> ns =
      std::chrono::steady_clock::now().time_since_epoch();
  args.GetReturnValue().Set(Number::New(iso, ns.count()));
}

// Green assert shims. A body that captures the `assert` namespace resolves it to
// this object (assert's native methods cannot be copied across isolates). Each
// throws a TypeError when the assertion fails, mirroring node:assert's contract.
static void GreenAssertOk(const FunctionCallbackInfo<Value>& args) {
  Isolate* iso = args.GetIsolate();
  if (args[0]->IsUndefined() || args[0]->IsNull() || args[0]->IsFalse()) {
    std::string msg = "assert.ok failed";
    if (args.Length() >= 2 && args[1]->IsString()) {
      String::Utf8Value m(iso, args[1]);
      if (*m) msg = *m;
    }
    iso->ThrowException(v8::Exception::TypeError(String::NewFromUtf8(
        iso, msg.c_str(), v8::NewStringType::kNormal).ToLocalChecked()));
  }
}

static void GreenAssertEqual(const FunctionCallbackInfo<Value>& args) {
  Isolate* iso = args.GetIsolate();
  if (!args[0]->StrictEquals(args[1])) {
    std::string msg = "assert.equal failed";
    if (args.Length() >= 3 && args[2]->IsString()) {
      String::Utf8Value m(iso, args[2]);
      if (*m) msg = *m;
    }
    iso->ThrowException(v8::Exception::TypeError(String::NewFromUtf8(
        iso, msg.c_str(), v8::NewStringType::kNormal).ToLocalChecked()));
  }
}

static void GreenAssertNotEqual(const FunctionCallbackInfo<Value>& args) {
  Isolate* iso = args.GetIsolate();
  if (args[0]->StrictEquals(args[1])) {
    std::string msg = "assert.notEqual failed";
    if (args.Length() >= 3 && args[2]->IsString()) {
      String::Utf8Value m(iso, args[2]);
      if (*m) msg = *m;
    }
    iso->ThrowException(v8::Exception::TypeError(String::NewFromUtf8(
        iso, msg.c_str(), v8::NewStringType::kNormal).ToLocalChecked()));
  }
}

static void InstallGreenGlobalsShell(Local<Context> green_ctx) {
  v8::Isolate::Scope isolate_scope(g_green_isolate);
  HandleScope scope(g_green_isolate);
  // Suspend green-isolate marking across the allocate + install window so GC
  // cannot move the freshly-created object and its functions while we root them.
  GreenGCPause gc(g_green_isolate);
  Context::Scope ctx_scope(green_ctx);
  Local<Object> g_shell = Object::New(g_green_isolate);
  auto set_method = [&](const char* name,
                        void (*cb)(const FunctionCallbackInfo<Value>&)) {
    Local<Function> f =
        FunctionTemplate::New(g_green_isolate, cb)
            ->GetFunction(green_ctx)
            .ToLocalChecked();
    Local<String> key = String::NewFromUtf8(g_green_isolate, name,
                                            v8::NewStringType::kNormal)
                            .ToLocalChecked();
    g_shell->Set(green_ctx, key, f).Check();
  };
  set_method("spawn", Spawn);
  set_method("spawnAll", SpawnAll);
  set_method("yield", Yield);
  set_method("addrOf", AddrOf);
  set_method("cowCloneOf", CowCloneOf);
  g_shell->Set(
      green_ctx,
      String::NewFromUtf8(g_green_isolate, "version",
                          v8::NewStringType::kNormal)
          .ToLocalChecked(),
      String::NewFromUtf8(g_green_isolate, "1",
                          v8::NewStringType::kNormal)
          .ToLocalChecked())
      .Check();
  // A body that references the green_threads binding as a free variable resolves
  // it against a green global. The captured name is arbitrary (here `g`, but
  // examples name it `green_threads`), so install the shell under every alias;
  // the closure lift deliberately skips function-bearing captured values because
  // their native methods can't be copied into green without reusing an embedding
  // SFI (an F1 hazard). The shell provides working green implementations.
  auto set_global = [&](const char* gname, Local<Object> gobj) {
    green_ctx->Global()->Set(
        green_ctx,
        String::NewFromUtf8(g_green_isolate, gname,
                            v8::NewStringType::kNormal)
            .ToLocalChecked(),
        gobj).Check();
  };
  set_global("g", g_shell);
  set_global("green_threads", g_shell);
  set_global("gt", g_shell);
  // Green shims for the Node namespaces bodies commonly capture that carry native
  // methods (performance, assert). Without these a body using, say,
  // `assert.notEqual(...)` would hit "assert is not defined". Built as plain
  // Objects (matching the `g` shell above) so their function members install via
  // Object::Set, which accepts instantiated Functions.
  {
    HandleScope ho(g_green_isolate);
    Local<Object> perf = Object::New(g_green_isolate);
    Local<Function> now_fn =
        FunctionTemplate::New(g_green_isolate, GreenPerformanceNow)
            ->GetFunction(green_ctx)
            .ToLocalChecked();
    perf->Set(
        green_ctx,
        String::NewFromUtf8(g_green_isolate, "now",
                            v8::NewStringType::kNormal)
            .ToLocalChecked(),
        now_fn).Check();
    set_global("performance", perf);
  }
  {
    HandleScope ho(g_green_isolate);
    Local<Object> assert_obj = Object::New(g_green_isolate);
    auto am = [&](const char* m,
                  void (*cb)(const FunctionCallbackInfo<Value>&)) {
      Local<Function> fn =
          FunctionTemplate::New(g_green_isolate, cb)
              ->GetFunction(green_ctx)
              .ToLocalChecked();
      assert_obj->Set(
          green_ctx,
          String::NewFromUtf8(g_green_isolate, m,
                              v8::NewStringType::kNormal)
              .ToLocalChecked(),
          fn).Check();
    };
    am("ok", GreenAssertOk);
    am("equal", GreenAssertEqual);
    am("notEqual", GreenAssertNotEqual);
    set_global("assert", assert_obj);
  }
}

// ====================================================================
// Binding registration
// ====================================================================

void InitGreenThreads(Local<Object> exports,
                       Local<Value> module,
                       Local<Context> context,
                       void* priv) {
  Isolate* isolate = Isolate::GetCurrent();
  HandleScope scope(isolate);
  // Register the green-isolate disposal so it runs before V8::Dispose() at
  // process exit (see DisposeGreenIsolate). Environment::GetCurrent is safe
  // here: the module is loaded inside an Environment's realm.
  Environment* env = Environment::GetCurrent(context);
  if (env != nullptr) {
    env->AtExit(DisposeGreenIsolate, nullptr);
  }
  node::SetMethod(context, exports, "spawn", Spawn);
  node::SetMethod(context, exports, "spawnAll", SpawnAll);
  node::SetMethod(context, exports, "yield", Yield);
  node::SetMethod(context, exports, "addrOf", AddrOf);
  node::SetMethod(context, exports, "cowCloneOf", CowCloneOf);
  node::SetMethod(context, exports, "__concurrent_entry", ConcurrentEntry);
  node::SetMethod(context, exports, "__m2_probe", M2Probe);
  node::SetMethod(context, exports, "version",
                  [](const FunctionCallbackInfo<Value>& args) {
                    args.GetReturnValue().Set(String::NewFromUtf8(
                        args.GetIsolate(), "1",
                        v8::NewStringType::kNormal).ToLocalChecked());
                  });
}

}  // namespace node

NODE_BINDING_CONTEXT_AWARE_INTERNAL(green_threads, node::InitGreenThreads)

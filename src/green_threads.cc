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
#include <execinfo.h>
#include <csignal>
#include <signal.h>
#include <cstring>
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

#include <fcntl.h>
#include <unistd.h>

#include "node_binding.h"
#include "util.h"

#include "uv.h"

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

// Diagnostic-only: install during an M2Probe run to capture the location and
// green-thread phase of a fatal SIGSEGV/SIGABRT (macOS writes no core). Prints
// phase + pthread + a backtrace, then re-raises for the default termination so
// the exit code stays 139. Guarded so it never leaks into a release build.
#ifdef V8_ENABLE_GREEN_THREADS
namespace {
bool gt_faulting = false;
void gt_fault_handler(int sig, siginfo_t* si, void* ucp) {
  (void)ucp;
  if (gt_faulting) return;  // avoid recursion / re-entrancy
  gt_faulting = true;
  const char* phase = ::getenv("GT_PHASE");
  ::fprintf(stderr,
            "\n[SIGFAULT] sig=%d phase=%s pthread=0x%lx fault_addr=%p\n",
            sig, phase ? phase : "idle", (unsigned long)pthread_self(),
            si ? (void*)si->si_addr : (void*)0);
  void* bt[48];
  int n = ::backtrace(bt, 48);
  ::backtrace_symbols_fd(bt, n, 2);
  ::fflush(stderr);
  struct sigaction sa;
  ::memset(&sa, 0, sizeof(sa));
  sa.sa_handler = SIG_DFL;
  ::sigaction(sig, &sa, nullptr);
  ::raise(sig);
}

// RAII: installs gt_fault_handler for the wrapping scope and restores the
// previous disposition on exit.
class GTFaultGuard {
 public:
  GTFaultGuard() {
    ::memset(&sa_impl_, 0, sizeof(sa_impl_));
    sa_impl_.sa_sigaction = gt_fault_handler;
    sa_impl_.sa_flags = SA_SIGINFO;
    ::sigaction(SIGSEGV, &sa_impl_, &prev_);
  }
  ~GTFaultGuard() { ::sigaction(SIGSEGV, &prev_, nullptr); }
 private:
  struct sigaction sa_impl_{};
  struct sigaction prev_{};
};
}  // namespace
#endif

namespace node {
// Forward declaration of the green `g` shell installer, defined below in node
// scope. LiftEmbeddingGlobals (in the anonymous namespace) calls it, so this
// lives here in node scope to match the definition's linkage/scope.
static void InstallGreenGlobalsShell(v8::Local<v8::Context> green_ctx);

namespace {

// Forward declaration (anonymous-namespace scope, matching the definition below
// so the mangled names agree). Called by EnsureGreenContext.
static void InstallGreenAsyncGlobals(v8::Local<v8::Context> green_ctx);

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
using v8::Promise;
using v8::Script;
using v8::String;
using v8::Undefined;
using v8::Value;

using vi::Address;  // v8::internal::Address (untagged heap pointer)

// Forward declaration: GreenIoWork (defined further below) is referenced by the
// GreenCallData.pending_reads vector. A declaration suffices -- unique_ptr /
// vector only need the complete type at destruction, which happens where
// GreenIoWork is fully defined.
struct GreenIoWork;
struct GreenCallData;

// One pending bridged embedding-Promise settle. A green async body awaiting a
// bridged emb Promise (e.g. `await fs.promises.readFile(...)`) does NOT suspend
// on an emb Promise directly -- the bridge hands the body a fresh GREEN promise
// GP and returns it immediately, so the body's await parks on GP. The body's
// owning drive worker later resolves GP (into its own green queue) once the emb
// Promise settles. The settle value lives in an emb-rooted Persistent
// (args[0] of the reaction, which runs on emb where the promise lives); the
// green resolve + marshal to green happen on the drive worker, matching the
// gtRead model (GreenIoWork stores buf/err on a libuv pool thread and the
// owning drive worker reads + resolves it after Reactivate).
struct GreenEmbReact {
  GreenCallData* data;             // Owning body (never freed for its lifetime).
  vi::GreenScheduler* sched;       // Shared async-drive scheduler.
  Persistent<Context> green_ctx_p; // Body's green context (resolve GP here).
  Persistent<Promise::Resolver> gp_res_p; // Green resolver GP = gp_res_p->GetPromise().
  Persistent<Value> settle_val_p;   // emb-rooted: fulfilled value or rejection reason.
  std::atomic<bool> emb_settled{false}; // release by reaction, acquire by drive worker.
  bool is_error;                    // Set at creation: false=fulfilled, true=rejected.
};

// Per-green-thread JavaScript invocation state. A single instance is shared
// between the scheduler's C++ entry point and the running spawnAll call.
struct GreenCallData {
  Isolate* isolate = nullptr;
  Persistent<Context> context_p;
  // Outstanding green-async reads (gtRead) submitted by this body, in
  // submission order. GreenGtRead appends the GreenIoWork here; the body's
  // owning drive worker resolves each finished read into the body's own queue
  // and frees it. Thread-local g_green_driving_body routes each submission to
  // the correct body; a body's reads are appended only by its setup (embedding
  // thread) and then its owning drive worker, never concurrently, so the vector
  // needs no lock. Kept alive on data (freeing it also frees the reads) until
  // the async teardown loop below clears it.
  std::vector<std::unique_ptr<GreenIoWork>> pending_reads;
  // Outstanding bridged emb-Promise settles (GreenEmbReact), submitted by this
  // body when a bridged method returns an emb Promise (e.g. fs.promises.*). The
  // owning drive worker resolves each settled GP into the body's green queue;
  // like pending_reads, appended only by the setup/green-thread that owns the
  // body and drained only by its owning drive worker -- never concurrently.
  std::vector<std::unique_ptr<GreenEmbReact>> pending_emb;
  // Owned per green-async body: its own MicrotaskQueue. Each async body drives
  // its continuations from its own queue (via Context::GetMicrotaskQueue), so
  // N workers can resume disjoint bodies concurrently without racing one shared
  // queue (DriveGreenAsyncBodiesParallel's per-worker body ownership). The embedder must
  // keep the queue alive until the context is gone; struct-destruction order
  // (context_p declared above) disposes the context first, then this.
  std::unique_ptr<v8::MicrotaskQueue> body_mq_p;
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
  // SpawnAll round id (per RunTasks call), stamped here and echoed into every
  // GT_CRASH_DEBUG print so worker-thread RESULT lines and the main-thread
  // RUNTASKS lines can be correlated to the same spawnAll.
  uint64_t round = 0;
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

  // --- Async (await) support ---
  // Set for a body whose SharedFunctionInfo is an async function (async fn /
  // async arrow / async generator / async concise method). Such a body runs on
  // the EMBEDDING isolate, where Node's async I/O bindings (setTimeout /
  // process / require) actually live -- the fresh green realm cannot host them
  // (its lifted setTimeout calls the Node-internal validateFunction, which is
  // not a global, so it ReferenceErrors). A sync body keeps running green, so
  // genuine compute still parallelizes on the green isolate; only async bodies
  // -- which are inherently serial around I/O -- run on embedding. Its result
  // row in the returned array is a caller-facing Promise resolved (or rejected)
  // by reactions chained onto the body promise and driven by the normal Node
  // event loop. No green parking, drain, or resumer is involved.
  bool awaiting = false;
  // Rooted on data->isolate (the embedding isolate for an async body): the
  // Resolver of the caller-facing Promise handed to the JS caller.
  Persistent<Promise::Resolver> caller_resolver_p;
  Persistent<Context> caller_ctx_p;

  // --- Green-async (await) support ---
  // An async body in shared-heap mode runs on the GREEN isolate (not the
  // embedding isolate). V8 saves an async function's JS stack in its own
  // closure, so when the body parks at `await`, fn->Call returns a pending
  // promise and the C++ frame is already gone -- the body is later resumed by
  // driving the green isolate's microtask queue (RunMicrotasks), with external
  // triggers (timers / I/O) progressed by running the embedding libuv loop.
  // The scheduler/pool only ever drives SYNC bodies, so green-async bodies
  // (async_green) are driven directly here. A green-async body (Option C): an
  // async function whose body runs on the GREEN isolate, so its control flow and
  // compute stay green and `await` cooperatively parks it (its continuation lives
  // in the async closure while fn->Call has returned a pending promise). It holds
  // a bare GreenThread* descriptor (created in SetupAndRunGreenAsyncBody) used
  // ONLY by the cooperative block/resume primitives to free the OS core while
  // parked on I/O -- NOT a runnable green thread and no COW view. `awaiting`
  // stays true so the pool never creates a runnable green thread / COW view for
  // one and the sync teardown loop skips it.
  bool async_green = false;
  // Rooted on data->isolate (the green isolate): the body's pending promise,
  // held from fn->Call until the drive phase settles it.
  Persistent<Promise> async_promise_p;
  // Rooted on the embedding isolate: the resolver of the caller-facing Promise
  // handed to the JS caller. The green body promise settles in the drive loop,
  // at which point a green reaction (GreenGreenCallerResolveFulfilled/Rejected)
  // Enters this isolate and resolves it. `caller_isolate` records which one.
  Isolate* caller_isolate = nullptr;
  // Set once the body promise has settled (a reaction fired and captured the
  // settle value). The drive loop waits for every green-async body to reach
  // this before returning so the caller Promises handed out are settled.
  bool async_settled = false;
  // The value the green body promise settled with (return value if fulfilled,
  // rejection reason if rejected) -- captured by the green settle reaction as a
  // GREEN Persistent (args[0] is a green handle). RunTasks copies it into the
  // embedding isolate and resolves the caller resolver with it.
  Persistent<Value> settle_value_p;
};

// Registry from green thread to its call data. The scheduler entry point is
// keyed by the GreenThread pointer the scheduler hands back on every call.
std::unordered_map<vi::GreenThread*, GreenCallData*> g_gt_registry;
// Monotonic spawnAll round id, stamped into GreenCallData::round so that
// GT_CRASH_DEBUG output can correlate a worker's RESULT write with the main
// thread's RUNTASKS read for the exact same spawnAll().
static std::atomic<uint64_t> g_green_spawn_round{0};

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

// Embedding loader for the green `require` bridge. `require`/`module` are module
// scoped (not on the embedding realm global in file-entry), so green bodies have
// no working require of their own in shared-heap mode. We capture ONE working
// loader once -- `process.mainModule.require`, which is bound to the entry module
// and loads node: builtins -- and root it on the embedding isolate. Captured
// lazily on first GreenBridgeRequire from the embedding realm; the bridge reads
// it on every call.
static Isolate* g_green_emb_loader_isolate = nullptr;
static Persistent<Value> g_green_emb_loader_p;
// The embedding isolate + realm context captured alongside the loader, so the
// green bridge can root emb-handles (the emb require-call receiver, the emb
// promise Then-reactions) and reconstruct the emb context from a persistent.
static Isolate* g_green_embedding_isolate = nullptr;
static Persistent<Context> g_green_emb_ctx_p;

// Capture (once) a working embedding loader and root it on the embedding
// isolate. require/module are module-scoped (not global) in file-entry, so we
// reach the entry module's own `require` through the always-global `process`:
// `process.mainModule.require` is `Module.prototype.require` bound to the entry
// module and loads node: builtins. Returns false (leaving loader unset) only if
// the chain is somehow unavailable -- GreenBridgeRequire then reports clearly.
static void GreenCaptureEmbLoader(Isolate* emb, Local<Context> emb_ctx) {
  if (!g_green_emb_loader_p.IsEmpty()) return;
  {
    v8::Isolate::Scope iso_scope(emb);
    HandleScope hs(emb);
    v8::Context::Scope cs(emb_ctx);
    Local<Value> proc;
    Local<String> pkey =
        String::NewFromUtf8(emb, "process", v8::NewStringType::kNormal)
            .ToLocalChecked();
    if (!emb_ctx->Global()->Get(emb_ctx, pkey).ToLocal(&proc) ||
        !proc->IsObject())
      return;
    // proc.mainModule.require (mainModule may be absent in some runtimes).
    Local<Value> main_mod;
    Local<String> mkey =
        String::NewFromUtf8(emb, "mainModule", v8::NewStringType::kNormal)
            .ToLocalChecked();
    if (!Local<Object>::Cast(proc)->Get(emb_ctx, mkey).ToLocal(&main_mod) ||
        !main_mod->IsObject())
      return;
    Local<Value> req;
    Local<String> rkey =
        String::NewFromUtf8(emb, "require", v8::NewStringType::kNormal)
            .ToLocalChecked();
    if (!Local<Object>::Cast(main_mod)->Get(emb_ctx, rkey).ToLocal(&req) ||
        !req->IsFunction())
      return;
    g_green_emb_loader_isolate = emb;
    g_green_emb_loader_p.Reset(emb, Local<Function>::Cast(req));
    g_green_embedding_isolate = emb;
    g_green_emb_ctx_p.Reset(emb, emb_ctx);
  }
}

// Embedding-promise settle reaction. Fires ON the embedding realm (the emb
// Promise lives there, so its reactions run there when its event loop progresses
// it). Records the settle value (args[0]) into an emb-rooted Persistent and flags
// settlement, then Reactivates the owning body's drive worker so the drive loop
// can resolve the GREEN promise GP the body awaited. Like GreenIoExecute handing
// buf/err back from a libuv pool thread, this only touches emb state + the
// reactivation signal -- all green work happens later on the owning drive worker.
static void GreenEmbPromiseReaction(const FunctionCallbackInfo<Value>& args) {
  GreenEmbReact* r = static_cast<GreenEmbReact*>(
      v8::External::Cast(*args.Data())->Value());
  v8::Isolate* emb = args.GetIsolate();
  v8::HandleScope hs(emb);
  r->settle_val_p.Reset(emb, args[0]);
  r->emb_settled.store(true, std::memory_order_release);
  if (r->sched != nullptr && r->data->gt != nullptr)
    r->sched->ReactivateGreenThread(r->data->gt);
}

// One bridged embedding method, exposed to green as a plain JS function. Its
// callback marshals args green->emb, calls the real emb method, and marshals the
// result emb->green. A Promise result is bridged to a fresh GREEN promise (GP)
// the body awaits (see GreenBridgeRequire/GreenEmbReact): the body suspends on GP
// while the emb Promise runs its async I/O on emb, and the drive loop resolves GP
// + Reactivates the body once the emb reaction settles.
struct GreenBridgeMethod {
  Persistent<Value> method_p;    // emb: the function to call.
  Persistent<Value> receiver_p;  // emb: the object it is a property of (this).
};

  // Per-OS-thread pointer to the green-async body whose continuation is currently
  // being driven on this thread. GreenGtRead installs each read here so the body's
  // owning worker can find and resolve it in the body's own queue. Set by the
  // embedding setup thread around a body's first await and by each drive worker
  // around that body's continuations; thread-local so distinct workers never
  // clobber each other's target.
  static thread_local GreenCallData* g_green_driving_body = nullptr;

  // Process-global pointer to the ONE shared GreenScheduler that drives the
  // green-async body pool (created by RunTasks and live for the whole spawnAll).
  // A green-async body's gtRead submits its read against this scheduler and its
  // completion (a libuv pool thread) reactivates the body through it, so the
  // drive worker can park until the I/O finishes. It is a plain (not thread-local)
  // global on purpose: the embedding setup thread, every spawned drive worker, and
  // GT_TRUE_PARALLEL's warm-up all read the same stable value for the run --
  // GreenIoExecute needs no global at all, since it reactivates via the per-read
  // owner pointer captured at submission.
  static vi::GreenScheduler* g_async_drive_scheduler = nullptr;

struct GreenValueCopier {
  Isolate* src;  // Isolate the source handle lives on.
  Isolate* dst;  // Isolate to rebuild the value on.
};

// Forward declaration: defined below; GreenBridgeMethodCallback (just after)
// uses it to marshal args/results across the green/embedding isolate boundary.
static std::unique_ptr<Persistent<Value>> CopyValueAcross(
    const GreenValueCopier& copier, Local<Value> value);

static void GreenBridgeMethodCallback(const FunctionCallbackInfo<Value>& args) {
  GreenBridgeMethod* m =
      static_cast<GreenBridgeMethod*>(v8::External::Cast(*args.Data())->Value());
  v8::Isolate* green = args.GetIsolate();
  v8::HandleScope hs(green);
  Local<Context> gctx = green->GetCurrentContext();
  const int argc = args.Length();

  Isolate* emb = g_green_emb_loader_isolate;
  Local<Context> emb_ctx = Local<Context>::New(emb, g_green_emb_ctx_p);

  // Marshal each green arg to embedding and unbox into emb Locals held in
  // emb_scope (below) so they stay rooted for the whole Call. CopyValueAcross
  // manages its own isolate/handle scopes internally and returns a durable
  // emb-rooted Persistent; we unbox it here while emb_scope is live.
  v8::TryCatch tc(emb);
  {
    v8::Isolate::Scope iso_scope(emb);
    HandleScope emb_hs(emb);
    v8::Context::Scope emb_ctx_scope(emb_ctx);

    std::vector<Local<Value>> emb_argv;
    emb_argv.reserve(argc);
    for (int i = 0; i < argc; i++) {
      GreenValueCopier copier{green, emb};
      auto cp = CopyValueAcross(copier, args[i]);
      emb_argv.push_back(Local<Value>::New(emb, *cp));
    }

    Local<Function> meth =
        Local<Function>::Cast(Local<Value>::New(emb, m->method_p));
    Local<Value> recv = Local<Value>::New(emb, m->receiver_p);
    Local<Value> result;
    if (!meth->Call(emb_ctx, recv, argc, emb_argv.data()).ToLocal(&result)) {
      // Embedding method threw: copy the exception back and rethrow on green.
      Local<Value> exc = tc.HasCaught() ? tc.Exception() : v8::Undefined(emb);
      {
        GreenValueCopier copier{emb, green};
        auto ep = CopyValueAcross(copier, exc);
        v8::Isolate::Scope green_scope(green);
        green->ThrowException(Local<Value>::New(green, *ep));
        args.GetReturnValue().SetUndefined();
        return;
      }
    }

    // Promise result: bridge GP (green) to the body's await via a pending emb
    // settle. The body suspends on GP; the drive loop resolves it once the emb
    // Promise settles.
    if (result->IsPromise()) {
      Local<Context> bctx = gctx;
      Local<Promise::Resolver> gp_res;
      if (!Promise::Resolver::New(bctx).ToLocal(&gp_res)) {
        green->ThrowException(v8::Exception::Error(
            String::NewFromUtf8(green,
                "green_threads: cannot create green promise resolver",
                v8::NewStringType::kNormal).ToLocalChecked()));
        args.GetReturnValue().SetUndefined();
        return;
      }
      Local<Promise> gp = gp_res->GetPromise();
      Local<Promise> emb_p = Local<Promise>::Cast(result);
      const bool is_async = (g_green_driving_body != nullptr);
      if (!is_async) {
        // No body driving -- cannot suspend an await on GP. Fail clearly rather
        // than silently drop the async result.
        Local<Value> e = v8::Exception::Error(String::NewFromUtf8(
            green, "green_threads: async bridge called without a driving body",
            v8::NewStringType::kNormal).ToLocalChecked());
        green->ThrowException(e);
        args.GetReturnValue().SetUndefined();
        return;
      }
      GreenCallData* data = g_green_driving_body;
      vi::GreenScheduler* sched = g_async_drive_scheduler;
      auto make_react = [&](bool is_err) {
        GreenEmbReact* r = new GreenEmbReact();
        r->data = data;
        r->sched = sched;
        r->is_error = is_err;
        r->green_ctx_p.Reset(green, bctx);
        r->gp_res_p.Reset(green, gp_res);
        return r;
      };
      GreenEmbReact* ful = make_react(false);
      GreenEmbReact* rej = make_react(true);
      Local<v8::External> ef = v8::External::New(green, ful);
      Local<v8::External> er = v8::External::New(green, rej);
      Local<Function> fulfillFn =
          Function::New(emb_ctx, GreenEmbPromiseReaction, ef).ToLocalChecked();
      Local<Function> rejectFn =
          Function::New(emb_ctx, GreenEmbPromiseReaction, er).ToLocalChecked();
      (void)emb_p->Then(emb_ctx, fulfillFn, rejectFn).ToLocalChecked();
      data->pending_emb.push_back(
          std::unique_ptr<GreenEmbReact>(ful));
      data->pending_emb.push_back(
          std::unique_ptr<GreenEmbReact>(rej));
      args.GetReturnValue().Set(gp);
      return;
    }

    // Sync result: marshal emb -> green within a green scope, then return it.
    Local<Value> gres;
    {
      GreenValueCopier copier{emb, green};
      auto cp = CopyValueAcross(copier, result);
      v8::Isolate::Scope green_scope(green);
      HandleScope green_hs(green);
      gres = Local<Value>::New(green, *cp);
    }
    args.GetReturnValue().Set(gres);
    return;
  }
}

// Wrap an embedding-isolate value into a green proxy so a green body sees a
// working Node API. Functions become GreenBridgeMethod-backed green functions
// (each call round-trips through the embedding isolate); plain objects recurse
// (so nested namespaces like fs.promises.readFile resolve lazily); everything
// else (strings/numbers/arrays/primitive objects) copies via CopyValueAcross.
// `receiver` is the embedding object `v` is a property of and is used as `this`
// when `v` is a function.
static Local<Value> GreenEmbWrapValue(Isolate* green, Local<Context> gctx,
                                      Isolate* emb, Local<Context> emb_ctx,
                                      Local<Value> v, Local<Value> receiver) {
  // One green HandleScope roots every handle created in this call and in any
  // recursion below (including the green proxies returned), so nested results
  // survive until they are Set into their parent object.
  HandleScope top(green);
  v8::Isolate::Scope green_iso(green);
  v8::Context::Scope gc(gctx);
  if (v->IsFunction()) {
    GreenBridgeMethod* m = new GreenBridgeMethod();
    {
      v8::Isolate::Scope emb_iso(emb);
      HandleScope ehe(emb);
      v8::Context::Scope ecs(emb_ctx);
      m->method_p.Reset(emb, v);
      m->receiver_p.Reset(emb, receiver);
    }
    Local<v8::External> ext = v8::External::New(green, m);
    return v8::FunctionTemplate::New(green, GreenBridgeMethodCallback, ext)
        ->GetFunction(gctx).ToLocalChecked();
  }
  // Plain JS object: enumerate on emb, wrapping each child and building the
  // green object. A method child uses `v` as `this`; a nested object child is
  // wrapped recursively so its own methods use itself as `this`.
  if (v->IsObject() && !v->IsArray() && !v->IsDate() && !v->IsRegExp() &&
      !v->IsMap() && !v->IsSet() && !v->IsTypedArray() &&
      !v->IsArrayBuffer() && !v->IsPromise()) {
    Local<Object> gobj = Object::New(green);
    {
      v8::Isolate::Scope emb_iso(emb);
      HandleScope ehe(emb);
      v8::Context::Scope ecs(emb_ctx);
      Local<Object> vobj = v.As<Object>();
      Local<Array> keys = Local<Array>::Cast(
          vobj->GetPropertyNames(emb_ctx)
              .ToLocalChecked());
      for (uint32_t i = 0; i < keys->Length(); i++) {
        Local<String> key =
            Local<String>::Cast(keys->Get(emb_ctx, i).ToLocalChecked());
        Local<Value> child = vobj->Get(emb_ctx, key).ToLocalChecked();
        Local<Value> wrapped;
        if (child->IsFunction()) {
          wrapped = GreenEmbWrapValue(green, gctx, emb, emb_ctx, child, v);
        } else if (child->IsObject() && !child->IsArray() &&
                   !child->IsDate() && !child->IsRegExp() && !child->IsMap() &&
                   !child->IsSet() && !child->IsTypedArray() &&
                   !child->IsArrayBuffer() && !child->IsPromise()) {
          wrapped = GreenEmbWrapValue(green, gctx, emb, emb_ctx, child, child);
        } else {
          GreenValueCopier copier;
          copier.src = emb;
          copier.dst = green;
          auto p = CopyValueAcross(copier, child);
          v8::Isolate::Scope green_scope(green);
          wrapped = Local<Value>::New(green, *p);
        }
        gobj->Set(gctx, key, wrapped).Check();
      }
    }
    return gobj;
  }
  // Non-function value: copy to green.
  GreenValueCopier copier;
  copier.src = emb;
  copier.dst = green;
  auto p = CopyValueAcross(copier, v);
  v8::Isolate::Scope green_scope(green);
  return Local<Value>::New(green, *p);
}

// Green global `require(specifier)`: load a Node module on the embedding realm
// (via the captured process.mainModule.require) and return its exports wrapped
// as a green proxy. Marshals errors back to green so the body sees the real
// exception.
static void GreenBridgeRequire(const FunctionCallbackInfo<Value>& args) {
  Isolate* green = args.GetIsolate();
  HandleScope hs(green);
  Local<Context> gctx = green->GetCurrentContext();

  std::string name;
  if (args[0]->IsString()) {
    v8::String::Utf8Value u(green, args[0]);
    name.assign(*u, u.length());
  } else {
    args.GetReturnValue().Set(
        v8::Exception::TypeError(
            String::NewFromUtf8(green,
                "green_threads.require(specifier): specifier must be a string",
                v8::NewStringType::kNormal).ToLocalChecked()));
    return;
  }

  Isolate* emb = g_green_emb_loader_isolate;
  if (emb == nullptr || g_green_emb_loader_p.IsEmpty()) {
    GreenCaptureEmbLoader(emb ? emb : green, green->GetCurrentContext());
    if (emb == nullptr || g_green_emb_loader_p.IsEmpty()) {
      args.GetReturnValue().Set(v8::Exception::Error(String::NewFromUtf8(
          green, "green_threads.require: embedding loader unavailable",
          v8::NewStringType::kNormal).ToLocalChecked()));
      return;
    }
    emb = g_green_emb_loader_isolate;
  }
  Local<Context> emb_ctx = Local<Context>::New(emb, g_green_emb_ctx_p);

  Local<Value> mod;
  v8::TryCatch tc(emb);
  {
    v8::Isolate::Scope iso_scope(emb);
    HandleScope he(emb);
    v8::Context::Scope ecs(emb_ctx);
    Local<Function> loader =
        Local<Function>::Cast(Local<Value>::New(emb, g_green_emb_loader_p));
    Local<String> nameLocal = String::NewFromUtf8(
        emb, name.c_str(), v8::NewStringType::kNormal).ToLocalChecked();
    Local<Value> argv[1] = {nameLocal};
    if (!loader->Call(emb_ctx, emb_ctx->Global(), 1, argv).ToLocal(&mod)) {
      Local<Value> exc = tc.HasCaught() ? tc.Exception() : v8::Undefined(emb);
      {
        GreenValueCopier copier{emb, green};
        auto ep = CopyValueAcross(copier, exc);
        v8::Isolate::Scope green_scope(green);
        green->ThrowException(Local<Value>::New(green, *ep));
        args.GetReturnValue().SetUndefined();
        return;
      }
    }
  }

  args.GetReturnValue().Set(
      GreenEmbWrapValue(green, gctx, emb, emb_ctx, mod, mod));
}

// Set (per-OS-thread) while a green body executes inside GreenJSEntry, so the
// crash diagnostic can tell a body run (green isolate) apart from embedding-
// isolate bookkeeping that might read a shared object. Declared here rather
// than inline to avoid capturing state across the C call.
static thread_local bool g_in_green_body = false;

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
  // Install the green async I/O bridge (`gtRead`) on the green global so an
  // async body can `await` real embedding libuv I/O while it runs green.
  InstallGreenAsyncGlobals(ctx);
  g_green_ctx.Reset(g_green_isolate, ctx);
}

static void LiftCapturedValues(Isolate* emb_isolate,
                               Local<Function> emb_fn,
                               Local<Context> green_ctx);

// Relocate an embedding-isolate function onto the green isolate by
// re-parsing its source, so a green body runs as green Code and free
// variables resolve against the lifted green global. `green_ctx` is the context
// the body's source is compiled against and its captured free variables are
// lifted onto: for a sync body this is the durable shared g_green_ctx; for a
// green-async body it is that body's own context (so the body's continuations
// run with its own lift + its own gtRead).
// Returns the relocated body rooted in a Persistent on the green isolate. The
// caller owns this durable handle; no caller handle scope is required because
// the function is rooted inside this function before its own scope ends.
static std::unique_ptr<Persistent<Function>> RelocateFunction(Isolate* emb_isolate,
                                                              Local<Function> emb_fn,
                                                              Local<Context> green_ctx) {
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
  // Compile + run on the green isolate in green_ctx. For a sync body this is the
  // durable shared g_green_ctx; for a green-async body it is that body's OWN
  // context (SetupAndRunGreenAsyncBody hands it in), so the body's captured free
  // variables lift onto its own realm and its continuations run from its own
  // microtask queue. Free variables in the body therefore resolve against those
  // lifted globals. The result is rooted in a Persistent before this scope ends,
  // so no Local dangles.
  GreenGCPause gc(g_green_isolate);
  v8::Isolate::Scope isolate_scope(g_green_isolate);
  HandleScope scope(g_green_isolate);
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG") && (value->IsTypedArray() || value->IsArrayBuffer())) {
    static int g_cap;
    if (g_cap++ < 20) {
      GTDebugCtx d = GTGreenThreadDebugCtx(src);
      ::fprintf(stderr, "[GTGT-CAP] #%d %s addr=%p src=%p is_green=%d\n",
                g_cap, value->IsTypedArray() ? "TypedArray" : "ArrayBuffer",
                (void*)addr, (void*)src, (int)d.is_green);
      ::fflush(stderr);
    }
  }
#endif
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    static int g_rb;
    if (g_rb++ < 40) {
      GTDebugCtx d = GTGreenThreadDebugCtx(dst);
      bool gempty = ctx->Global().IsEmpty();
      ::fprintf(stderr, "[GTGT-REBUILD] #%d rebuilder_iso=%p is_green=%d tblock=%p "
                        "global_empty=%d\n",
                g_rb, (void*)dst, (int)d.is_green, (void*)d.tblock,
                (int)gempty);
      ::fflush(stderr);
    }
  }
#endif
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
    // Enter the source context as current on the source isolate before
    // GreenCapture walks the value. GreenCapture reads string-valued and
    // property-key members via `String::Utf8Value`, which resolves the
    // isolate's *current* context (it takes no context argument); with no
    // context entered on the green isolate that read resolves a null context
    // whose map word is 0xffffffffffffffff and the value->ToString() below
    // SEGVs. This mirrors Phase 2 (rebuild), which already enters `dst_ctx`
    // for the same reason. For a non-green source `ctx` is already current, so
    // the nested Context::Scope is a harmless no-op there.
    v8::Context::Scope ctx_scope(ctx);
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

// Prime ONE body's maglev -> optimized OSR on the green isolate WITHOUT going
// through the scheduler entry point (GreenJSEntry) or its iteration_count_ state
// machine. GreenJSEntry drives the recursive Model-A entry and bumps the pool
// gt's non-atomic iteration_count_, which corrupted dispatch. Direct fn->Call
// primes compilation for the shared SFI without touching any pool gt.
static void PrimeBody(GreenCallData* data) {
  if (data == nullptr) return;
  v8::Isolate::Scope isolate_scope(data->isolate);
  HandleScope scope(data->isolate);
  Local<Context> ctx = Local<Context>::New(data->isolate, data->context_p);
  Context::Scope context_scope(ctx);
  Local<Function> fn = Local<Function>::New(data->fn_isolate, data->fn_p);
  Local<Value> recv = ctx->Global();
  std::vector<Local<Value>> argv;
  argv.reserve(data->args_p.size());
  for (auto& arg : data->args_p)
    argv.push_back(Local<Value>::New(data->isolate, *arg));
  v8::TryCatch tc(data->isolate);
  const bool was_in_body = g_in_green_body;
  g_in_green_body = true;
  fn->Call(ctx, recv, argv.size(), argv.data());
  g_in_green_body = was_in_body;
}

// ====================================================================
// Async green bodies (await): run on the embedding isolate, Promise bridge
// --------------------------------------------------------------------
// A body that is an async function (async fn / async arrow / async generator /
// async concise method) is detected by its SharedFunctionInfo FunctionKind
// (GTIsAsyncFunction, in green_threads_gc.cc) and runs on the EMBEDDING isolate,
// not the fresh green realm. Node's async I/O bindings (setTimeout /
// setImmediate / process / require) live on the embedding realm; the green
// realm's lifted setTimeout calls the Node-internal validateFunction, which is
// not a global there, so async I/O on green ReferenceErrors. Sync bodies keep
// running green so genuine compute still parallelizes on the green isolate.
//
// On the embedding isolate the async body returns a pending Promise from
// fn->Call. We build the caller-facing Promise+Resolver here and hang
// fulfill/reject reactions on the body promise via Promise::Then; the caller
// Promise is set as the task's result row. When Node's event loop settles the
// body (async I/O completes) the reactions fire ON the embedding isolate and
// resolve/reject the caller resolver -- no green parking, drain, or resumer, so
// nothing performs V8 handle work off the embedding thread. The reactions are
// driven by the normal Node event loop (the JS caller awaits the returned
// Promise(s)), so no extra scheduling handle is required.
static void GreenAsyncResolveFulfilled(const FunctionCallbackInfo<Value>& args) {
  GreenCallData* data = static_cast<GreenCallData*>(
      v8::External::Cast(*args.Data())->Value());
  v8::Local<v8::Context> rctx =
      v8::Local<v8::Context>::New(data->isolate, data->caller_ctx_p);
  v8::Local<v8::Promise::Resolver> resolver =
      v8::Local<v8::Promise::Resolver>::New(data->isolate, data->caller_resolver_p);
  resolver->Resolve(rctx, args[0]);
  // The body promise has settled and disowned this reaction, so nothing else
  // references data now: release the rooted handles and free the call data.
  data->caller_resolver_p.Reset();
  data->caller_ctx_p.Reset();
  delete data;
}
static void GreenAsyncResolveRejected(const FunctionCallbackInfo<Value>& args) {
  GreenCallData* data = static_cast<GreenCallData*>(
      v8::External::Cast(*args.Data())->Value());
  v8::Local<v8::Context> rctx =
      v8::Local<v8::Context>::New(data->isolate, data->caller_ctx_p);
  v8::Local<v8::Promise::Resolver> resolver =
      v8::Local<v8::Promise::Resolver>::New(data->isolate, data->caller_resolver_p);
  resolver->Reject(rctx, args[0]);
  data->caller_resolver_p.Reset();
  data->caller_ctx_p.Reset();
  delete data;
}

// ====================================================================
// Green async I/O bridge (Option C: async body runs GREEN)
// --------------------------------------------------------------------
// A green async body runs on the GREEN isolate, so its control flow and CPU
// stay green. To await real async I/O -- which only exists on the embedding
// isolate's libuv -- it awaits the green global gtRead(path, encoding). The
// call returns a green promise immediately (the body parks) while the bytes are
// read on the libuv thread pool; on completion the result is resolved back onto
// that green promise, resuming the parked body via a green microtask.
//
// The dedicated green I/O loop is separate from Node's main libuv loop so
// pumping it inside g.spawnAll() never perturbs Node's event-loop state. The
// drive loop in RunTasks alternates pumping it with the green microtask
// checkpoint, so while one green body is parked on gtRead the others keep
// running -- Go-goroutine-style cooperative parking.
// ====================================================================

// Dedicated libuv loop for the green I/O bridge. Created lazily on first use
// (always the drive thread in this path), so no lock is needed.
uv_loop_t* GreenIOLoop() {
  static uv_loop_t* loop = nullptr;
  if (loop == nullptr) {
    loop = new uv_loop_t;
    uv_loop_init(loop);
  }
  return loop;
}

// Outstanding green I/O requests, so the drive loop knows when external work is
// pending (block on it) versus idle (drain non-I/O awaits without blocking).
static std::atomic<int> g_green_io_active{0};

// Per-outstanding-read work passed to the libuv thread pool: the path, the
// green resolver to resolve on completion, and the bytes read. uv_work_t has no
// user-data field, but it is the first (and only) member of this heap struct,
// so execute/complete reinterpret_cast the req back here.
struct GreenIoWork {
  uv_work_t req;
  std::string path;
  std::string encoding;
  Persistent<Promise::Resolver> green_resolver_p;  // resolved by the owning worker
  // Owner the moment this read finishes: the body's GreenThread* + the shared
  // async-drive scheduler it reactivates. GreenIoExecute (a libuv pool thread)
  // calls owner_sched->ReactivateGreenThread(owner_gt) here -- not in the
  // owning worker -- so the worker wakes the instant I/O completes instead of
  // polling. Null for the non-owned fallback path (never used for a real body).
  vi::GreenThread* owner_gt = nullptr;
  vi::GreenScheduler* owner_sched = nullptr;
  // Ownership + completion flags. `owned` is set when GreenGtRead hands the read
  // to the body's owning worker (data->pending_reads); a non-owned read is
  // resolved + freed here on GreenIoComplete (a fallback, unreachable for a real
  // async body). `finished` is set (release) by GreenIoExecute once buf/err are
  // valid, so the owning worker that reads it (acquire) observes a stable result.
  // `consumed` is set (release) by the owning driver AFTER it has resolved the
  // promise and copied buf into its own storage -- it gates GreenIoComplete, the
  // race-free free point (the libuv after-callback, which runs strictly after
  // the pool worker's post-execute bookkeeping). Until consumed, a completion
  // pumped before the driver drained it must not free w (the driver still needs
  // w to resolve).
  bool owned = false;
  std::atomic<bool> finished{false};
  std::atomic<bool> consumed{false};
  char* buf = nullptr;
  size_t len = 0;
  int err = 0;
};

static void GreenIoExecute(uv_work_t* req) {
  GreenIoWork* w = reinterpret_cast<GreenIoWork*>(req);
  // Wake the owning body's drive worker now that this read has a terminal result
  // (success OR failure). Must run after finished.store(true) above (or just
  // before, below) so the worker's acquire observes a stable buf/err. Guarded:
  // the non-owned fallback path never binds an owner.
  auto reactivate = [](GreenIoWork* w) {
    if (w->owner_gt != nullptr && w->owner_sched != nullptr)
      w->owner_sched->ReactivateGreenThread(w->owner_gt);
  };
  int fd = ::open(w->path.c_str(), O_RDONLY);
  if (fd < 0) { w->err = errno; w->len = 0; w->buf = nullptr; w->finished.store(true, std::memory_order_release); reactivate(w); g_green_io_active.fetch_sub(1, std::memory_order_relaxed); return; }
  long sz = 0;
  struct stat st;
  if (::fstat(fd, &st) == 0) sz = st.st_size;
  char* buf = static_cast<char*>(::malloc(sz > 0 ? sz : 1));
  if (buf == nullptr) { ::close(fd); w->err = ENOMEM; w->len = 0; w->buf = nullptr; w->finished.store(true, std::memory_order_release); reactivate(w); g_green_io_active.fetch_sub(1, std::memory_order_relaxed); return; }
  size_t off = 0;
  ssize_t r;
  while (off < static_cast<size_t>(sz) &&
         (r = ::read(fd, buf + off, static_cast<size_t>(sz) - off)) > 0)
    off += static_cast<size_t>(r);
  ::close(fd);
  w->buf = buf;
  w->len = off;
  w->err = 0;
  w->finished.store(true, std::memory_order_release);
  reactivate(w);
  g_green_io_active.fetch_sub(1, std::memory_order_relaxed);
}

static void GreenIoComplete(uv_work_t* req, int status) {
  GreenIoWork* w = reinterpret_cast<GreenIoWork*>(req);
  if (w->owned) {
    // The owning driver resolves the promise (into its own microtask queue --
    // same thread that checkpoints it) and owns buf; it copies buf before it
    // sets consumed. Free here, in the libuv after-callback, which runs only
    // AFTER the pool worker finished all post-execute bookkeeping on this req
    // (moving w->wq into the loop, uv_async_send). Freeing w from the driver
    // raced with that pool reference (SIGSEGV worker(arg=0x0) on the pool).
    // Defer the free until consumed so a completion pumped before its driver
    // drained it does not free a read the driver still has to resolve.
    //
    // P1 (single-loop pump): freeing runs on the dedicated libuv pump thread,
    // concurrently with the drive workers' green JS. buf is malloc'd and w is a
    // plain heap struct -- neither touches the green heap, so the green-isolate
    // Persistent reset moved to the driver (it Resolves, then Resets, here). Only
    // race-free free/delete remain, so this is safe off the drive workers' lock.
    if (!w->consumed.load(std::memory_order_acquire)) return;
    ::free(w->buf);
    delete w;
    return;
  }
  // Non-owned fallback (unreachable for a real green-async body -- the setup and
  // drive paths always set g_green_driving_body before any gtRead resolves).
  // Resolve inline and free here, matching the pre-fan-out serial behaviour.
  {
    v8::Isolate::Scope iso_scope(g_green_isolate);
    v8::HandleScope hs(g_green_isolate);
    v8::Local<v8::Context> ctx =
        v8::Local<v8::Context>::New(g_green_isolate, g_green_ctx);
    v8::Context::Scope ctx_scope(ctx);
    v8::Local<v8::Promise::Resolver> res =
        v8::Local<v8::Promise::Resolver>::New(g_green_isolate, w->green_resolver_p);
    if (status != 0 || w->err != 0) {
      int code = w->err != 0 ? w->err : status;
      std::string msg = std::string("green_threads gtRead: ") + std::string(strerror(code));
      res->Reject(ctx, v8::Exception::Error(
          v8::String::NewFromUtf8(g_green_isolate, msg.c_str(),
                                  v8::NewStringType::kNormal).ToLocalChecked()));
    } else {
      std::string contents(w->buf, w->len);
      res->Resolve(ctx, v8::String::NewFromUtf8(g_green_isolate, contents.c_str(),
                                                v8::NewStringType::kNormal).ToLocalChecked());
    }
  }
  ::free(w->buf);
  delete w;
}

// gtRead(path, encoding) -- the green global an async body awaits. Runs green;
// returns a green promise that resolves (or rejects) when the file read, done on
// the embedding libuv pool, completes.
static void GreenGtRead(const FunctionCallbackInfo<Value>& args) {
  v8::Isolate* green = args.GetIsolate();
  v8::HandleScope hs(green);
  v8::Local<v8::Context> ctx = green->GetCurrentContext();
  v8::Local<v8::Promise::Resolver> res;
  if (!v8::Promise::Resolver::New(ctx).ToLocal(&res)) return;

  GreenIoWork* w = new GreenIoWork();
  w->buf = nullptr;
  w->len = 0;
  w->err = 0;
  if (args[0]->IsString()) {
    v8::String::Utf8Value p(green, args[0]);
    w->path.assign(*p, p.length());
  }
  if (args[1]->IsString()) {
    v8::String::Utf8Value e(green, args[1]);
    w->encoding.assign(*e, e.length());
  }
  g_green_io_active.fetch_add(1, std::memory_order_relaxed);
  w->green_resolver_p.Reset(green, res);
  // Bind this read to its owning body + the shared async-drive scheduler so
  // GreenIoExecute (a libuv pool thread) can reactivate the body the instant
  // the read finishes -- the body does not run on this pool thread, it runs on
  // a drive worker that parks until the Reactivate wakes it. g_green_driving_body
  // is set by the setup thread (first read) and by each drive worker (later
  // continuations); g_async_drive_scheduler is set by RunTasks for the whole run.
  if (g_green_driving_body != nullptr) {
    w->owned = true;
    w->owner_gt = g_green_driving_body->gt;
    w->owner_sched = g_async_drive_scheduler;
  }
  uv_queue_work(GreenIOLoop(), &w->req, GreenIoExecute, GreenIoComplete);
  if (g_green_driving_body != nullptr) {
    // Hand the read to the body's OWN drive worker: it owns both the resolution
    // (into the body's own queue -- same thread that checkpoints that queue, so
    // the lock-free MicrotaskQueue never sees a concurrent enqueue here and a
    // drain elsewhere) and the freeing. w->owned stays false for the fallback
    // path, where GreenIoComplete resolves + frees it.
    g_green_driving_body->pending_reads.push_back(
        std::unique_ptr<GreenIoWork>(w));
  }
  args.GetReturnValue().Set(res->GetPromise());
}

// Install the green async I/O globals on the green context's global object so a
// relocated async body's gtRead(...) free variable resolves here.
static void InstallGreenAsyncGlobals(Local<Context> green_ctx) {
  v8::Isolate::Scope isolate_scope(g_green_isolate);
  HandleScope scope(g_green_isolate);
  Context::Scope ctx_scope(green_ctx);
  GreenGCPause gc(g_green_isolate);  // protect the freshly-installed global
  Local<Function> gt =
      FunctionTemplate::New(g_green_isolate, GreenGtRead)->GetFunction(green_ctx)
          .ToLocalChecked();
  Local<String> key = String::NewFromUtf8(
      g_green_isolate, "gtRead", v8::NewStringType::kNormal).ToLocalChecked();
  green_ctx->Global()->Set(green_ctx, key, gt).Check();
  // Green `require(name)`: load a Node module on the embedding realm and return
  // its exports as a green proxy, so a green async body can `require('node:*')`.
  {
    Local<Function> req =
        FunctionTemplate::New(g_green_isolate, GreenBridgeRequire)
            ->GetFunction(green_ctx)
            .ToLocalChecked();
    Local<String> req_key = String::NewFromUtf8(
        g_green_isolate, "require", v8::NewStringType::kNormal).ToLocalChecked();
    green_ctx->Global()->Set(green_ctx, req_key, req).Check();
  }
}

// The reaction chained onto the green-async body promise (installed as BOTH the
// fulfilled and the rejected handler). It fires ON the green isolate when the
// body settles, capturing args[0] -- the green-isolate value the body settled
// with, i.e. its return value if fulfilled or the rejection reason otherwise --
// into a green Persistent. It deliberately does NOT switch isolate scope or
// touch the embedding caller resolver: a body settle reaction must stay on the
// green isolate so the drive loop's green Isolate::Scope stays balanced and so
// the (green) value handle is read on its own isolate. RunTasks copies the
// captured value to the embedding isolate and resolves the caller resolver.
static void GreenGreenCaptureSettle(const FunctionCallbackInfo<Value>& args) {
  GreenCallData* data = static_cast<GreenCallData*>(
      v8::External::Cast(*args.Data())->Value());
  data->settle_value_p.Reset(data->isolate, args[0]);
}

// Build the data + caller Promise for a green-async body, run it green to its
// first await (fn->Call returns the pending body promise), and chain the green
// reactions that resolve the embedding caller Promise on settlement.
static GreenCallData* SetupAndRunGreenAsyncBody(Isolate* emb, Local<Context> emb_ctx,
                                                Isolate* green, Local<Function> fn,
                                                Local<Array> task) {
  GreenCallData* data = new GreenCallData();
  data->awaiting = true;
  data->async_green = true;
  data->caller_isolate = emb;
  data->isolate = green;
  data->fn_isolate = green;

  // A GreenThread* descriptor for cooperative I/O block/resume. It is NOT run by
  // the scheduler (async bodies are driven directly here) and holds no arch stack
  // or COW view -- it only carries the kBlocked state + reactivation flag that
  // the async-drive parks against. Created here, before the first gtRead below,
  // so the read (and thus this body's completion Reactivate) can target it.
  data->gt = new vi::GreenThread(reinterpret_cast<vi::Isolate*>(green));

  // Create THIS body's own green context + microtask queue. Each green-async
  // body gets a private context (not the shared g_green_ctx) so a per-worker
  // drive can resume one body's continuations from its own queue with no race
  // on a shared context/queue. The microtask queue is owned by data->body_mq_p
  // and must outlive the context (struct-destruction order disposes the
  // context first, then this). gtRead is installed so the body's `await
  // gtRead(...)` free variable resolves; embedding globals like setTimeout are
  // intentionally NOT lifted -- a green-async body bridges I/O through gtRead.
  std::unique_ptr<Persistent<Function>> relocated;
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_DEBUG_ENTCTX")) {
    ::fprintf(stderr, "[ANDAB-START] body=%p iso=%p tid=0x%lx round=%llu\n",
              (void*)data, (void*)green, (unsigned long)pthread_self(),
              (unsigned long long)g_green_spawn_round.load(std::memory_order_relaxed));
    ::fflush(stderr);
  }
#endif
  {
    v8::Isolate::Scope iso_scope(green);
    HandleScope hs(green);
    Local<Context> body_ctx = Context::New(green);
    data->body_mq_p = v8::MicrotaskQueue::New(green);
    body_ctx->SetMicrotaskQueue(data->body_mq_p.get());
    InstallGreenAsyncGlobals(body_ctx);
    data->context_p.Reset(green, body_ctx);
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_DEBUG_ENTCTX")) {
      static std::unordered_map<GreenCallData*, void*>& g_body_ctx_addr =
          *new std::unordered_map<GreenCallData*, void*>();
      const void* ctx_addr =
          reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(*body_ctx));
      ::fprintf(stderr,
                "[CTX-SETUP] body=%p round=%llu ctx=%p empty=%d\n",
                (void*)data,
                (unsigned long long)data->round, (void*)ctx_addr,
                (int)data->context_p.IsEmpty());
      ::fflush(stderr);
      g_body_ctx_addr[data] = const_cast<void*>(ctx_addr);
    }
#endif
    // Relocate the function onto THIS body's context and lift its captured free
    // variables there, so the body runs green with its own lift + gtRead.
    relocated = RelocateFunction(emb, fn, body_ctx);
  }
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_DEBUG_ENTCTX")) {
    ::fprintf(stderr, "[ANDAB] body=%p after-block1 iso=%p tid=0x%lx\n",
              (void*)data, (void*)green, (unsigned long)pthread_self());
    ::fflush(stderr);
  }
#endif

  // Caller-facing Promise (embedding): handed to the JS caller, resolved once
  // the green body settles.
  {
    v8::Isolate::Scope iso_scope(emb);
    HandleScope hs(emb);
    Local<Promise::Resolver> resolver =
        Promise::Resolver::New(emb_ctx).ToLocalChecked();
    data->caller_resolver_p.Reset(emb, resolver);
    data->caller_ctx_p.Reset(emb, emb_ctx);
  }

  data->fn_p.Reset(green, *relocated);
  GreenValueCopier args_copier;
  args_copier.src = emb;
  args_copier.dst = green;
  const uint32_t argc = task->Length();
  for (uint32_t a = 1; a < argc; a++) {
    Local<Value> arg = task->Get(emb_ctx, a).ToLocalChecked();
    data->args_p.push_back(CopyValueAcross(args_copier, arg));
  }

  // Run the body green up to its first await (or to completion if it has none).
  // An async function always returns a promise, so out is a promise here.
  {
    v8::Isolate::Scope iso_scope(green);
    GreenGCPause gc(green);
    HandleScope hs(green);
    Local<Context> gctx = Local<Context>::New(green, data->context_p);
    Context::Scope ctx_scope(gctx);
    Local<Function> gfn = Local<Function>::New(green, data->fn_p);
    // Install this body as the driving body so its first gtRead (phase 1)
    // registers into data->pending_reads for its owning drive worker to resolve
    // later. Restored below the body's first await has been captured.
    GreenCallData* const prior_driving = g_green_driving_body;
    g_green_driving_body = data;
    std::vector<Local<Value>> argv;
    for (auto& arg : data->args_p) argv.push_back(Local<Value>::New(green, *arg));
    Local<Value> out;
    if (!gfn->Call(gctx, gctx->Global(), argv.size(), argv.data()).ToLocal(&out) ||
        !out->IsPromise()) {
      // Not actually async: reject the caller with a clear error.
      v8::Isolate::Scope iso_scope2(emb);
      HandleScope hs2(emb);
      Local<Promise::Resolver> resolver =
          Local<Promise::Resolver>::New(emb, data->caller_resolver_p);
      Local<Context> rctx = Local<Context>::New(emb, data->caller_ctx_p);
      Local<Value> ty = v8::Exception::TypeError(String::NewFromUtf8(
          emb, "green_threads body must be an async function",
          v8::NewStringType::kNormal).ToLocalChecked());
      resolver->Reject(rctx, ty);
      return data;
    }
    g_green_driving_body = prior_driving;
    data->async_promise_p.Reset(green, Local<Promise>::Cast(out));
    // Chain the green reaction that captures the body's settle value. The same
    // function is installed as both the fulfilled and rejected handler: in
    // either case args[0] is the value the body settled with and RunTasks must
    // copy it into the embedding caller resolver.
    Local<Value> ext = v8::External::New(green, data);
    Local<Function> settle =
        Function::New(gctx, GreenGreenCaptureSettle, ext).ToLocalChecked();
    (void)Local<Promise>::Cast(out)
        ->Then(gctx, settle, settle)
        .ToLocalChecked();
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_DEBUG_ENTCTX")) {
      ::fprintf(stderr, "[ANDAB] body=%p after-block2 iso=%p tid=0x%lx\n",
                (void*)data, (void*)green, (unsigned long)pthread_self());
      ::fflush(stderr);
    }
#endif
  }
  return data;
}

// Drive every queued green-async body to settlement across an OS worker pool.
//
// Each body owns a private green context + MicrotaskQueue (installed in
// SetupAndRunGreenAsyncBody). A body is resumed by resolving its outstanding
// gtRead promises -- which enqueue that body's awaited continuation into the
// body's OWN queue -- then checkpointing that same queue. Because the V8
// MicrotaskQueue is lock-free, these two steps MUST happen on the SAME thread,
// so each body is owned by exactly one drive worker that both resolves its
// finished reads and checkpoints its queue; distinct workers drive disjoint
// bodies (round-robin partition), so no per-body queue is ever touched by two
// threads at once.
//
// An async worker runs on its own OS thread with a fresh PerIsolateThreadData
// and never sets CurrentGreenThread(), so CowEnabledThread() returns null and
// the CoW write barrier is a no-op during the drive -- async bodies allocate
// only fresh objects, kept disjoint by the CAS bump pointer while RunTasks holds
// the AlwaysAllocate window (now covering all-async). N workers each holding
// v8::Isolate::Scope + HandleScope on the green isolate reuses the proven sync
// true-parallel recipe; per-worker handle-scope routing (SetIsolateHandleScope
// PerWorker) is armed around the whole window so handle-scope / entered-context
// state is per-OS-thread.
//
// Returns false if the stall limit was hit (a body waits on I/O that never
// completes).
//
// g_green_mutator_lock: coarse green-mutator lock (M2a baseline). Serializes the
// concurrent green workers so at most ONE OS worker touches the green heap's
// shared execution-state (the isolate-independent JS-entry microtask runner
// re-arms x25 from the shared execution_state_base_ mirror; without this guard a
// sibling worker's refresh clobbers it mid-arm -> cross-worker context desync
// -> SIGSEGV). True all-mutator parallelism is the later milestone; this is the
// "mutex-protected" phase N workers run behind. Defined here (near its async
// driver) though referenced below by the M2Probe pool too.
static std::mutex g_green_mutator_lock;

static bool DriveGreenAsyncBodiesParallel(Isolate* green,
                                          std::vector<GreenCallData*>& datas) {
  // Halt incremental marking for the whole drive: parked async bodies keep their
  // JS stack inside their async closure (rooted by data->async_promise_p), every
  // gtRead resolver is a green Persistent, and the settle reaction captures a
  // green value -- a collection could otherwise relocate/reclaim any of these.
  // The AlwaysAllocate window armed by RunTasks (now covering all-async too)
  // already holds marking stopped here; GreenGCPause is the belt-and-suspenders
  // guard for the all-async case where no window was armed in the mixed path.
  GreenGCPause gc(green);

  // Async bodies (async_green) are driven directly here (never through the
  // scheduler's RunGreenThread). Each holds a bare GreenThread* descriptor
  // (created in SetupAndRunGreenAsyncBody) that the cooperative block/resume
  // uses to free the OS core while the body is parked on I/O -- the descriptor
  // is NOT a runnable green thread; its continuations run in this drive loop,
  // blocked via GreenScheduler::BlockGreenThread and woken via
  // ReactivateGreenThread. Sync bodies (awaiting) are driven by the pool and
  // have no reads/queue to drive.
  std::vector<GreenCallData*> async_bodies;
  async_bodies.reserve(datas.size());
  for (GreenCallData* d : datas)
    if (d->async_green) async_bodies.push_back(d);
  if (async_bodies.empty()) return true;

  int num_workers = static_cast<int>(std::thread::hardware_concurrency());
  if (num_workers < 1) num_workers = 1;
  if (const char* wenv = ::getenv("GT_NUM_WORKERS")) {
    int requested = atoi(wenv);
    if (requested > 0 && requested < num_workers) num_workers = requested;
  }
  if (num_workers > static_cast<int>(async_bodies.size()))
    num_workers = static_cast<int>(async_bodies.size());

  // Round-robin partition: body i is owned by worker (i % num_workers).
  std::vector<std::vector<GreenCallData*>> partitions(num_workers);
  for (size_t i = 0; i < async_bodies.size(); i++)
    partitions[i % num_workers].push_back(async_bodies[i]);

  const int max_idle = 2000000;
  std::vector<std::thread> workers;
  workers.reserve(num_workers);

  auto run_partition = [&](std::vector<GreenCallData*> bodies) {
    // Activate the green isolate on THIS OS worker + per-worker handle scope.
    // The per-thread CurrentGreenThread() slot left null here (no gt for async),
    // which is what makes the CoW barrier a no-op for async-allocated objects.
    v8::Isolate::Scope iso_scope(green);
    v8::HandleScope handle_scope(green);
    // Shared async-drive scheduler (created in RunTasks, alive for this run).
    // Every drive worker blocks against and is reactivated through this ONE
    // coordinator, so a read submitted during setup -- before any worker existed
    // -- still has a valid scheduler to Reactivate against on completion.
    vi::GreenScheduler* sched = g_async_drive_scheduler;
    int idle = 0;
    for (;;) {
      // Consume Reactivate signals that arrived while we were parked or working,
      // BEFORE re-testing the predicate and re-parking -- otherwise a leftover
      // entry would keep this worker awake. ReactivateGreenThread sets the body's
      // react_pending flag before pushing to the shared react_queue_, so draining
      // here resets that flag (Reactivate skips an already-pending body) and keeps
      // the queue from growing unbounded: without it a body's SECOND read could
      // never be reactivated. This is the standard condition-variable contract --
      // drain the wakeners, then re-check the predicate. Drain is a no-op when the
      // queue is empty (the common per-iteration case) and is race-safe: it only
      // clears react_pending / unlinks blocked bodies; THIS worker's own readiness
      // is decided by the finished-read predicate below, not this queue.
      if (sched != nullptr) sched->DrainReactivations();
      bool all_settled = true;
      bool progressed = false;
      for (GreenCallData* d : bodies) {
        if (d->async_settled) continue;
        all_settled = false;

        // Serialize this worker's microtask execution against the other workers
        // sharing this green isolate (lock held for the whole per-body
        // iteration). PerformCheckpoint runs the bodies' promise microtasks,
        // which touch the ISOLATE-GLOBAL GlobalHandles table (MakeWeak/Release
        // of the green result roots) and shared promise/task bookkeeping that is
        // NOT worker-safe: without this guard N workers drive their bodies'
        // reactions concurrently and corrupt global handle nodes -> "Check
        // failed: node->IsInUse()" FATAL (reproduced ~100% under GT_NUM_WORKERS>=8
        // in the parallel gate). js-entry x25 arming is per-worker (Stage 1) and
        // the mirror write is gated (Stage 2), but microtask-execution shared
        // state still requires per-iteration serialization, so the async drive is
        // correctly SERIAL here; true-parallel I/O drive needs the GlobalHandles/
        // microtask path made worker-safe (a separate co-design).
        std::lock_guard<std::mutex> mutator_lock(g_green_mutator_lock);

        // Take ownership of this thread's driving slot for the whole iteration.
        // A finished read's resolve may RESUME the body's continuation inline
        // (its awaited reaction runs during res->Resolve); that continuation's
        // next gtRead must then route into THIS body's pending_reads and Reactivate
        // THIS body's gt. g_green_driving_body is per-OS-thread thread_local, and
        // the value the setup thread stashed is invisible on this worker, so pin it
        // to d here and restore it after PerformCheckpoint below.
        GreenCallData* const prior_driving = g_green_driving_body;
        g_green_driving_body = d;

        // Drain FINISHED gtReads OUT of pending_reads BEFORE resolving them.
        // Resolving a finished read may inline-resume the body, which then pushes
        // its next read into pending_reads -- mutating the vector we would be
        // iterating, so empty the finished ones first. pending_reads then holds
        // only unfinished reads (possibly empty). `finished` holds raw pointers:
        // GreenIoComplete (the libuv after-callback) owns freeing them, so the
        // vector must not delete -- a double free would otherwise follow.
        std::vector<std::unique_ptr<GreenIoWork>> still;
        std::vector<GreenIoWork*> finished;
        still.reserve(d->pending_reads.size());
        for (auto& rd : d->pending_reads) {
          if (rd->finished.load(std::memory_order_acquire))
            finished.push_back(rd.release());
          else
            still.push_back(std::move(rd));
        }
        d->pending_reads.swap(still);

        // Resolve the finished reads. Resolve runs with the body's own context
        // current so the awaited reaction enqueues into THIS body's queue. A
        // synchronous resume here pushes the next read into the just-drained
        // pending_reads; we iterate the local `finished` vector, not pending_reads,
        // so that re-entry is safe.
        for (auto& rd : finished) {
          v8::Local<v8::Context> bctx =
              v8::Local<v8::Context>::New(green, d->context_p);
          {
            v8::Context::Scope ctx_scope(bctx);
            v8::Local<v8::Promise::Resolver> res =
                v8::Local<v8::Promise::Resolver>::New(green, rd->green_resolver_p);
            if (rd->err != 0) {
              std::string msg =
                  std::string("green_threads gtRead: ") +
                  std::string(strerror(rd->err));
              res->Reject(bctx, v8::Exception::Error(
                  v8::String::NewFromUtf8(green, msg.c_str(),
                                          v8::NewStringType::kNormal)
                                              .ToLocalChecked()));
            } else {
              std::string contents(rd->buf, rd->len);
              res->Resolve(bctx, v8::String::NewFromUtf8(
                  green, contents.c_str(), v8::NewStringType::kNormal)
                                      .ToLocalChecked());
            }
          }
          rd->consumed.store(true, std::memory_order_release);
          // P1: the driver owns the green resolver Persistent (GreenIoComplete no
          // longer touches green-isolate state), so release it here after resolving.
          rd->green_resolver_p.Reset();
          progressed = true;
        }

        // Resolve bridged emb-Promise settles: GreenEmbPromiseReaction (running
        // on the embedding realm) set emb_settled and Reactivated this worker.
        // Drain settled entries and resolve the body's awaited GP with the emb
        // settle value (marshaled emb->green), mirroring the gtRead
        // resolve-then-checkpoint pattern so the body's await continuation runs
        // at the PerformCheckpoint below.
        {
          std::vector<std::unique_ptr<GreenEmbReact>> kept;
          std::vector<GreenEmbReact*> done;
          for (auto& re : d->pending_emb)
            if (re->emb_settled.load(std::memory_order_acquire))
              done.push_back(re.release());
            else
              kept.push_back(std::move(re));
          d->pending_emb.swap(kept);
          if (!done.empty()) {
            Isolate* emb = d->caller_isolate;
            GreenValueCopier copier;
            copier.src = emb;
            copier.dst = green;
            for (auto re : done) {
              Local<Promise::Resolver> res =
                  Local<Promise::Resolver>::New(green, re->gp_res_p);
              Local<Value> emb_val = Local<Value>::New(emb, re->settle_val_p);
              auto p = CopyValueAcross(copier, emb_val);
              {
                Local<Context> rctx =
                    Local<Context>::New(green, d->context_p);
                Context::Scope cs(rctx);
                Local<Value> gval = Local<Value>::New(green, *p);
                if (re->is_error)
                  res->Reject(rctx, gval);
                else
                  res->Resolve(rctx, gval);
              }
              progressed = true;
              delete re;
            }
          }
        }


        // Resume continuations now enqueued in THIS body's own queue and run
        // them under the SAME lock. The microtask execution (PerformCheckpoint)
        // below shares isolate-global handle/promise state (GlobalHandles,
        // promise reaction jobs) with the other workers' PerformCheckpoint, so it
        // stays serialized (see the top comment). The per-entry mirror-resync
        // scaffolding was removed earlier because it was obsolete once arming
        // became per-worker (Stage 1) -- but that scaffolding was only ever
        // exercised under this same lock, so removing it does not change the
        // serialization.
        v8::Local<v8::Context> bctx2 =
            v8::Local<v8::Context>::New(green, d->context_p);
        bctx2->GetMicrotaskQueue()->PerformCheckpoint(green);
        // The just-enqueued reads are picked up by the libuv thread pool; their
        // completion callbacks (GreenIoComplete) drain on the dedicated pump
        // thread (P1), not here, so this worker never calls uv_run. The reads
        // already have finished+consumed set by the driver's resolve loop above, so
        // the pump's free is race-free. g_green_driving_body stays pinned to this
        // body's descriptor while its continuation runs (see the top comment).
        g_green_driving_body = prior_driving;

        // Mark settled once the body promise leaves kPending; its reaction then
        // captured the settle value, and RunTasks resolves the caller Promise.
        if (!d->async_settled && !d->async_promise_p.IsEmpty()) {
          v8::Local<v8::Promise> p =
              v8::Local<v8::Promise>::New(green, d->async_promise_p);
          if (p->State() != v8::Promise::PromiseState::kPending)
            d->async_settled = true;
        }
        if (d->async_settled) progressed = true;
      }
      if (all_settled) break;
      if (++idle > max_idle) return false;
      // No read was resolvable and nothing advanced this pass: every remaining
      // body is parked on an in-flight I/O. Cooperatively block each body
      // (stamping kBlocked so the OS core is freed) and park THIS worker until a
      // libuv pool thread reactivates one of them -- replacing the old
      // sleep_for(50us)/max_idle busy-poll that burned a core per parked body.
      // Reactivate fires from GreenIoExecute (after finished.store(true)), so the
      // worker wakes the instant an I/O finishes rather than every 50us.
      if (!progressed) {
        bool any_starved = false;
        for (GreenCallData* d : bodies) {
          if (d->async_settled) continue;
          // A body with no outstanding read cannot be reactivated (nothing will
          // call Reactivate for it); block only bodies parked on real I/O, so a
          // genuinely starved body keeps the idle counter spinning to the safety
          // bound instead of parking on a CV that never fires.
          if (d->pending_reads.empty() && d->pending_emb.empty()) { any_starved = true; continue; }
          if (d->gt != nullptr && sched != nullptr)
            sched->BlockGreenThread(d->gt);
        }
        // Park only when every parked body has an outstanding read whose
        // completion will Reactivate it -- otherwise never block and let the
        // idle counter bound the run. Park on a predicate that checks THIS
        // worker's OWN finished reads, not the shared react_queue_: under the
        // single drive scheduler another worker may DrainReactivations and clear
        // this worker's reactivation, stranding it in the predicate wait forever
        // while its own completed read never drains. A read's finished flag is set
        // (before Reactivate) and stays visible to its owner, so this predicate is
        // steal-proof and still frees the core.
        auto park_pred = [&] {
          for (GreenCallData* d : bodies) {
            if (d->async_settled) continue;
            for (auto& rd : d->pending_reads)
              if (rd->finished.load(std::memory_order_acquire)) return true;
            // A bridged emb-Promise may have settled (its reaction set
            // emb_settled + Reactivated); include it so this worker
            // self-wakes even if the Reactivate notify was drained elsewhere.
            for (auto& re : d->pending_emb)
              if (re->emb_settled.load(std::memory_order_acquire)) return true;
          }
          return false;
        };
        if (!any_starved && sched != nullptr)
          sched->WaitForReactivationWithPred(park_pred);
      }
    }
  };

  // P1: dedicated libuv pump thread. The single shared GreenIOLoop must be pumped
  // by exactly one caller (libuv rule), but that caller must NOT be each drive
  // worker serialized on g_green_mutator_lock (that was the async ceiling). One
  // pump thread owns uv_run(GreenIOLoop()) for the whole drive and drains every
  // completion callback -- GreenIoComplete now only frees malloc'd buf + the plain
  // GreenIoWork (no green-isolate dynamic state), so it is safe on the pump
  // concurrently with the workers' green JS. Workers keep uv_queue_work (thread-
  // safe) to submit reads, then park until Reactivate; the pump frees the completed
  // reads. UV_RUN_DEFAULT makes it block idle instead of spinning.
  std::atomic<bool> pump_stop{false};
  std::thread pump;
  {
    uv_loop_t* pump_loop = GreenIOLoop();
    pump = std::thread([pump_loop, &pump_stop]() {
      while (!pump_stop.load(std::memory_order_acquire))
        uv_run(pump_loop, UV_RUN_DEFAULT);
    });
  }

  for (int i = 0; i < num_workers; i++)
    workers.emplace_back(run_partition, std::move(partitions[i]));
  for (std::thread& w : workers) w.join();

  // Release the pump cleanly: its final UV_RUN_DEFAULT drains + frees every
  // completion callback already queued (all reads are consumed once the workers
  // joined), then it sees pump_stop and exits. Only ever one uv_run at a time --
  // no main-thread pump, no uv_stop (which would discard a pending free instead of
  // running it).
  pump_stop.store(true, std::memory_order_release);
  if (pump.joinable()) pump.join();

  return true;
}

// Build the GreenCallData for an async body and run it on the embedding isolate.
// The caller-facing Promise is recovered from data->caller_resolver_p at result
// time. An async function never throws or returns a non-promise, so a failed or
// non-promise Call here means a non-async Function slipped past GTIsAsyncFunction;
// we bridge a rejected caller Promise with a clear TypeError instead of leaving
// the caller with a hanging promise.
static GreenCallData* SetupAndRunAsyncBody(Isolate* emb, Local<Context> ctx,
                                           Local<Function> fn, Local<Array> task) {
  GreenCallData* data = new GreenCallData();
  data->awaiting = true;
  data->isolate = emb;
  data->fn_isolate = emb;
  data->context_p.Reset(emb, ctx);
  data->fn_p.Reset(emb, fn);
  const uint32_t argc = task->Length();
  for (uint32_t a = 1; a < argc; a++) {
    data->args_p.push_back(
        std::make_unique<Persistent<Value>>(emb, task->Get(ctx, a).ToLocalChecked()));
  }
  Local<Function> body_fn = Local<Function>::New(emb, data->fn_p);
  std::vector<Local<Value>> argv;
  argv.reserve(data->args_p.size());
  for (auto& arg : data->args_p) argv.push_back(Local<Value>::New(emb, *arg));
  Local<Value> body_out;
  if (!body_fn->Call(ctx, ctx->Global(), argv.size(), argv.data()).ToLocal(&body_out) ||
      !body_out->IsPromise()) {
    Local<Promise::Resolver> resolver = Promise::Resolver::New(ctx).ToLocalChecked();
    data->caller_resolver_p.Reset(emb, resolver);
    data->caller_ctx_p.Reset(emb, ctx);
    Local<Value> ty = v8::Exception::TypeError(
        String::NewFromUtf8(emb,
            "green_threads body must be an async function",
            v8::NewStringType::kNormal).ToLocalChecked());
    Local<Context> rctx = Local<Context>::New(emb, data->caller_ctx_p);
    Local<Promise::Resolver>::New(emb, data->caller_resolver_p)->Reject(rctx, ty);
    return data;
  }
  // Chain fulfill/reject reactions onto the body promise so its settled
  // value/reason resolves/rejects the caller resolver on the embedding isolate.
  Local<Promise::Resolver> resolver = Promise::Resolver::New(ctx).ToLocalChecked();
  data->caller_resolver_p.Reset(emb, resolver);
  data->caller_ctx_p.Reset(emb, ctx);
  Local<Value> ext = v8::External::New(emb, data);
  Local<Function> fulfill =
      Function::New(ctx, GreenAsyncResolveFulfilled, ext).ToLocalChecked();
  Local<Function> reject =
      Function::New(ctx, GreenAsyncResolveRejected, ext).ToLocalChecked();
  (void)Local<Promise>::Cast(body_out)
      ->Then(ctx, fulfill, reject)
      .ToLocalChecked();
  return data;
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_CRASH_DEBUG")) {
    ::fprintf(stderr,
              "[GTDBG-GREENJSENTRY] round=%u gt=%p pc=%p iter=%d tid=%ld\n",
              (unsigned)data->round, (void*)gt, (void*)gt->pc_raw(),
              (unsigned)gt->iteration_count(), (long)pthread_self());
    ::fflush(stderr);
  }
#endif

  // Multiple OS workers drive bodies concurrently inside ONE shared green heap.
  // Per-worker scope state (handle-scope phase) is what makes overlap correct:
  //   - the live handle cursor + handle blocks live in THIS worker's
  //     HandleScopeImplementer (per-OS-thread, allocated in
  //     FindOrAllocatePerThreadDataForThisThread under green_thread_mode_),
  //   - entered_contexts_ is a member of that same per-worker impl, and both
  //     Context::Enter and Context::Exit route through the per-worker
  //     handle_scope_implementer() accessor under the opt-in flag held true for
  //     the whole dispatch window (see RunTasks / M2Probe); set_context() sets
  //     the per-worker ThreadLocalTop::context_ via x25.
  // Because entered-context / cursor / set_context are per-worker, two workers
  // can now Enter/Exit their own contexts without the "Cannot exit
  // non-entered context" abort that hit the old shared per-isolate impl.
  // (Verified: the disjoint test runs 4 concurrent workers with the serializer
  // released and is clean -- the parallel form of Stage-2-1 root routing.)
  //
  // What STILL serializes the body (so GreenThreadCowSharedWriterScope guards
  // the whole body here):
  //   1. Isolate activation + JS-invoke permission: the v8::Isolate::Scope below
  //      activates THIS green isolate on every OS worker at once, and
  //      JS-invoke permission is an isolate-level flag -- concurrent
  //      activation/invoke trips "Invoke in DisallowJavascriptExecutionScope"
  //      (SIGABRT) and torn state (SIGSEGV). Isolate::Activate/Deactivate and
  //      the invoke-disallow flag are not yet per-worker.
  //   2. Shared compilation state: map tiering, feedback vectors, and the
  //      JSDispatchTable are process-global and mutated by concurrent bodies
  //      (e.g. Runtime_CompileOptimizedOSRFromMaglev OSR). Narrowing the writer
  //      scope to these ops instead of the whole body is the follow-up.
  // Releasing this scope is Stage 3, gated on (1) made per-worker and (2) the
  // narrow compilation-state guard kept.
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    ::fprintf(stderr,
              "[GTGT-AFTERSCOPE] pthread=0x%lx data_isolate=%p cur_iso=%p "
              "is_green=%d\n",
              (unsigned long)pthread_self(), (void*)data->isolate,
              (void*)GTGreenThreadCurrentIsolate(),
              (int)data->isolate->isGreenThreadMode());
    ::fflush(stderr);
  }
#endif

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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    static std::atomic<int> g_concurrent_bodies{0};
    const int prev = g_concurrent_bodies.fetch_add(1, std::memory_order_acq_rel);
    const int conc = prev + 1;
    GTDebugCtx d = GTGreenThreadDebugCtx(data->isolate);
    void* cur_iso = node::GTGreenThreadCurrentIsolate();
    ::fprintf(stderr,
              "[GTGT-ENTER] tid=0x%lx gt=%p iso=%p cur_iso=%p conc=%d "
              "is_green=%d same=%d\n"
              "   main_iso=%p main_heap=%p tblock=%p\n",
              (unsigned long)pthread_self(), (void*)gt, (void*)data->isolate,
              cur_iso, conc, (int)d.is_green,
              (int)(cur_iso == (void*)data->isolate), d.main_iso, d.main_heap,
              d.tblock);
    ::fflush(stderr);
    g_concurrent_bodies.fetch_sub(1, std::memory_order_acq_rel);
  }
#endif
  v8::TryCatch body_tc(data->isolate);
  const bool was_in_body = g_in_green_body;
  g_in_green_body = true;
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTExecSnap s = GTGreenThreadExecSnapshot();
    GTDebugCtx d = GTGreenThreadDebugCtx(data->isolate);
    ::fprintf(stderr,
              "[GTGT-PRECALL] tid=0x%lx iso=%p cur_iso=%p cur_green=%d "
              "cur_ptd=%p ptd_iso=%p ptd_green=%d tlt=%p central=%d "
              "data_green=%d\\n",
              (unsigned long)pthread_self(), (void*)data->isolate, s.cur_iso,
              s.cur_green, s.cur_ptd, s.ptd_iso, s.ptd_green, s.tlt, s.central,
              (int)d.is_green);
    ::fflush(stderr);
  }
#endif
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DUMP")) {
    node::GTGreenThreadEntryDump(data->isolate);
  }
#endif
  const bool called = fn->Call(ctx, recv, argv.size(), argv.data()).ToLocal(&out);
  g_in_green_body = was_in_body;
  if (!called) {
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
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_CRASH_DEBUG")) {
      ::fprintf(stderr,
                "[GTDBG-THREW] gt=%p tid=%ld\n", (void*)gt,
                (long)pthread_self());
      ::fflush(stderr);
    }
#endif
    gt->set_state(vi::GreenThreadState::kComplete);
    return;
  }
   if (out->IsPromise()) {
     data->threw = true;
     data->exc_is_error = true;
     data->exc_name = "TypeError";
     data->exc_msg = "green_threads body that awaits must be an async function";
     gt->set_state(vi::GreenThreadState::kComplete);
     return;
   }
   data->result_p.Reset(data->isolate, out);
#ifdef V8_ENABLE_GREEN_THREADS
   if (::getenv("GT_CRASH_DEBUG")) {
     ::fprintf(stderr,
               "[GTDBG-RESULT] round=%u gt=%p data=%p result_p=%p isempty=%d tid=%ld\n",
               (unsigned)data->round, (void*)gt, (void*)data, (void*)&data->result_p,
               (int)data->result_p.IsEmpty(), (long)pthread_self());
     ::fflush(stderr);
   }
#endif
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
  // Hold the green isolate current ONLY when bodies run concurrently -- the
  // GT_TRUE_PARALLEL path (ParallelBodyDispatchActive), where N OS workers drive
  // bodies on the SAME green isolate at once. Each worker must then attribute its
  // running GT to this worker's own PerIsolateThreadData (invariant I1) via
  // CurrentGreenThread()/SetCurrentGreenThread(), instead of the shared
  // process-wide g_current_green_thread_ slot. Without it, every worker reads and
  // writes ONE shared "current GT": a body dispatched to one worker is seen by
  // another as its own current and RunGreenThread() runs it a second time on a
  // concurrent thread. iteration_count_ is a plain non-atomic uint32_t, so the
  // two writers race (lost update -> a GreenJSEntry reads a torn iter before the
  // body completed) -> the SIGSEGV / empty-result crash we were chasing. Under
  // green_thread_mode several workers Enter/Exit this isolate concurrently on
  // purpose -- each worker's EntryStackItem chain routes to its own
  // PerIsolateThreadData, so the shared scope keeps scheduler bookkeeping
  // worker-local.
  //
  // Serialized guards (CoW writer scope, one body at a time) never set
  // ParallelBodyDispatchActive, so no scope is held here and the shared-global
  // slot is exactly what the CoW machinery relies on -- that path stays
  // byte-for-byte the pre-pool behavior.
  const bool hold_scope = vi::GreenThreadCow::ParallelBodyDispatchActive();
  g_current_pool_scheduler = sched;
  if (hold_scope && g_green_isolate != nullptr) {
    v8::Isolate::Scope iso_scope(g_green_isolate);
    sched->ScheduleLoop();
  } else {
    sched->ScheduleLoop();
  }
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_CRASH_DEBUG")) {
    for (uint32_t i = 0; i < gts.size(); i++)
      ::fprintf(stderr, "[GTDBG-ASSIGN] task=%u -> scheduler[%d%%] gt=%p\n", i,
                i % num_workers, (void*)gts[i]);
    ::fflush(stderr);
  }
#endif

  for (int i = 0; i < num_workers; i++)
    workers_.emplace_back(&GreenThreadPool::WorkerEntry, this, i);
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_CRASH_DEBUG")) {
    int64_t completed = 0;
    for (int s = 0; s < num_workers; s++)
      completed += schedulers_[s]->total_threads_completed();
    ::fprintf(stderr,
              "[GTDBG-POOL] num_workers=%d gts=%zu completed=%lld\n", num_workers,
              gts.size(), (long long)completed);
    for (int s = 0; s < num_workers; s++) {
      const char* firstgt = gts.empty() ? "?" : "x";
      ::fprintf(stderr, "  scheduler[%d] assigned gt=%p\n", s, firstgt);
    }
    ::fflush(stderr);
  }
#endif
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

  // Phase 2: allocate per-thread state for the now-known-valid tasks. datas[i]
  // maps to task i so the results array keeps task order. In shared-heap mode a
  // green async body runs GREEN on the green isolate (Option C) -- its control
  // flow stays green and `await` parks it cooperatively -- so it is set up by
  // SetupAndRunGreenAsyncBody and driven by the async drive loop; a sync body
  // also runs green here for parallelism. In serial mode there is no green
  // isolate, so an async body runs on the embedding isolate (SetupAndRunAsyncBody)
  // where Node's async bindings live.
  std::vector<GreenCallData*> datas(n);
  const bool shared_heap = GreenHeapEnabled();
  // Whether any task is a sync body. The green isolate and the always-allocation
  // (GC-free) window are only meaningful when a sync body will run on it; an
  // all-async dispatch runs purely on the embedding isolate and must not touch
  // the green heap or arm always-allocation on the embedding isolate.
  bool has_sync = false;
  bool has_async = false;
  for (uint32_t i = 0; i < n; i++) {
    if (GTIsAsyncFunction(isolate, fns[i]))
      has_async = true;
    else
      has_sync = true;
  }
  if (shared_heap) {
    // Create the green isolate + context when a sync body OR an async body
    // needs it. An async body runs green (Option C) and needs the green context
    // -- which hosts the gtRead I/O bridge global -- and its heap; a body that
    // is purely async still routes onto the green isolate. Sync bodies and
    // their argument copies each root their outputs in a Persistent, so nothing
    // dangles across the isolate switch.
    if (has_sync || has_async) {
      green_isolate_ret = GetGreenIsolate(isolate);
      vi_isolate_for_gt = reinterpret_cast<vi::Isolate*>(green_isolate_ret);
      // Arm the GC-free (AlwaysAllocate) window BEFORE any green-heap context
      // allocation. SetupAndRunGreenAsyncBody below (and EnsureGreenContext) call
      // Context::New(green), which runs the Genesis bootstrapper and allocates
      // Maps/functions; with GC enabled a collection could reclaim a bootstrap
      // Map still referenced mid-bootstrapping -> SIGSEGV in
      // JSObject::MigrateToMap (observed: deterministic once the green heap has
      // grown across several spawnAll rounds). Sync-body setup runs here too, so
      // the window must cover all of it. Released at the existing Leave site.
      node::GreenEnterAlwaysAllocation(green_isolate_ret);
      // Ensure the green context lives in the durable persistent g_green_ctx
      // (creating it + lifting embedding globals on first use).
      EnsureGreenContext(isolate, ctx);
    }

    // The shared green-async drive reactivates bodies through this scheduler.
    // It must exist before any SetupAndRunGreenAsyncBody runs -- a body's first
    // gtRead is submitted during setup and binds this scheduler as its Reactivate
    // target. Live for the whole RunTasks; destroyed once the drive completes.
    g_async_drive_scheduler = new vi::GreenScheduler(vi_isolate_for_gt);

    GreenValueCopier args_copier;
    args_copier.src = isolate;
    args_copier.dst = green_isolate_ret;
    for (uint32_t i = 0; i < n; i++) {
      if (GTIsAsyncFunction(isolate, fns[i])) {
        // Async body runs GREEN on the green isolate (Option C): its control
        // flow stays green and `await` parks it cooperatively; the async drive
        // loop below resumes it. Arguments are copied green (like sync bodies)
        // so the body touches only the green heap.
        datas[i] = SetupAndRunGreenAsyncBody(isolate, ctx, green_isolate_ret,
                                             fns[i], task_arrays[i]);
        continue;
      }
      // A green body: relocate the embedding Function onto the green isolate
      // (kept rooted on the embedding isolate so its captured closure survives)
      // and copy its arguments so bodies mutate private copies and the shared
      // `parent` in the harness is never corrupted. COW clones land on the
      // green heap too.
      Local<Array> task = task_arrays[i];
      GreenCallData* data = new GreenCallData();
      data->isolate = green_isolate_ret;
      // context_p is rooted from the durable persistent g_green_ctx -- no green
      // handle scope is needed.
      data->context_p.Reset(green_isolate_ret, g_green_ctx);
      // RelocateFunction returns a durable Persistent rooted inside itself, so
      // no caller handle scope is needed; Reset copies that handle into
      // data->fn_p. fn_isolate is green, so GreenJSEntry recovers the handle on
      // the green isolate.
      // RelocateFunction wants the green body context as a Local. Construct it
      // inside a green Isolate::Scope + HandleScope: at this point the embedding
      // thread holds no active HandleScope on the green isolate (EnsureGreenContext's
      // local HandleScope ended on return), so an unrooted Local<Context>::New(green)
      // would hit V8's "Cannot create a handle without a HandleScope" fatal -- this
      // was introduced when RelocateFunction's context argument was changed from a
      // context it created internally to one the caller builds. RelocateFunction
      // switches to green again internally; the returned Persistent is durable.
      std::unique_ptr<Persistent<Function>> relocated;
      {
        v8::Isolate::Scope green_iso_scope(green_isolate_ret);
        HandleScope green_scope(green_isolate_ret);
        relocated = RelocateFunction(
            isolate, fns[i],
            Local<Context>::New(green_isolate_ret, g_green_ctx));
      }
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
      datas[i] = data;
    }
  } else {
    for (uint32_t i = 0; i < n; i++) {
      if (GTIsAsyncFunction(isolate, fns[i])) {
        // Async body: runs on the embedding isolate (no green isolate in this
        // mode), bridged to the caller Promise.
        datas[i] = SetupAndRunAsyncBody(isolate, ctx, fns[i], task_arrays[i]);
        continue;
      }
      Local<Array> task = task_arrays[i];
      GreenCallData* data = new GreenCallData();
      data->isolate = isolate;
      data->fn_isolate = isolate;
      data->context_p.Reset(isolate, ctx);
      data->fn_p.Reset(isolate, fns[i]);
      const uint32_t argc = task->Length();
      for (uint32_t a = 1; a < argc; a++) {
        data->args_p.push_back(
            std::make_unique<Persistent<Value>>(isolate, task->Get(ctx, a).ToLocalChecked()));
      }
      datas[i] = data;
    }
  }

  const uint32_t m = datas.size();
  // Stamp every call data with this spawnAll's round so GT_CRASH_DEBUG prints
  // (worker RESULT and main RUNTASKS) can be correlated to the same call.
  const uint64_t round = g_green_spawn_round.fetch_add(1, std::memory_order_relaxed);
  for (GreenCallData* data : datas) data->round = round;

  // Create a green thread for each SYNC body only. Async bodies (data->awaiting)
  // hold no RUNNABLE green thread -- in serial mode they run on the embedding
  // isolate via SetupAndRunAsyncBody; in shared-heap mode they run GREEN via
  // SetupAndRunGreenAsyncBody and are driven by the async drive loop below. They
  // DO hold a bare GreenThread* descriptor (created in SetupAndRunGreenAsyncBody)
  // used only for cooperative block/resume, never scheduled by this pool. Gating
  // CreateGreenThread on !awaiting keeps g_gt_registry, the worker pool, and the
  // teardown loop below green-only.
  std::vector<vi::GreenThread*> gts;
  gts.reserve(datas.size());
  for (GreenCallData* data : datas) {
    if (data->awaiting) continue;
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
  const bool enable_cas = shared_heap && (has_sync || has_async);
  // GC-free allocation window armed BEFORE the workers spawn so thread creation
  // and every body's first allocation happen-before any collection. While the
  // count is non-zero the heap grows instead of collecting (young scavenger,
  // mark-compact, and incremental marking all skip), so no concurrent collection
  // can relocate or reclaim the green-isolate context or objects a worker still
  // holds mid-body. Without it, g.spawnAll() driving two or more green threads on
  // the shared green isolate tore the second body's green-isolate native context
  // out from under it (SIGSEGV, ~5% of runs under GT_NUM_WORKERS>1) -- the same
  // torn-context class the __m2_probe dispatcher fixes by holding this window
  // around its own concurrent fan-out. Held only in shared-heap mode, where
  // bodies run on the green isolate concurrently; serial Model-A execution never
  // needs it. The `(has_sync || has_async)` gate means an all-async dispatch
  // (no green threads, but N workers still allocating + checkpointing parked
  // async bodies on the shared green heap) gets this same GC-free window -- its
  // parked closures, gtRead resolvers, and settle values must be protected just
  // as sync bodies' are.
  const bool gt_shared = shared_heap && (has_sync || has_async);
  // The CAS bump-pointer flag stays here: it only matters once the N workers
  // allocate concurrently (after AlwaysAllocate is armed above).
  if (enable_cas) vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(working_isolate, true);
  // Handle-scope phase: this is the ONLY window where N OS workers run bodies
  // concurrently on the green heap, so opt into per-worker handle-state routing
  // here (and only here). Set before the workers spawn (thread create is a
  // happens-before) and cleared after the join (happens-after), so it is stable
  // true for every body and false throughout setup/teardown.
  vi::SetIsolateHandleScopePerWorker(reinterpret_cast<vi::Isolate*>(working_isolate), true);

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
    // TRUE-CPU-PARALLEL opt-in (GT_TRUE_PARALLEL): the discriminator dispatches
    // allocation-light, already-optimized bodies that touch no shared-writer
    // state, so it is safe to skip the whole-body GreenThreadCowSharedWriterScope
    // and let the N OS workers run concurrently. Guarded by an explicit opt-in on
    // purpose -- the allocation-heavy concurrent suite trips shared-compilation
    // (OSR) races without the whole-body serialize, so it stays serialized unless
    // it opts in. Set before the workers spawn (thread-create happens-before) and
    // cleared after the join, so setup/teardown keep the shared impl routing.
    const bool true_parallel = ::getenv("GT_TRUE_PARALLEL") != nullptr;
    if (true_parallel) {
      vi::GreenThreadCow::SetParallelBodyDispatchActive(true);
      // M2Probe's parallel path keeps every body entry serial with respect to
      // shared compile state by warming each body up ONCE on this embedding
      // thread (with concurrent compilation suspended) before any worker runs.
      // Doing the same here means the N OS workers never trigger Compile ->
      // Code/Map/feedback-vector mutation concurrently; by the pool runs the
      // optimized Code already exists and workers only READ it. Without this the
      // first concurrent body entry compiles on N workers at once, corrupting
      // shared compile state and tearing the JS-entry execution state (the
      // StackFrame::iteration_depth(this=0) / InterruptsScope SEGVs seen here).
      vi::GreenThreadCow::SetConcurrentCompilationSuspended(true);
      // All bodies of this dispatch share the same JS functions (the
      // discriminator hands every green thread the identical `body`/`work`),
      // whose SharedFunctionInfos live on the green isolate's compile cache.
      // Warm up ONE body: it runs work() past its maglev tier-up budget, which
      // triggers the maglev->optimized OSR compilation synchronously (compilation
      // is suspended here, so it runs inline on this embedding thread) and leaves
      // work() optimized for every worker -- so the pool run never compiles and
      // the concurrent-OSR frame-iterator race that SEGV'd here cannot occur.
      // Re-running all N bodies would cost a full serial baseline inside
      // green_ms and destroy the speedup metric, so prime just the first.
      if (!g_gt_registry.empty())
        PrimeBody(g_gt_registry.begin()->second);
    }

    GreenThreadPool pool;
    if (!gts.empty()) pool.Run(gts, vi_isolate_for_gt, num_workers);

    if (true_parallel) {
      vi::GreenThreadCow::SetParallelBodyDispatchActive(false);
      vi::GreenThreadCow::SetConcurrentCompilationSuspended(false);
    }
  }

  // Drive every green-async body to settlement INSIDE the AlwaysAllocate+CAS+
  // handle-scope-per-worker window (still armed above; released just below).
  // Their caller-facing Promises must be resolved (by the green settle reaction)
  // before results are gathered. Runs on the green isolate; a no-op when there
  // are no green-async bodies. N OS workers fan the bodies out across the pool,
  // so the window -- AlwaysAllocate (GC-free) + the CAS bump pointer (disjoint
  // concurrent allocation) + per-worker handle-scope routing -- must still be
  // held here. Concurrent compilation is suspended so the workers never compile
  // the green isolate's code paths at once; each async body has its own relocated
  // SFI, so CPU work still parallelizes (the bodies run phase-2 bytecode
  // unoptimized -- a documented perf note; warming them up is impractical since
  // each awaits I/O). The drive's own GreenGCPause is the belt-and-suspenders GC
  // guard for the parked bodies' promises and resolvers.
  if (shared_heap && has_async) {
    vi::GreenThreadCow::SetConcurrentCompilationSuspended(true);
    DriveGreenAsyncBodiesParallel(green_isolate_ret, datas);
    vi::GreenThreadCow::SetConcurrentCompilationSuspended(false);
  }

  // The drive owns all Reactivate targets; once it returns every body has
  // settled (no outstanding reads, no libuv pool thread will Reactivate again).
  // Tear the shared scheduler down so a late Reactivate (if any) hits stopping_
  // rather than a freed object, matching Shutdown's guard elsewhere.
  if (has_async && g_async_drive_scheduler != nullptr) {
    g_async_drive_scheduler->Shutdown();
    delete g_async_drive_scheduler;
    g_async_drive_scheduler = nullptr;
  }

  vi::SetIsolateHandleScopePerWorker(reinterpret_cast<vi::Isolate*>(working_isolate), false);
  if (enable_cas) vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(working_isolate, false);

  // Gather results. A body that threw had its exception captured and cleared
  // by GreenJSEntry (see there); we stop populating as soon as we notice so the
  // caller can propagate the original exception rather than construct a
  // replacement. Because we must not Set a return value once an exception is
  // pending, we surface the fact through *threw_out and, at the end, re-throw
  // the captured exception on this isolate.
  //
  // The GC-free (AlwaysAllocate) window is still held here (armed above and
  // released only just before return, below). The green-heap copy path runs
  // CopyValueAcross, which allocates rebuilt objects onto the shared green heap;
  // with collection re-enabled those allocations tripped the heap's
  // young_gc_during_full_gc_sweeping_ invariant (a FATAL) and, more subtly,
  // let a collection relocate/reclaim the young green objects the task results
  // and the Persistent handles reference -- so a result Persistent read back in
  // the loop came up empty (null Local<Value> -> SIGSEGV in GreenCapture).
  // __m2_probe proves this holds: it arms the window before the fan-out and
  // disarms only after gathering all results. Not armed off-CT mode, where
  // bodies never share the green heap and nothing here allocates on it.
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
    // Set the async rows first: their caller-facing Promises live on the
    // embedding isolate, so they must be built OUTSIDE the green Isolate::Scope
    // below. A green async body settled during the drive loop and captured its
    // settle value on the green heap; copy that value in here (on the embedding
    // isolate, whose context is current) into the caller resolver and resolve
    // it, so GetPromise() then returns the settled caller Promise to the JS
    // caller. A serial-mode async body's resolver was created in
    // SetupAndRunAsyncBody and is already resolved by its own reaction.
    for (uint32_t i = 0; i < m; i++) {
      if (!datas[i]->awaiting) continue;
      // The green async body settled during the drive loop; its reaction
      // captured the settle value as a green Persistent. Copy it to the
      // embedding isolate and resolve the caller-facing Promise so GetPromise()
      // returns a settled promise to the JS caller. Runs on the embedding
      // isolate with its context current (the JS context) -- exactly like the
      // sync result copy below -- so GreenRebuild lands objects in a valid
      // realm.
      Local<Promise::Resolver> resolver =
          Local<Promise::Resolver>::New(isolate, datas[i]->caller_resolver_p);
      std::unique_ptr<Persistent<Value>> emb_val;
      if (!datas[i]->settle_value_p.IsEmpty()) {
        GreenValueCopier copier;
        copier.src = green_isolate_ret;
        copier.dst = isolate;
        // Root green_val in a handle scope held ACROSS the copy. green_val is a
        // Local backed by a handle-stack slot; once that scope closes, the slot is
        // free and CopyValueAcross's own nested HandleScope reclaims it, clobbering
        // the value mid-copy (observed: a green number rebuilt into an embedding
        // object). Keep green_scope/green_hs live through CopyValueAcross so the
        // source handle stays valid.
        v8::Isolate::Scope green_scope(green_isolate_ret);
        HandleScope green_hs(green_isolate_ret);
        Local<Value> green_val =
            Local<Value>::New(green_isolate_ret, datas[i]->settle_value_p);
        emb_val = CopyValueAcross(copier, green_val);
        // Only resolve once the settle value was captured. If the drive loop
        // stalled (returned false before this body settled), leave the caller
        // promise pending -- the documented "valid, unsettled" outcome -- and
        // avoid dereferencing an empty emb_val (a null read).
        Local<Context> rctx = Local<Context>::New(isolate, datas[i]->caller_ctx_p);
        resolver->Resolve(rctx, Local<Value>::New(isolate, *emb_val));
      }
      results->Set(ctx, i, resolver->GetPromise()).Check();
    }
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
      // Async bodies' caller Promises were already Set outside the green scope
      // above; skip them here (they hold no green result_p).
      if (datas[i]->awaiting) continue;
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
#ifdef V8_ENABLE_GREEN_THREADS
      if (::getenv("GT_CRASH_DEBUG")) {
        ::fprintf(stderr,
                  "[GTDBG-RUNTASKS] round=%u i=%u gt=%p data=%p result_p=%p threw=%d result_p.isempty=%d tid=%ld\n",
                  (unsigned)datas[i]->round, i, (void*)datas[i]->gt,
                  (void*)datas[i], (void*)&datas[i]->result_p, (int)datas[i]->threw,
                  (int)datas[i]->result_p.IsEmpty(), (long)pthread_self());
        ::fflush(stderr);
      }
#endif
      std::unique_ptr<Persistent<Value>> emb_result_p =
          CopyValueAcross(out_copier, green_result);
      results->Set(ctx, i, Local<Value>::New(isolate, *emb_result_p)).Check();
    }
  } else {
    for (uint32_t i = 0; i < m; i++) {
      // Async bodies' caller Promises are Set below inline (embedding isolate).
      if (datas[i]->awaiting) {
        Local<Promise::Resolver> resolver =
            Local<Promise::Resolver>::New(isolate, datas[i]->caller_resolver_p);
        results->Set(ctx, i, resolver->GetPromise()).Check();
        continue;
      }
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
    // Async bodies are owned by their own settle reaction (GreenAsyncResolveFulfilled/
    // Rejected), which disposes the call data once the caller Promise settles;
    // deleting them here would use-after-free that pointer. Sync bodies always have
    // a green thread (shared and serial models alike) to tear down below.
    if (data->awaiting) continue;
    vi::GreenThreadCow::Teardown(data->gt);
    delete data->gt;
    // clear() destroys each unique_ptr, which Disposes its rooted Persistent.
    data->args_p.clear();
    data->result_p.Reset();
    data->fn_p.Reset();
    g_gt_registry.erase(data->gt);
    delete data;
  }

  // Teardown green-async bodies. The pool never created a RUNNABLE green thread
  // for them -- only a bare GreenThread* descriptor (created in
  // SetupAndRunGreenAsyncBody, freed below) used for cooperative block/resume.
  // The green settle reaction (GreenGreenCaptureSettle) only captured the body's
  // settle value on the green heap; RunTasks copied it into the embedding caller
  // resolver in the results loop above. So dispose the green body promise, the
  // captured settle value, the embedding caller resolvers, the bare descriptor,
  // and the copied green args/fn here. This runs after results gathered the
  // (now settled) caller Promises above.
  for (GreenCallData* data : datas) {
    if (!data->async_green) continue;
    data->async_promise_p.Reset();
    data->settle_value_p.Reset();
    data->caller_resolver_p.Reset();
    data->caller_ctx_p.Reset();
    data->args_p.clear();
    data->fn_p.Reset();
    // The pool never ran this descriptor through RunGreenThread/COW, so it needs
    // no GreenThreadCow::Teardown -- just free the bare GreenThread* created in
    // SetupAndRunGreenAsyncBody (its destructor is a no-op: no stack, no COW view).
    if (data->gt != nullptr) {
      delete data->gt;
      data->gt = nullptr;
    }
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
  if (gt_shared) node::GreenLeaveAlwaysAllocation(working_isolate);
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

// Coarse green-mutator lock (M2a baseline). See the declaration near the async
// driver (DriveGreenAsyncBodiesParallel) for the full rationale. It serializes
// concurrent body execution so at most ONE OS worker mutates the green heap's
// shared state (bump-pointer allocator, cached Maps, and any GC) at a time.
// Concurrent object-literal creation races here: 8 threads create objects of the
// same shape, resolve the single cached Map, and mutate it via
// Map::InobjectSlackTrackingStep while their young-space bump pointers collide
// -> EXC_BAD_ACCESS in Factory/Map stores.
// True all-mutator parallelism (I3 stop-the-world GC + I4 serialized allocation)
// is the later milestone; this lock is the "mutex-protected" phase the plan
// anticipates and the safe correctness baseline N workers run behind. Kept
// process-global: there is exactly one green isolate per process.
// (Defined next to the async driver; the field below is the declaration site's
//  companion usage context, not a second definition.)

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
#ifdef V8_ENABLE_GREEN_THREADS
  GTFaultGuard gt_fault_guard;
#endif
  Isolate* isolate = args.GetIsolate();
  HandleScope scope(isolate);
  Local<Context> ctx = isolate->GetCurrentContext();
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
    ::fprintf(stderr, "[GTGT-M2START] pthread=0x%lx iso=%p is_green=%d\n",
              (unsigned long)pthread_self(), (void*)isolate, (int)d.is_green);
    ::fflush(stderr);
  }
#endif

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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    ::fprintf(stderr,
              "[GTGT-M2GREEN] pthread=0x%lx emb=%p green=%p "
              "is_green=%d\n",
              (unsigned long)pthread_self(), (void*)isolate, (void*)green,
              (int)green->isGreenThreadMode());
    ::fflush(stderr);
  }
  if (::getenv("GT_GT_DEBUG")) ::setenv("GT_PHASE", "setup", 1);
#endif

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
    std::unique_ptr<Persistent<Function>> relocated = RelocateFunction(isolate, emb_fn, Local<Context>::New(green, g_green_ctx));

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
  const bool skip_warmup = (::getenv("GT_M2_WARMUP") != nullptr);
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) ::setenv("GT_PHASE", "warmup", 1);
  // Warmup compiles each body (first call triggers Compile -> Maglev/Turboshaft).
  // Suspend concurrent (background) compilation HERE, before warmup, so that
  // compilation runs synchronously on THIS embedding thread inside GreenJSEntry's
  // green Isolate::Scope -- producing Code linked to the green isolate/context.
  // If warmup compiled on a background compiler thread (embedding isolate), the
  // resulting Code embeds emb-derived pointers, and a worker running it on the
  // green heap faults (EXC_BAD_ACCESS at 0x3ff / 0xbeeddead). Disarmed after the
  // workers join, below.
  vi::GreenThreadCow::SetConcurrentCompilationSuspended(true);
#endif
  // Suppress EMBEDDING's own collection for the whole probe window, not just
  // green's. The green isolate has its own heap (objects on it fail the
  // embedding heap's Contains check), yet a green object is reachable from an
  // embedding GC root, so embedding's background mark-compact can walk it. In
  // true-parallel mode (GT_M2_PARALLEL) no CoW view is active, so
  // ShouldUseBackgroundThreads()==true and that concurrent GC runs while
  // workers mutate the heap in place -> torn-state SEGV. Always-allocate makes
  // the heap grow instead of collecting; arming it here (before warmup) means
  // no embedding collection is ever in flight while a body runs. Restored after
  // the workers join, below. This is the "safe access in ALL cases" guarantee:
  // while the shared heap is live no other isolate collects it.
  node::GreenEnterAlwaysAllocation(isolate);
  if (!skip_warmup) {
    for (Work& w : works) {
      vi::SetCurrentGreenThread(w.gt);
      GreenJSEntry(reinterpret_cast<vi::Isolate*>(green), w.gt);
    }
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
  // TRUE-PARALLEL bodies mutate the shared heap in place, so no CoW view is
  // active (ActiveCount()==0); workers reserve disjoint young-gen pointers (CAS
  // alloc via SetGreenHeapConcurrentAllocationEnabled) and write their own
  // objects.
  // Concurrent-JIT-compiler suspension for TRUE-PARALLEL body execution
  // (GT_M2_PARALLEL=1). In this mode no CoW view is active (ActiveCount()==0),
  // so GreenThreadCOWSuspendedConcurrentGC() is false and V8's concurrent
  // Turbofan / Maglev / baseline-batch compilers would compile on background OS
  // workers against the heap the mutating workers rewrite in place -- torn
  // SharedFunctionInfo / feedback-vector / Map reads that land on a dead pointer
  // (EXC_BAD_ACCESS at 0xbeeddead) on the next deopt. Force those compilation
  // paths synchronous (no background compiler thread touches the heap) for the
  // whole concurrent body window; GC stays concurrent (Heap::ShouldUseBackground
  // Threads is untouched) so the parallel allocation window never runs out of
  // heap. Complements SetGreenHeapConcurrentAllocationEnabled, which makes the
  // young-gen pointers disjoint. Disarmed after the workers join, below.
  vi::GreenThreadCow::SetConcurrentCompilationSuspended(true);
  // Item 1 (I3) / item 2: announce the stop-the-world GC BEFORE spawning the
  // workers, so their entry safepoints observe the request and park before
  // allocating. This quiescence (not the old
  // GreenThreadCOWSuspendedConcurrentGC suppression) is what makes the
  // synchronous collection below safe: the collector walks only published,
  // reported, quiescent roots.
  if (force_gc_mode != M2ForcedGCMode::kNone) {
    vi::GreenThreadCow::RequestAllMutatorsStop(nullptr, nullptr);
  }
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) ::setenv("GT_PHASE", "workers", 1);
#endif
  {
    v8::Isolate::Scope green_scope(green);

    // GC-free allocation window for the concurrent body execution below.
    // While this count is non-zero the heap grows instead of collecting (young
    // scavenger, mark-compact, and incremental marking all skip), so no
    // worker-triggered GC can reclaim or relocate an object that another
    // worker still references mid-body -- making every fast-path map /
    // dictionary-fallback / heap-object-field read in a JIT body observe a
    // consistent heap. Without this, a concurrent collection leaves a freed
    // pointer (faulting x5==0x400 on a reclaimed object) in a live body -- the
    // SIGSEGV the 8-worker parallel allocation hit. Armed BEFORE the workers
    // spawn so thread creation happens-before each body's first allocation;
    // the forced collection below is taken OUTSIDE this window and re-arms it,
    // and GC resumes after the workers join (all bodies finished, no live
    // mid-body reference).
    node::GreenEnterAlwaysAllocation(green);

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
    // Handle-scope phase: N OS workers run bodies concurrently here, so opt into
    // per-worker handle-state routing on the green isolate (and only here). Set
    // before the workers spawn (thread create is a happens-before) and cleared
    // after the join, so the single-threaded M2Probe setup sequence that ran
    // above keeps the shared IsolateData handle cursor unchanged.
    vi::SetIsolateHandleScopePerWorker(reinterpret_cast<vi::Isolate*>(green), true);
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
    // The GC-free window above would make this collection a no-op -- with
    // always_allocate active the heap only grows, so M2ForceGC could never
    // reclaim the parked markers. Turn it OFF for the collection, then re-arm
    // it BEFORE the workers resume and run their bodies: the bodies, not the
    // collection, are the concurrent window that must observe a consistent heap.
    node::GreenLeaveAlwaysAllocation(green);
    M2ForceGC(green, force_gc_mode);
    node::GreenEnterAlwaysAllocation(green);
  }
   // Resume the parked workers so the joined bodies can run and finish.
   if (force_gc_mode != M2ForcedGCMode::kNone) {
     vi::GreenThreadCow::ResumeAllMutators();
   }
   // Release any worker still parked in WaitAndClaim, then join the pool.
   pool.Shutdown();
   for (std::thread& wt : workers) wt.join();
   // Bodies are done and no OS worker is executing a green body on the green
   // isolate anymore, so restore the shared handle-scope routing for any
   // post-join teardown on the embedding thread.
   vi::SetIsolateHandleScopePerWorker(reinterpret_cast<vi::Isolate*>(green), false);
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_GT_DEBUG")) {
      GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
      ::fprintf(stderr, "[GTGT-PHASE] after-join iso=%p is_green=%d\n",
                (void*)isolate, (int)d.is_green);
      ::fflush(stderr);
    }
#endif
    vi::GreenThreadCow::SetConcurrentCompilationSuspended(false);
    // Bodies are done and every mid-body worker reference is now dead, so end
    // the GC-free window here: ordinary GC may reclaim/relocate freely again.
    node::GreenLeaveAlwaysAllocation(green);
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_GT_DEBUG")) {
      GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
      ::fprintf(stderr, "[GTGT-PHASE] after-GreenLeaveAlways iso=%p is_green=%d\n",
                (void*)isolate, (int)d.is_green);
      ::fflush(stderr);
    }
#endif
  }

  // Release the embedding-heap suppression now that all workers have joined and
  // no body is still mutating the heap, so embedding may reclaim normally again.
  node::GreenLeaveAlwaysAllocation(isolate);

  vi::GreenThreadCow::SetGreenHeapConcurrentAllocationEnabled(green, false);

#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
    ::fprintf(stderr, "[GTGT-PHASE] pre-ArrayNew iso=%p is_green=%d\n",
              (void*)isolate, (int)d.is_green);
    ::fflush(stderr);
  }
#endif

  // 3. Copy results back to the embedding isolate.
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) ::setenv("GT_PHASE", "copyback", 1);
#endif
  Local<Array> results = Array::New(isolate, n);
  bool all_ok = true;
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
    ::fprintf(stderr, "[GTGT-PHASE] collection START iso=%p is_green=%d\n",
              (void*)isolate, (int)d.is_green);
    ::fflush(stderr);
  }
#endif
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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
    ::fprintf(stderr, "[GTGT-PHASE] copyback DONE iso=%p is_green=%d\n",
              (void*)isolate, (int)d.is_green);
    ::fflush(stderr);
  }
#endif

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
#ifdef V8_ENABLE_GREEN_THREADS
  if (::getenv("GT_GT_DEBUG")) {
    GTDebugCtx d = GTGreenThreadDebugCtx(isolate);
    ::fprintf(stderr, "[GTGT-PHASE] M2 report SET iso=%p is_green=%d\n",
              (void*)isolate, (int)d.is_green);
    ::fflush(stderr);
  }
#endif
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
  // Green `require(name)` global (sync bodies share this context). Async bodies
  // also get it via InstallGreenAsyncGlobals. Loads a Node module on the
  // embedding realm and returns its exports as a green proxy.
  {
    Local<Function> req =
        FunctionTemplate::New(g_green_isolate, GreenBridgeRequire)
            ->GetFunction(green_ctx)
            .ToLocalChecked();
    set_global("require", req);
  }
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

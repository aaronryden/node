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
#include <utility>
#include <vector>

#include "node_binding.h"
#include "util.h"

#include "v8.h"

#include "src/common/ptr-compr.h"
#include "src/execution/green-thread.h"
#include "src/execution/green-thread-cow.h"
#include "src/execution/green-thread-scheduler.h"
#include "src/execution/heap-ref-table.h"

namespace node {
namespace {

namespace vi = v8::internal;

using v8::Array;
using v8::BigInt;
using v8::Context;
using v8::Function;
using v8::FunctionCallbackInfo;
using v8::HandleScope;
using v8::Isolate;
using v8::Local;
using v8::Number;
using v8::Object;
using v8::Persistent;
using v8::String;
using v8::Value;

using vi::Address;  // v8::internal::Address (untagged heap pointer)

// Per-green-thread JavaScript invocation state. A single instance is shared
// between the scheduler's C++ entry point and the running spawnAll call.
struct GreenCallData {
  Isolate* isolate = nullptr;
  Persistent<Context> context_p;  // The context the body runs in.
  Persistent<Function> fn_p;      // The user's green-thread body.
  // Arguments passed to the body. Stored as local handles rather than persistent
  // ones: v8::Persistent<Value> is non-copyable and this stripped V8 has no
  // UniquePersistent, so a std::vector<Persistent<Value>> cannot grow. The
  // handles stay rooted because they live within the RunTasks HandleScope that
  // drives ScheduleLoop (and therefore every GreenJSEntry) below them.
  std::vector<Local<Value>> args_p;
  Persistent<Value> result_p;   // Body return value.
  bool threw = false;
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

  if (::getenv("GT_COW_DEBUG"))
    ::fprintf(stderr, "[GTDBG-BODY] GreenJSEntry body start iter=%u os=%ld gt=%p current=%p\n",
              (unsigned)gt->iteration_count(), (long)pthread_self(), (void*)gt, (void*)vi::CurrentGreenThread());

  // Phase machine: iteration 1 runs the body; anything else has already run.
  // In the recursive Model A, ScheduleLoop calls RunGreenThread once per GT and
  // Yield() drives the rest via recursion, so the body always runs at iter 1.
  // CurrentGreenThread() is already set to this GT by the caller before entry.
  if (gt->iteration_count() >= 2) {
    return;
  }

  HandleScope scope(data->isolate);
  Local<Context> ctx = Local<Context>::New(data->isolate, data->context_p);
  Context::Scope context_scope(ctx);

  Local<Function> fn = Local<Function>::New(data->isolate, data->fn_p);
  Local<Value> recv = ctx->Global();

  std::vector<Local<Value>> argv;
  argv.reserve(data->args_p.size());
  for (auto& handle : data->args_p) {
    argv.push_back(Local<Value>::New(data->isolate, handle));
  }

  Local<Value> out;
  if (::getenv("GT_COW_DEBUG")) {
    ::fprintf(stderr, "[GTDBG-NODE] fn->Call-BEGIN iter=%u current==gt=%d count=%u cow_en=%d ref_table=%p\n",
              (unsigned)gt->iteration_count(),
              (vi::CurrentGreenThread() == gt) ? 1 : 0,
              (unsigned)vi::GreenThreadCow::ActiveCount(),
              (int)gt->cow_enabled(), (void*)gt->cow_ref_table());
  }
  if (!fn->Call(ctx, recv, argv.size(), argv.data()).ToLocal(&out)) {
    data->threw = true;
    gt->set_state(vi::GreenThreadState::kComplete);
    return;
  }
  if (::getenv("GT_COW_DEBUG")) {
    ::fprintf(stderr, "[GTDBG-NODE] fn->Call-END iter=%u current==gt=%d count=%u ref_table=%p\n",
              (unsigned)gt->iteration_count(),
              (vi::CurrentGreenThread() == gt) ? 1 : 0,
              (unsigned)vi::GreenThreadCow::ActiveCount(),
              (void*)gt->cow_ref_table());
  }
  data->result_p.Reset(data->isolate, out);
  gt->set_state(vi::GreenThreadState::kComplete);
}

// Retrieves (creating if necessary) the process-local green scheduler for the
// given isolate. A single scheduler owns the global runqueue and processor
// pool; green threads are inherently one-scheduler-per-embedding. The scheduler
// is initialized (processors created) once, lazily, on first use.
vi::GreenScheduler* GetScheduler(Isolate* isolate) {
  static std::unordered_map<vi::Isolate*, std::unique_ptr<vi::GreenScheduler>>
      schedulers;
  vi::Isolate* vi_isolate = reinterpret_cast<vi::Isolate*>(isolate);
  auto it = schedulers.find(vi_isolate);
  if (it != schedulers.end()) return it->second.get();

  auto scheduler = std::unique_ptr<vi::GreenScheduler>(
      new vi::GreenScheduler(vi_isolate));
  vi::GreenScheduler* raw = scheduler.get();
  raw->Initialize(1);
  schedulers[vi_isolate] = std::move(scheduler);
  return raw;
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
  if (::getenv("GT_COW_DEBUG"))
    ::fprintf(stderr, "[GTDBG-NODE] CreateGreenThread gt=%p os_thread=%ld\n",
              (void*)gt, (long)pthread_self());
  gt->set_pc_raw(reinterpret_cast<Address>(&GreenJSEntry));
  gt->set_state(vi::GreenThreadState::kRunnable);
  vi::GreenThreadCow::Initialize(gt);
  if (::getenv("GT_COW_DEBUG"))
    ::fprintf(stderr, "[CREATE-GT] gt=%p ref_table=%p cow_en=%d iter=%d\n",
              (void*)gt, (void*)gt->cow_ref_table(), (int)gt->cow_enabled(),
              (unsigned)gt->iteration_count());
  return gt;
}

// Shared core: schedule every task in `tasks` ([[fn, ...args]]) and run all of
// their green threads to completion. Returns the per-task results array.
//
// If a body throws, its exception is left pending on the isolate and *threw_out
// is set so the caller avoids returning a value and lets that exception surface.
// A malformed `tasks` argument throws before any green thread is scheduled.
Local<Array> RunTasks(Isolate* isolate, Local<Context> ctx, Local<Array> tasks,
                      bool* threw_out) {
  HandleScope scope(isolate);
  const uint32_t n = tasks->Length();
  *threw_out = false;

  vi::Isolate* vi_isolate = reinterpret_cast<vi::Isolate*>(isolate);
  vi::GreenScheduler* scheduler = GetScheduler(isolate);

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
      return Array::New(isolate, 0);
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
      return Array::New(isolate, 0);
    }
    fns.push_back(fn);
    task_arrays.push_back(task);
  }

  // Phase 2: allocate per-thread state for the now-known-valid tasks.
  std::vector<GreenCallData*> datas;
  datas.reserve(n);
  for (uint32_t i = 0; i < n; i++) {
    Local<Array> task = task_arrays[i];
    GreenCallData* data = new GreenCallData();
    data->isolate = isolate;
    data->context_p.Reset(isolate, ctx);
    data->fn_p.Reset(isolate, fns[i]);
    const uint32_t argc = task->Length();
    for (uint32_t a = 1; a < argc; a++) {
      data->args_p.push_back(task->Get(ctx, a).ToLocalChecked());
    }
    datas.push_back(data);
  }

  const uint32_t m = datas.size();

  // Create each green thread (which enables COW, allocates its ref table and
  // registers it as COW-active), register its call data, and enqueue it into
  // the scheduler runqueue.
  for (GreenCallData* data : datas) {
    vi::GreenThread* gt = CreateGreenThread(vi_isolate);
    data->gt = gt;
    g_gt_registry[gt] = data;
    scheduler->EnqueueGlobal(gt);
  }

  // Drive every green thread to completion.
  scheduler->ScheduleLoop();

  // Gather results. A body that threw leaves its exception pending on the
  // isolate (see GreenJSEntry); we stop populating as soon as we notice so the
  // caller can propagate the original exception rather than construct a
  // replacement. Because we must not Set a return value once an exception is
  // pending, we surface the fact through *threw_out.
  Local<Array> results = Array::New(isolate, m);
  bool threw = false;
  for (uint32_t i = 0; i < m; i++) {
    if (datas[i]->threw) {
      threw = true;
      break;
    }
    results->Set(ctx, i, Local<Value>::New(isolate, datas[i]->result_p)).Check();
  }
  if (threw) {
    *threw_out = true;
  }

  // Release green threads and call data.
  for (GreenCallData* data : datas) {
    vi::GreenThreadCow::Teardown(data->gt);
    delete data->gt;
    data->result_p.Reset();
    data->fn_p.Reset();
    g_gt_registry.erase(data->gt);
    delete data;
  }

  return results;
}

}  // namespace

// ====================================================================
// DEBUG: opaque native logging hook, kept alive so the JIT cannot elide the
// body statements it wraps. Prints its JS args as text to stderr under
// GT_COW_DEBUG. Lets the body prove whether the write happens on every call.
void Dbg(const FunctionCallbackInfo<Value>& args) {
  Isolate* isolate = args.GetIsolate();
  std::string out;
  for (int i = 0; i < args.Length(); i++) {
    Local<Value> v = args[i];
    if (v->IsNumber()) {
      char buf[64];
      snprintf(buf, sizeof(buf), "%.17g", Local<Number>::Cast(v)->Value());
      out += buf;
    } else if (v->IsString()) {
      String::Utf8Value utf8(isolate, v);
      out += *utf8;
    } else if (v->IsObject()) {
      Address a = UntaggedObjectAddress(Local<Object>::Cast(v));
      char buf[32];
      snprintf(buf, sizeof(buf), "0x%lx", static_cast<unsigned long>(a));
      out += buf;
    } else if (v->IsUndefined()) {
      out += "undefined";
    } else if (v->IsNull()) {
      out += "null";
    } else {
      out += "?";
    }
    out += "|";
  }
  if (getenv("GT_COW_DEBUG") != nullptr) {
    ::fprintf(stderr, "[DBG] %s\n", out.c_str());
    ::fflush(stderr);
  }
  args.GetReturnValue().SetUndefined();
}

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

// green_threads.yield()
// Cooperative checkpoint: hands control to the next green thread and returns
// when scheduling returns here. No-op when called outside a running green
// thread (there is nothing to yield to).
void Yield(const FunctionCallbackInfo<Value>& args) {
  vi::GreenThread* gt = vi::CurrentGreenThread();
  if (gt == nullptr) return;  // Not inside a green thread; nothing to yield.
  vi::GreenScheduler* scheduler = GetScheduler(args.GetIsolate());
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
  if (getenv("GT_COW_DEBUG") != nullptr) {
    ::fprintf(stderr, "[CowCloneOf] lookup addr=0x%lx\n", static_cast<unsigned long>(addr));
    ::fflush(stderr);
  }
  if (gt->cow_ref_table()->FindCopy(addr, &copy)) {
    args.GetReturnValue().Set(
        BigInt::NewFromUnsigned(isolate, static_cast<uint64_t>(copy)));
  } else {
    args.GetReturnValue().SetUndefined();
  }
}

// TEMPORARY diagnostic: force-clone via the C++ path (deep-test Path A),
// bypassing the write-barrier. Returns the copy addr or undefined.
void CowForce(const FunctionCallbackInfo<Value>& args) {
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
  Address copy = vi::GreenThreadCow::CowCheckBeforeStore(gt->isolate(), addr);
  if (copy == addr) {
    args.GetReturnValue().SetUndefined();
  } else {
    args.GetReturnValue().Set(
        BigInt::NewFromUnsigned(isolate, static_cast<uint64_t>(copy)));
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
  node::SetMethod(context, exports, "spawn", Spawn);
  node::SetMethod(context, exports, "spawnAll", SpawnAll);
  node::SetMethod(context, exports, "yield", Yield);
  node::SetMethod(context, exports, "addrOf", AddrOf);
  node::SetMethod(context, exports, "cowCloneOf", CowCloneOf);
  node::SetMethod(context, exports, "__dbg", Dbg);
   node::SetMethod(context, exports, "__cowForce", CowForce);
  node::SetMethod(context, exports, "version",
                  [](const FunctionCallbackInfo<Value>& args) {
                    args.GetReturnValue().Set(String::NewFromUtf8(
                        args.GetIsolate(), "1",
                        v8::NewStringType::kNormal).ToLocalChecked());
                  });
}

}  // namespace node

NODE_BINDING_CONTEXT_AWARE_INTERNAL(green_threads, node::InitGreenThreads)

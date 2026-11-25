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
#include "src/objects/js-function.h"
#include "src/objects/js-function-inl.h"
#include "src/objects/scope-info.h"
#include "src/objects/scope-info-inl.h"
#include "src/objects/contexts.h"
#include "src/objects/contexts-inl.h"
#include "src/objects/string.h"
#include "src/objects/string-inl.h"
#include "src/api/api.h"
#include "src/api/api-inl.h"

namespace node {
namespace vi = v8::internal;

// Read the GT_M2_FORCE_GC environment variable. A single getenv() compare over
// the disabled path is negligible and race-free (M2Probe may run with workers on
// multiple OS threads), and when unset the caller performs no GC at all, so the
// existing probe path is unchanged.
M2ForcedGCMode M2ParseForcedGCEnv() {
  const char* val = ::getenv("GT_M2_FORCE_GC");
  if (val == nullptr) return M2ForcedGCMode::kNone;
  std::string s(val);
  if (s == "young") return M2ForcedGCMode::kYoung;
  if (s == "full") return M2ForcedGCMode::kFull;
  if (s == "both") return M2ForcedGCMode::kBoth;
  // Unknown value: behave as unset rather than aborting the whole probe.
  return M2ForcedGCMode::kNone;
}

// Whether `emb_fn` is a JavaScript async function (async function, async arrow,
// async generator, or async concise method). Detected from the SharedFunctionInfo's
// FunctionKind so an async body can be routed to the embedding realm -- where
// Node's async I/O bindings (setTimeout / process / require) actually live --
// WITHOUT running it. Running the head just to observe a returned Promise would
// double-execute the synchronous prefix and steal green-isolate parallelism from
// genuinely synchronous bodies, so the check is a static FunctionKind read.
bool GTIsAsyncFunction(v8::Isolate* /*emb_isolate*/,
                       v8::Local<v8::Function> emb_fn) {
  v8::Function* emb_fn_ptr = *emb_fn;
  vi::Tagged<vi::JSFunction> fn_tag =
      vi::Cast<vi::JSFunction>(*v8::Utils::OpenHandle(emb_fn_ptr));
  vi::Tagged<vi::SharedFunctionInfo> sfi_tag = fn_tag->shared();
  if (sfi_tag.is_null()) return false;
  return vi::IsAsyncFunction(sfi_tag->kind());
}

// Force a young and/or full collection on the green isolate, gated by `mode`.
// See green_threads_gc.h for the full rationale. The caller guarantees the active
// COW green threads are parked + rooted, so a synchronous collection on the green
// heap walks only reported, quiescent state.
void M2ForceGC(v8::Isolate* green_isolate, M2ForcedGCMode mode) {
  if (mode == M2ForcedGCMode::kNone || green_isolate == nullptr) return;
  vi::Isolate* vi_iso = reinterpret_cast<vi::Isolate*>(green_isolate);
  vi::Heap* heap = vi_iso->heap();
  if (heap == nullptr) return;

  const vi::GCFlag kNoFlags = vi::GCFlag::kNoFlags;
  const vi::GarbageCollectionReason kReason = vi::GarbageCollectionReason::kTesting;
  const v8::GCCallbackFlags kForced = v8::GCCallbackFlags::kGCCallbackFlagForced;

  // A single full (mark-compact) forced collection. It is run while the green
  // views are parked + rooted (see M2Probe), so it walks only their published,
  // quiescent roots -- exactly what the I3 handshake guarantees.
  //
  // A forced full collection is *forced* (the kForced callback flag sets
  // Heap::is_current_gc_forced_), so SelectMarkingType() picks atomic marking:
  // the cppgc concurrent-marking task is never started, which keeps the marker
  // cleanly torn down. The green isolate's incremental marking is stopped by
  // GreenGCPause across the probe (RelocateFunction re-parse + warmup must run
  // with GC frozen), and a non-forced young collection here would start a
  // concurrent marker that races the stopped marking during teardown. A full
  // mark-compact evacuates the nursery too (all roots are marked), so the single
  // full collection safely subsumes a young + full sequence for any mode.
  if (mode != M2ForcedGCMode::kNone) {
    heap->CollectAllGarbage(kNoFlags, kReason, kForced);
  }
}

// Enter the GC-free allocation window on the green isolate. See green_threads_gc.h:
// with always_allocate non-zero the heap grows instead of collecting, so no
// concurrent collection can reclaim or relocate an object a live worker still
// references mid-body (the parallel-allocation SIGSEGV fix). Called around the
// concurrent body window from M2Probe in green_threads.cc.
void GreenEnterAlwaysAllocation(v8::Isolate* green_isolate) {
  if (green_isolate == nullptr) return;
  vi::Heap* heap = reinterpret_cast<vi::Isolate*>(green_isolate)->heap();
  if (heap == nullptr) return;
  heap->EnterAlwaysAllocation();
}

void GreenLeaveAlwaysAllocation(v8::Isolate* green_isolate) {
  if (green_isolate == nullptr) return;
  vi::Heap* heap = reinterpret_cast<vi::Isolate*>(green_isolate)->heap();
  if (heap == nullptr) return;
  heap->LeaveAlwaysAllocation();
}

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

// Discover the names of the variables the green body `emb_fn` captures from its
// enclosing lexical scopes (its free variables), innermost-first.
//
// Defined here -- not in green_threads.cc -- because the walk reads the internal
// scope-info / context chain (js-function.h -> Context::previous(),
// ScopeInfo::ContextInlinedLocalName), and those headers drag V8's heap.h +
// V8 tracing/trace-event.h in. Keeping that here (the single TU allowed to
// include V8's heap headers) leaves heap.h, and therefore V8's trace-event.h,
// out of green_threads.cc, which must keep node's tracing/trace_event.h (see
// GreenGCPause above). Returns plain std::string names only: every V8-internal
// operation -- reinterpretting the public handle, the context walk, and String
// decoding -- happens inside this TU, so the caller keeps its public-API surface.
std::vector<std::string> CaptureGreenClosureVars(v8::Isolate* emb_isolate,
                                                 v8::Local<v8::Function> emb_fn) {
  std::vector<std::string> names;
  std::unordered_set<std::string> visited;  // a name is captured once.
  // CommonJS wrapper bindings are module/embedding infrastructure, not captured
  // user state -- skip them so only genuine free vars reach the green global.
  auto is_commonjs_builtin = [](const std::string& n) {
    return n == "__dirname" || n == "__filename" || n == "__paths" ||
           n == "exports" || n == "module" || n == "require";
  };
  {
    v8::Isolate::Scope isolate_scope(emb_isolate);
    v8::HandleScope scope(emb_isolate);  // root the closure-context chain.
    // Bridge the public Function to the internal JSFunction. V8's own
    // OpenHandle handles the !V8_ENABLE_DIRECT_HANDLE slot indirection (and any
    // pointer compression) exactly as the handle machinery intends, yielding a
    // typed Handle<JSReceiver>; Cast downcasts to JSFunction. A raw "deref the
    // handle slot then strip the tag" pointer instead lands on the arrow's Map
    // (its first word), where context() is not a Context -- so OpenHandle is
    // required, not merely the tag strip. This is the one TU permitted to
    // include V8's api headers, so none leak into green_threads.cc.
    v8::Function* emb_fn_ptr = *emb_fn;
    vi::Tagged<vi::JSFunction> fn_tag =
        vi::Cast<vi::JSFunction>(*v8::Utils::OpenHandle(emb_fn_ptr));
    // Access through the Tagged, not a raw JSFunction*: in this pointer-
    // compressed build the accessors need the cage base the Tagged carries,
    // so a reinterpret_cast<JSFunction*>(fn_tag.ptr()) reads unmapped memory.
    vi::Tagged<vi::Context> ctx = fn_tag->context();
    while (!ctx.is_null()) {
      vi::Tagged<vi::ScopeInfo> si = ctx->scope_info();
      if (!vi::Is<vi::ScopeInfo>(si)) break;  // dangling scope_info: stop safely.
      if (si->IsEmpty()) break;  // native/global empty-scope info: stop the walk.
      if (si->HasInlinedLocalNames()) {  // forward name enumeration is available.
        const int count = si->ContextLocalCount();
        for (int var = 0; var < count; var++) {
          vi::Tagged<vi::String> name_tag = si->ContextInlinedLocalName(var);
          if (vi::ScopeInfo::VariableIsSynthetic(name_tag)) continue;  // `this` / temporaries.
          vi::DirectHandle<vi::String> name_direct(name_tag,
                reinterpret_cast<vi::Isolate*>(emb_isolate));
          size_t utf8_len =
              vi::String::Utf8Length(reinterpret_cast<vi::Isolate*>(emb_isolate),
                                     name_direct);
          std::string name(utf8_len + 1, '\0');
          vi::String::WriteUtf8(reinterpret_cast<vi::Isolate*>(emb_isolate),
                                name_direct, &name[0], name.size(),
                                vi::String::Utf8EncodingFlag::kNullTerminate);
          name.resize(utf8_len);
          if (visited.count(name) || is_commonjs_builtin(name)) continue;
          visited.insert(name);
          names.push_back(name);
        }
      }
      ctx = ctx->previous();
    }
  }
  return names;
}

// Read the current values of `emb_fn`'s captured free variables. Walks the
// closure-context chain exactly like CaptureGreenClosureVars; for each inlined
// local `var` of a context, ScopeInfo::ContextInlinedLocalName(var) names it and
// Context::Get(ctx, ContextHeaderLength() + var, isolate) reads the value bound
// at that SAME slot (the ScopeInfo name index `var` maps to the context slot
// ContextHeaderLength() + var; see ScopeInfo::ContextSlotIndex). The
// context-cell unwrapping in Context::Get turns a kConst/kSmi/kInt32/kFloat64
// slot into the actual bound value. Each value is bridged to a public
// Local<Value> with Utils::Convert (the one internal->public bridge, and it is
// a plain slot reinterpretation, so it is safe for any tagged value). The value
// is yielded while the enclosing HandleScope keeps it rooted -- the caller must
// consume it inside the callback (e.g. structured-capture it) since no Local is
// returned to the caller's own scope. TheHole slots (a name with no live binding)
// are skipped. Names, order, and the CommonJS/synthetic/dedup filters match
// CaptureGreenClosureVars, so a value is yielded for every name that helper
// returned. This TU owns every V8-internal operation, so the caller (which stays
// on the public API) just receives (name, value) pairs.
void CaptureGreenClosureValues(
    v8::Isolate* emb_isolate, v8::Local<v8::Function> emb_fn,
    const std::function<void(const std::string&, v8::Local<v8::Value>)>& yield) {
  auto is_commonjs_builtin = [](const std::string& n) {
    return n == "__dirname" || n == "__filename" || n == "__paths" ||
           n == "exports" || n == "module" || n == "require";
  };
  std::unordered_set<std::string> visited;  // a name is captured once.
  {
    v8::Isolate::Scope isolate_scope(emb_isolate);
    v8::HandleScope scope(emb_isolate);  // root the closure-context chain.
    vi::Isolate* vi_iso = reinterpret_cast<vi::Isolate*>(emb_isolate);
    v8::Function* emb_fn_ptr = *emb_fn;
    vi::Tagged<vi::JSFunction> fn_tag =
        vi::Cast<vi::JSFunction>(*v8::Utils::OpenHandle(emb_fn_ptr));
    vi::Tagged<vi::Context> ctx = fn_tag->context();
    while (!ctx.is_null()) {
      vi::Tagged<vi::ScopeInfo> si = ctx->scope_info();
      if (!vi::Is<vi::ScopeInfo>(si)) break;
      if (si->IsEmpty()) break;  // native/global empty-scope info: stop the walk.
      if (si->HasInlinedLocalNames()) {  // forward name enumeration is available.
        vi::DirectHandle<vi::Context> ctx_dh(ctx, vi_iso);
        const int count = si->ContextLocalCount();
        for (int var = 0; var < count; var++) {
          vi::Tagged<vi::String> name_tag = si->ContextInlinedLocalName(var);
          if (vi::ScopeInfo::VariableIsSynthetic(name_tag)) continue;  // `this` / temporaries.
          vi::DirectHandle<vi::String> name_direct(name_tag, vi_iso);
          size_t utf8_len = vi::String::Utf8Length(vi_iso, name_direct);
          std::string name(utf8_len + 1, '\0');
          vi::String::WriteUtf8(vi_iso, name_direct, &name[0], name.size(),
                                vi::String::Utf8EncodingFlag::kNullTerminate);
          name.resize(utf8_len);
          if (visited.count(name) || is_commonjs_builtin(name)) continue;
          visited.insert(name);
          // The value bound at this same context slot. Context::Get unwraps the
          // context cell (kConst/kSmi/kInt32/kFloat64) to the real bound value.
          // The ScopeInfo local `var` is a *name* index; its binding lives in the
          // context slot ContextHeaderLength() + var (scope-info.cc: ContextSlotIndex
          // maps name->ContextHeaderLength()+index), so read THAT slot, not `var`.
          vi::DirectHandle<vi::Object> value_dh =
              vi::Context::Get(ctx_dh, si->ContextHeaderLength() + var, vi_iso);
          if (vi::IsTheHole(*value_dh)) continue;  // no live binding at this slot.
          v8::Local<v8::Value> value =
              v8::Utils::Convert<vi::Object, v8::Value>(value_dh);
          yield(name, value);
        }
      }
      ctx = ctx->previous();
    }
  }
}

// Debug-only snapshot of the executing OS thread's green-thread execution-state
// context. See green_threads_gc.h.
GTDebugCtx GTGreenThreadDebugCtx(v8::Isolate* iso) {
  vi::Isolate* v = reinterpret_cast<vi::Isolate*>(iso);
  GTDebugCtx r{false, nullptr, nullptr, nullptr};
  if (v == nullptr) return r;
  r.is_green = v->is_green_thread_mode();
  r.main_iso = static_cast<void*>(v->main_thread_local_isolate());
  r.main_heap = static_cast<void*>(v->main_thread_local_heap());
  r.tblock = static_cast<void*>(v->thread_local_top());
  return r;
}

void* GTGreenThreadCurrentIsolate() {
  vi::Isolate* cur = vi::Isolate::TryGetCurrent();
  return static_cast<void*>(cur);
}

GTExecSnap GTGreenThreadExecSnapshot() {
  GTExecSnap r{nullptr, 0, nullptr, nullptr, 0, nullptr, 0};
  vi::Isolate* cur = vi::Isolate::TryGetCurrent();
  r.cur_iso = static_cast<void*>(cur);
  if (cur == nullptr) return r;
  r.cur_green = cur->is_green_thread_mode() ? 1 : 0;
  vi::Isolate::PerIsolateThreadData* ptd = cur->CurrentPerIsolateThreadData();
  r.cur_ptd = static_cast<void*>(ptd);
  if (ptd != nullptr) {
    vi::Isolate* piso = ptd->isolate();
    r.ptd_iso = static_cast<void*>(piso);
    r.ptd_green = (piso != nullptr && piso->is_green_thread_mode()) ? 1 : 0;
  }
  r.tlt = static_cast<void*>(cur->thread_local_top());
  // "central" here means the executing thread is NOT running on its own green
  // per-worker block: either the current isolate is not green, or the per-thread
  // data is bound to a non-green isolate (thread_local_top() routes to central).
  r.central = (r.cur_green == 1 && r.ptd_green == 1 && r.cur_ptd != nullptr)
                  ? 0
                  : 1;
  return r;
}

// Env-gated entry-state dump. Runs only when GT_GT_DUMP is set; every other
// path is a single unconditional call, so it costs nothing in the shipped build
// beyond the getenv() compare. dumps the ThreadLocalTop entry-state a Maglev
// body reads on first JS entry, plus isolate_root() (the root-register base the
// allocation builtins read the young pointer from). The block address itself
// distinguishes the warmup (central) block from a fresh per-worker block.
void GTGreenThreadEntryDump(v8::Isolate* iso) {
  if (::getenv("GT_GT_DUMP") == nullptr) return;
  vi::Isolate* v_iso = reinterpret_cast<vi::Isolate*>(iso);
  if (v_iso == nullptr) return;
  vi::ThreadLocalTop* tlt = v_iso->thread_local_top();
  ::fprintf(stderr,
            "[TLDUMP] block=%p c_entry_fp=%p c_fn=%p ctx=%p topmost_ctx=%p "
            "exc=%p handler=%p js_entry_sp=%p central_sp=%p central_lim=%p "
            "sec_sp=%p sec_lim=%p is_central_flag=%d isolate_root=%p\n",
            (void*)tlt, (void*)tlt->c_entry_fp_, (void*)tlt->c_function_,
            (void*)tlt->context_.ptr(), (void*)tlt->topmost_script_having_context_.ptr(),
            (void*)tlt->exception_.ptr(), (void*)tlt->handler_,
            (void*)tlt->js_entry_sp_, (void*)tlt->central_stack_sp_,
            (void*)tlt->central_stack_limit_, (void*)tlt->secondary_stack_sp_,
            (void*)tlt->secondary_stack_limit_,
            (int)tlt->is_on_central_stack_flag_, (void*)v_iso->isolate_root());
  {
    const vi::LinearAllocationArea& lab =
        v_iso->isolate_data()->new_allocation_info();
    ::fprintf(stderr,
              "[GTYOUNG] lab.start=%p lab.top=%p lab.limit=%p "
              "isolate_data=%p\n",
              (void*)lab.start(), (void*)lab.top(), (void*)lab.limit(),
              (void*)v_iso->isolate_data());
  }
  ::fflush(stderr);
}


}  // namespace node

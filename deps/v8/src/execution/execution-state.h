// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// M2 execution-state ABI (candidate b, phase i: transport proof).
//
// Background / why this exists
// ---------------------------
// V8 reaches its per-thread execution state (context / exception / handler /
// js_entry_sp / c_entry_fp / pending_handler_*, ...) from GENERATED CODE by
// baking each field's offset relative to the kRootRegister (x26):
//
//     generated: kRootRegister + IsolateData::GetOffset(kContext)
//     runtime:   thread_local_top()->context_
//
// The two sides are only equivalent because they address ONE shared
// ThreadLocalTop (IsolateData::thread_local_top_). That is fine for a single
// mutator but fatal for N OS threads sharing one heap: a sibling worker running
// the same heap clobbers context_/exception_/handler_/js_entry_sp_ that another
// worker's generated code is addressing through its own baked offset.
//
// The fix (candidate b) installs a per-OS-thread ExecutionState block reached
// through a dedicated register (kExecutionStateRegister) instead of kRootRegister:
//
//     generated: kExecutionStateRegister + ExecutionState::FieldOffset(id)
//     runtime:   current_worker()->field_address(id)
//
// The block embeds the WHOLE ThreadLocalTop by value as its first member
// (&ExecutionState == &top), so the field reached is &top + offsetof-in-TLT.
// FieldOffset(id) is derived from the SAME IsolateData::GetOffset(id) that the
// shared code bakes, reduced by the holder's position within IsolateData:
//
//     FieldOffset(id) == GetOffset(id) - IsolateData::GetOffset(kThreadLocalTop)
//                     == offsetof(ThreadLocalTop, <field>)      (by construction)
//
// The static_asserts below pin that identity. Because both generated code and
// runtime derive FieldOffset from the single source of truth GetOffset(id), the
// two can never desync (which was the failure mode of a runtime-only redirect).
// HandleScopeData ownership (next/limit/level) lives in a separate per-worker
// block owned by HandleScopeImplementer and is wired in the handle-scope phase.

#ifndef V8_EXECUTION_EXECUTION_STATE_H_
#define V8_EXECUTION_EXECUTION_STATE_H_

#include "src/execution/isolate-data.h"
#include "src/execution/thread-local-top.h"

namespace v8::internal {

// Per-OS-thread execution state for a green-thread (multi-mutator) isolate.
//
// One ExecutionState lives per active green worker. The embedded ThreadLocalTop
// (member {top}, first so &ExecutionState == &top) carries the fields that used
// to live in the shared IsolateData::thread_local_top_; per-worker they no longer
// collide between simultaneously-running OS threads.
class ExecutionState {
 public:
  ExecutionState() { top.Clear(); }

  ThreadLocalTop top;

  // Owner / park metadata. Populated when a worker registers this block and
  // cleared on park/teardown. Not reached through baked offsets, so not part of
  // the address invariant.
  Isolate* owner_isolate_ = nullptr;
  int worker_index_ = -1;
  bool parked_ = false;

  // Position of the embedded ThreadLocalTop within IsolateData's baked layout,
  // in kRootRegister-relative terms. This is exactly the offset that must be
  // subtracted from IsolateData::GetOffset(id) to turn the shared, root-relative
  // field offset into this block's own top-relative field offset (FieldOffset).
  // It is the same value the shared code already bakes for the holder itself.
  static constexpr intptr_t ThreadLocalTopHolderOffset() {
    return IsolateData::GetOffset(IsolateFieldId::kThreadLocalTop);
  }

  // Baked, top-relative offset of an execution-state field within this block.
  // Derived from the one source of truth (IsolateData::GetOffset), reduced by
  // the holder's position, so runtime field_address() and generated code bake
  // the identical value and can never drift apart.
  static constexpr intptr_t FieldOffset(IsolateFieldId id) {
    return IsolateData::GetOffset(id) - ThreadLocalTopHolderOffset();
  }

  // Worker-relative address of an execution-state field within THIS block. The
  // ExecutionState register points at &top (== &ExecutionState), so this is
  //   &top + FieldOffset(id)
  // where FieldOffset(id) == offsetof(ThreadLocalTop, <field_>). Reuses the single
  // source of truth GetOffset(id), reduced by the holder offset, so it stays
  // numerically identical to what generated code bakes.
  Address field_address(IsolateFieldId id) const {
    DCHECK(is_execution_field(id));
    return reinterpret_cast<Address>(&top) + FieldOffset(id);
  }

  ThreadLocalTop* top_ptr() { return &top; }

  // True iff {id} is a BAKED execution-state field owned by the embedded
  // ThreadLocalTop. These are the only fields generated code addresses through
  // the execution-state register, so field_address() is valid only for exactly
  // this set (context / exception / topmost-script-having-context / c_entry_fp /
  // handler / c_function / js_entry_sp / is_on_central_stack_flag / the
  // pending-handler channel (context, entrypoint, constant pool, fp, sp) /
  // num_frames_above_pending_handler). Runtime-only ThreadLocalTop members that
  // are not baked (pending_message_, last_api_entry_) are NOT in this set; they
  // live in the same per-worker block and are reached via top_ptr() in phase ii.
  // HandleScopeData / StackGuard / NewAllocationInfo fields live in other
  // holders and are rejected here.
  bool is_execution_field(IsolateFieldId id) const {
    switch (id) {
      case IsolateFieldId::kContext:
      case IsolateFieldId::kException:
      case IsolateFieldId::kTopmostScriptHavingContext:
      case IsolateFieldId::kCEntryFP:
      case IsolateFieldId::kHandler:
      case IsolateFieldId::kCFunction:
      case IsolateFieldId::kJSEntrySP:
      case IsolateFieldId::kIsOnCentralStackFlag:
      case IsolateFieldId::kPendingHandlerContext:
      case IsolateFieldId::kPendingHandlerEntrypoint:
      case IsolateFieldId::kPendingHandlerConstantPool:
      case IsolateFieldId::kPendingHandlerFP:
      case IsolateFieldId::kPendingHandlerSP:
      case IsolateFieldId::kNumFramesAbovePendingHandler:
        return true;
      default:
        return false;
    }
  }
};

// The baked ThreadLocalTop field offsets, reduced to top-relative, must land on
// the same bytes as offsetof(ThreadLocalTop, <field_>). These are the identity
// the ABI depends on: any ThreadLocalTop field added/removed/reordered breaks a
// static_assert here and is caught at compile time, never by a silent desync.
static_assert(offsetof(ThreadLocalTop, context_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kContext));
static_assert(offsetof(ThreadLocalTop, exception_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kException));
static_assert(offsetof(ThreadLocalTop, topmost_script_having_context_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kTopmostScriptHavingContext));
static_assert(offsetof(ThreadLocalTop, c_entry_fp_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kCEntryFP));
static_assert(offsetof(ThreadLocalTop, handler_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kHandler));
static_assert(offsetof(ThreadLocalTop, js_entry_sp_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kJSEntrySP));
static_assert(offsetof(ThreadLocalTop, is_on_central_stack_flag_) ==
              ExecutionState::FieldOffset(IsolateFieldId::kIsOnCentralStackFlag));

}  // namespace v8::internal

#endif  // V8_EXECUTION_EXECUTION_STATE_H_

// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <sys/mman.h>

#include <cstdlib>

#include "src/execution/green-thread.h"

#include "src/base/platform/platform.h"
#include "src/execution/isolate.h"
#include "src/execution/isolate-inl.h"
#include "src/execution/green-thread-scheduler.h"

namespace v8 {
namespace internal {

// --- Current green thread accessors ---

namespace {
// Current green thread for the active green-thread context on this OS thread.
//
// The Node binding drives green threads recursively on a single embedding OS
// thread (the "Model A" scheduler), so a process-wide instance is correct
// here. A plain (non-TLS) variable is deliberate: with the `local-dynamic` TLS
// model, the same `thread_local` symbol resolved across force-linked V8 static
// archives aliases multiple distinct TLS blocks, so SetCurrentGreenThread()
// writes one copy while the write-barrier's CurrentGreenThread() read observes
// a different, always-null copy. Sharing one non-TLS instance guarantees the
// writer and every reader observe the same location.
GreenThread* g_current_green_thread_ = nullptr;
}  // namespace

// --- Class-level counters (atomic, thread-safe) ---

std::atomic<int64_t> GreenThread::total_created_{0};
std::atomic<int64_t> GreenThread::total_completed_{0};

// --- GreenThread implementation ---

GreenThread::GreenThread(Isolate* isolate)
    : isolate_(isolate),
      processor_(nullptr),
      state_(static_cast<uint8_t>(GreenThreadState::kRunnable)),
      pc_raw_(kNullAddress),
      frame_pointer_(kNullAddress),
      next_(nullptr),
      prev_(nullptr),
      blocked_next_(nullptr),
      blocked_prev_(nullptr),
      stack_(kNullAddress),
      stack_top_(kNullAddress),
      stack_limit_(kNullAddress),
      stack_ptr_(kNullAddress),
      flags_(0),
      iteration_count_(0),
      cow_ref_table_(nullptr),
      cow_enabled_(false),
      register_mask_(0),
      regs_{},
      awaited_promise_(kNullAddress),
      resolved_await_value_(kNullAddress),
      resume_cb_(nullptr),
      resume_data_(nullptr),
      handle_scope_(nullptr),
      has_result_(false),
      result_(kNullAddress) {
  total_created_++;
}

GreenThread::~GreenThread() {
  if (stack_ != kNullAddress) {
    FreeGreenThreadLocalStorage(this);
  }
  if (cow_ref_table_) {
    delete cow_ref_table_;
    cow_ref_table_ = nullptr;
  }
}

// --- Stack management ---

void GreenThread::InitStack(Address base, int size) {
  stack_ = base;
  // Stack grows downward: top is the highest address (start of mapping),
  // limit is the low end (guard page is below it).
  stack_top_ = base + size;
  stack_limit_ = base + kGreenThreadGuardSize;
  // Initialize stack pointer to the top (next allocation goes down).
  stack_ptr_ = stack_top_;
}

void GreenThread::FreeGreenThreadLocalStorage(GreenThread* gt) {
  if (gt && gt->stack_ != kNullAddress) {
    // Use munmap directly since OS::Free/Release are private.
    munmap(reinterpret_cast<void*>(gt->stack_),
           gt->stack_top_ - gt->stack_);
    gt->stack_ = kNullAddress;
    gt->stack_top_ = kNullAddress;
    gt->stack_limit_ = kNullAddress;
    gt->stack_ptr_ = kNullAddress;
  }
}

// --- Current green thread accessors ---

// Attribution is worker-local inside a green-thread isolate (invariant I1):
// each OS worker stores its current GT in its own PerIsolateThreadData, so
// several workers running concurrently in one green isolate never clobber a
// shared slot. Isolate::CurrentPerIsolateThreadData() is a proper thread_local
// (not the aliased plain symbol the process-global slot uses), so it is safe
// to call from the write barrier on any worker. For any other isolate we fall
// back to the process-wide g_current_green_thread_ slot, preserving the legacy
// single-OS-thread (Model A) behavior byte-for-byte.
GreenThread* CurrentGreenThread() {
  Isolate* iso = Isolate::TryGetCurrent();
  if (iso != nullptr && iso->is_green_thread_mode()) {
    Isolate::PerIsolateThreadData* pd = Isolate::CurrentPerIsolateThreadData();
    if (pd != nullptr) return pd->current_green_thread();
  }
  return g_current_green_thread_;
}

void SetCurrentGreenThread(GreenThread* thread) {
  Isolate* iso = Isolate::TryGetCurrent();
  if (iso != nullptr && iso->is_green_thread_mode()) {
    Isolate::PerIsolateThreadData* pd = Isolate::CurrentPerIsolateThreadData();
    if (pd != nullptr) {
      pd->set_current_green_thread(thread);
      return;
    }
  }
  g_current_green_thread_ = thread;
}

}  // namespace internal
}  // namespace v8

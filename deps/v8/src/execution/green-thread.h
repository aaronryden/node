// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_EXECUTION_GREEN_THREAD_H_
#define V8_EXECUTION_GREEN_THREAD_H_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <setjmp.h>
#include <unordered_set>

#include "include/v8config.h"
#include "src/base/platform/platform.h"
#include "src/common/globals.h"
#include "src/execution/heap-ref-table.h"

namespace v8 {

namespace internal {

class GreenProcessor;
class HeapRefTable;
class Isolate;

// Handle-scope phase routing gate. The embedder calls this to opt a green-thread
// isolate into per-worker handle-cursor routing while its N OS workers run bodies
// concurrently (RunTasks pool.Run / M2Probe worker pool). Declared here against
// the forward-declared Isolate so green_threads.cc can toggle the flag WITHOUT
// including src/execution/isolate.h (which pulls heap.h -> trace-event.h and
// collides with node's tracing/trace_event.h).
void SetIsolateHandleScopePerWorker(Isolate* iso, bool on);

// --- Constants ---

// Runqueue capacity for GreenProcessor.
constexpr int kGreenQueueCapacity = 256;

// Register spill buffer size for green thread state (bytes).
// 16 registers x 8 bytes per register on 64-bit.
// Register spill area: general purpose, NEON/VFP, and control registers.
constexpr int kGreenRegsSize = 696;  // GPR 168 + pad 8 + NEON 512 + FPSR 4 + FPCR 4

// Guard page size for stack overflow detection.
constexpr int kGreenThreadGuardSize = 4 * KB;

// Green thread stack size in bytes (16KB initial for v1).
constexpr int kGreenThreadStackInitial = 16 * KB;

// Register mask bits.
enum : uint32_t {
  kGreenRegMaskRoot = 1u << 0,
  kGreenRegMaskCage = 1u << 1,
  kGreenRegMaskBytecode = 1u << 2,
  kGreenRegMaskDispatch = 1u << 3
};

// GreenThread states.
enum class GreenThreadState : uint8_t {
  kRunnable,  // In a P's runqueue, waiting to run
  kRunning,   // Currently executing on an OS thread
  kBlocked,   // Suspended on an external event (I/O); OS thread freed for other
  kComplete   // Finished execution
};

// GreenThread - a user-level green thread (a "G" in Go terminology).
//
// Each green thread has its own small stack, saved program counter, and
// register state. The descriptor is kept to one cache line for minimal
// scheduling overhead.
class GreenThread {
 public:
  // Iteration counter.
  uint32_t iteration_count() const { return iteration_count_; }
  void increment_iteration_count() { iteration_count_++; }
  GreenThread(Isolate* isolate);
  ~GreenThread();

  // --- State management ---

  Isolate* isolate() const { return isolate_; }
  GreenThreadState state() const {
    return static_cast<GreenThreadState>(state_.load(std::memory_order_acquire));
  }
  void set_state(GreenThreadState s) {
    state_.store(static_cast<uint8_t>(s), std::memory_order_release);
  }

  // --- Processor ---

  GreenProcessor* processor() const { return processor_; }
  void set_processor(GreenProcessor* p) { processor_ = p; }

  // --- Execution state ---

  Address pc_raw() const { return pc_raw_; }
  void set_pc_raw(Address p) { pc_raw_ = p; }

  Address frame_pointer() const { return frame_pointer_; }
  void set_frame_pointer(Address fp) { frame_pointer_ = fp; }

  // --- Stack management ---

  Address stack() const { return stack_; }
  void set_stack(Address s) { stack_ = s; }

  Address stack_top() const { return stack_top_; }
  void set_stack_top(Address s) { stack_top_ = s; }

  Address stack_limit() const { return stack_limit_; }
  void set_stack_limit(Address s) { stack_limit_ = s; }

  Address stack_ptr() const { return stack_ptr_; }
  void set_stack_ptr(Address sp) { stack_ptr_ = sp; }

  void InitStack(Address base, int size);
  static void FreeGreenThreadLocalStorage(GreenThread* gt);

  // Check if the stack pointer has overflowed past the guard page.
  bool HasStackOverflowed() const { return stack_ptr_ < stack_limit_; }

  // --- Scheduler links ---

  // Doubly-linked list for live_threads_ (managed by GreenScheduler).
  GreenThread* next() const { return next_; }
  void set_next(GreenThread* n) { next_ = n; }
  GreenThread* prev() const { return prev_; }
  void set_prev(GreenThread* p) { prev_ = p; }

  // Separate links for blocked list (prevents queue ops from severing GC list).
  GreenThread* blocked_next() const { return blocked_next_; }
  void set_blocked_next(GreenThread* n) { blocked_next_ = n; }
  GreenThread* blocked_prev() const { return blocked_prev_; }
  void set_blocked_prev(GreenThread* p) { blocked_prev_ = p; }

  // --- Flags ---

  uint32_t flags() const { return flags_; }
  void set_flags(uint32_t f) { flags_ = f; }

  // --- COW state ---

  HeapRefTable* cow_ref_table() const { return cow_ref_table_; }
  void set_cow_ref_table(HeapRefTable* t) { cow_ref_table_ = t; }
  bool cow_enabled() const { return cow_enabled_; }
  void set_cow_enabled(bool e) { cow_enabled_ = e; }

  // Green thread dirty set: objects that the GT has written to since the
  // last flush. The GT's stack and registers hold stale references to these
  // objects (the main-heap pointers). After cloning, the dirty set is flushed
  // to redirect all such references to the GT-local copies.
  void add_cow_dirty_object(Address obj) {
    cow_dirty_objects_.insert(obj);
    cow_dirty_flag_ = true;
  }
  void clear_cow_dirty_objects() { cow_dirty_objects_.clear(); }
  std::unordered_set<Address>& cow_dirty_objects_ref() { return cow_dirty_objects_; }
  bool cow_dirty_flag() const { return cow_dirty_flag_; }
  void set_cow_dirty_flag(bool f) { cow_dirty_flag_ = f; }

  // --- Register mask + saved register buffer ---

  uint32_t register_mask() const { return register_mask_; }
  void set_register_mask(uint32_t m) { register_mask_ = m; }
  uint8_t* regs() const { return const_cast<uint8_t*>(regs_); }


  // --- Await state (Phase 5) ---

  Address awaited_promise() const { return awaited_promise_; }
  void set_awaited_promise(Address p) { awaited_promise_ = p; }

  Address resolved_await_value() const { return resolved_await_value_; }
  void set_resolved_await_value(Address v) { resolved_await_value_ = v; }

  // --- Resume callback (for blocked thread wakeup) ---

  using ResumeCallback = void (*)(Isolate* isolate, void* thread, void* data);
  ResumeCallback resume_cb() const { return resume_cb_; }
  void set_resume_cb(ResumeCallback cb) { resume_cb_ = cb; }
  void* resume_data() const { return resume_data_; }
  void set_resume_data(void* d) { resume_data_ = d; }

  // --- I/O-reactivation flag (blocked-list bookkeeping) ---
  // Set under GreenScheduler::react_mutex_ by ReactivateGreenThread (which may
  // run on a libuv pool thread) and cleared under the same lock by the owning
  // worker's DrainReactivations: a reactivation is a one-shot signal, so the
  // flag dedups repeated I/O completions for a body still parked and prevents
  // the same gt being enqueued twice into the runnable queue.
  bool react_pending() const { return react_pending_; }
  void set_react_pending(bool r) { react_pending_ = r; }

  // --- GC handshake (invariant I3) ---
  //
  // A stop-the-world GC is driven by a coordinator that release-stores
  // gc_requested and then spin-joins every active green thread until it has
  // published its roots and parked here. This is the mutator-safepoint the
  // handshake relies on: a worker may claim to be quiescent only after its
  // registers, stack, and HeapRefTable have been published. The worker blocks
  // on gc_cv_ (under gc_mutex_) until the coordinator clears gc_requested and
  // notifies it on resume. These members are only touched while gc_requested is
  // set (env-gated forced-GC test path / future true concurrency); the default
  // serial path never parks.
  std::mutex gc_mutex_;
  std::condition_variable gc_cv_;
  std::atomic<bool> gc_parked_{false};
  bool gc_parked() const { return gc_parked_.load(std::memory_order_acquire); }
  void set_gc_parked(bool v) { gc_parked_.store(v, std::memory_order_release); }
  std::mutex& gc_mutex() { return gc_mutex_; }
  std::condition_variable& gc_cv() { return gc_cv_; }

  // --- Class-level counters ---

  static int64_t TotalCreated() { return total_created_.load(std::memory_order_relaxed); }
  static int64_t TotalCompleted() { return total_completed_.load(std::memory_order_relaxed); }

  // Delete copy/assign.
  GreenThread(const GreenThread&) = delete;
  GreenThread& operator=(const GreenThread&) = delete;

 private:
  friend class GreenScheduler;
  friend class GreenProcessor;
  friend class HandleScope;
  friend void RunGreenThread(GreenThread* gt);

  // Execution state.
  Isolate* isolate_;
  GreenProcessor* processor_;
  std::atomic<uint8_t> state_;
  Address pc_raw_;
  Address frame_pointer_;

  // Scheduler links (live list).
  GreenThread* next_;
  GreenThread* prev_;

  // Scheduler links (blocked list, separate from live list).
  GreenThread* blocked_next_;
  GreenThread* blocked_prev_;

  // Stack management.
  Address stack_;
  Address stack_top_;
  Address stack_limit_;
  Address stack_ptr_;

  // Flags.
  uint32_t flags_ = 0;

  // Iteration counter: incremented each time RunGreenThread is called.
  // Used by the entry function to track which phase to execute.
  uint32_t iteration_count_ = 0;

  // COW state.
  HeapRefTable* cow_ref_table_;
  bool cow_enabled_;

  // Dirty set: objects that have been written to since the last flush.
  // Each entry is the original (main-heap) address that was written to.
  // The GT's stack/registers hold stale references to these objects.
  // A bool flag signals whether the set is non-empty.
  std::unordered_set<Address> cow_dirty_objects_;
  bool cow_dirty_flag_;

  // Register mask + saved buffer (for context switch).
  uint32_t register_mask_;
  uint8_t regs_[kGreenRegsSize];

  // Await state.
  Address awaited_promise_;
  Address resolved_await_value_;

  // Resume callback for blocked thread wakeup.
  ResumeCallback resume_cb_;
  void* resume_data_;

  // I/O-reactivation flag (guarded by GreenScheduler::react_mutex_). See the
  // accessor comment above.
  bool react_pending_ = false;

  // Handle scope (Phase 4 - for V8 API interactions).
  void* handle_scope_;

  // Result storage (for completed threads with return values).
  bool has_result_;
  Address result_;

  // --- Class-level counters (atomic, thread-safe) ---
  static std::atomic<int64_t> total_created_;
  static std::atomic<int64_t> total_completed_;
};

// --- Thread-local accessors ---

// Get the current green thread active on this OS thread, or nullptr.
V8_EXPORT_PRIVATE GreenThread* CurrentGreenThread();

// Set the current green thread for this OS thread.
// Called by the scheduler when switching between green threads.
V8_EXPORT_PRIVATE void SetCurrentGreenThread(GreenThread* thread);

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_GREEN_THREAD_H_

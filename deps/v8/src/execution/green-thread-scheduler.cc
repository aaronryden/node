// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread-scheduler.h"

#include <sys/mman.h>

#include <cstdlib>
#include <deque>
#include <pthread.h>
#include <vector>

#include "src/execution/green-processor.h"
#include "src/execution/isolate.h"
#include "src/execution/green-thread-cow.h"

// Declare ARM64 register save/restore functions from green-thread-asm.cc.
extern "C" void green_thread_save_registers_arm64(uint8_t* dst);
extern "C" void green_thread_restore_registers_arm64(const uint8_t* src);

namespace v8 {
namespace internal {

// Thread-local processor mapping.
thread_local GreenProcessor* GreenScheduler::g_processor_ = nullptr;

// Page size constant for stack allocation (ARM64 default on macOS: 16KB).
// We use 4KB which is the minimum supported on most platforms.
constexpr int kGreenThreadPageSize = 4096;

GreenScheduler::GreenScheduler(Isolate* isolate)
    : isolate_(isolate),  // Reserved for future scheduler features
      num_processors_(0),
      global_queue_head_(nullptr),
      global_queue_tail_(nullptr),
      blocked_list_head_(nullptr),
      stopping_(false),
      total_threads_created_(0),
      total_threads_completed_(0) {
  for (int i = 0; i < kMaxGreenProcessors; i++) {
    processors_[i] = nullptr;
  }
  // Suppress unused field warning - isolate is used in future scheduler hooks.
  (void)isolate_;
}

GreenScheduler::~GreenScheduler() {
  Shutdown();
}

void GreenScheduler::Initialize(int num_processors) {
  num_processors_ = num_processors;
  if (num_processors_ > kMaxGreenProcessors) {
    num_processors_ = kMaxGreenProcessors;
  }

  for (int i = 0; i < num_processors_; i++) {
    processors_[i] = new GreenProcessor(this, i);
    processors_[i]->set_active(true);
  }
}

void GreenScheduler::Shutdown() {
  for (int i = 0; i < num_processors_; i++) {
    if (processors_[i] != nullptr) {
      delete processors_[i];
      processors_[i] = nullptr;
    }
  }
  num_processors_ = 0;

  // Clear global queue.
  global_queue_head_ = nullptr;
  global_queue_tail_ = nullptr;

  // Clear the reactivation queue and wake a worker that may be parked waiting
  // for a reactivation, so teardown / destruction never hangs.
  {
    std::lock_guard<std::mutex> lk(react_mutex_);
    stopping_ = true;
    react_queue_.clear();
  }
  react_cv_.notify_all();

  blocked_list_head_ = nullptr;

  // Clear thread-local processor.
  g_processor_ = nullptr;
}

void GreenScheduler::ClearProcessorQueue() {
  GreenProcessor* p = GetOrCreateProcessor();
  if (p != nullptr) {
    p->clear_queue();
  }
}

GreenProcessor* GreenScheduler::GetOrCreateProcessor() {
  if (g_processor_ != nullptr) {
    return g_processor_;
  }

  // Create a new processor if we have room.
  for (int i = 0; i < num_processors_ && i < kMaxGreenProcessors; i++) {
    if (processors_[i] != nullptr) {
      g_processor_ = processors_[i];
      return g_processor_;
    }
  }

  // Fall back to first processor.
  if (num_processors_ > 0 && processors_[0] != nullptr) {
    g_processor_ = processors_[0];
    return g_processor_;
  }

  return nullptr;
}

GreenThread* GreenScheduler::CreateThread(Isolate* isolate, void* entry, void* arg) {
  // Create the green thread descriptor.
  GreenThread* gt = new GreenThread(isolate);

  // Allocate stack memory using mmap.
  // Layout: [guard page][usable stack]
  int alloc_size = kGreenThreadStackInitial + kGreenThreadPageSize;
  void* stack_base = mmap(nullptr, alloc_size,
                          PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

  if (stack_base != MAP_FAILED) {
    // Protect the guard page
    mprotect(stack_base, kGreenThreadPageSize, PROT_NONE);
    // Stack starts after the guard page
    Address usable_base = reinterpret_cast<Address>(stack_base) + kGreenThreadPageSize;
    gt->set_stack(reinterpret_cast<Address>(stack_base));
    gt->InitStack(usable_base, kGreenThreadStackInitial);
  }

  // Set up the entry point and argument.
  gt->set_pc_raw(reinterpret_cast<Address>(entry));
  gt->set_flags(arg ? kGreenRegMaskBytecode : 0);

  // Enable COW for the green thread.
  gt->set_cow_enabled(true);
  gt->set_cow_ref_table(new HeapRefTable());

  total_threads_created_++;

  return gt;
}

void GreenScheduler::Schedule(GreenThread* gt) {
  GreenProcessor* p = GetOrCreateProcessor();
  p->Enqueue(gt);

  // If no thread is currently running, switch to this one.
  if (p->current_thread() == nullptr) {
    p->set_current_thread(gt);
    SetCurrentGreenThread(gt);
  } else {
    // If a thread is already running, move the new thread to the
    // front of the processor queue so it gets picked up on yield.
    // Remove from tail (where it was just added) and insert at head.
    if (gt->prev() != nullptr) {
      gt->prev()->set_next(gt->next());
    } else {
      p->queue_head_ = gt->next();
    }
    if (gt->next() != nullptr) {
      gt->next()->set_prev(gt->prev());
    } else {
      p->queue_tail_ = gt->prev();
    }
    // Insert at head.
    gt->set_next(p->queue_head_);
    gt->set_prev(nullptr);
    if (p->queue_head_ != nullptr) {
      p->queue_head_->set_prev(gt);
    } else {
      p->queue_tail_ = gt;
    }
    p->queue_head_ = gt;
  }
}

// Yield execution to the next green thread.
//
// This function saves the current thread's state, re-enqueues it to the
// global queue, picks the next thread from the global queue, increments
// its iteration count, and calls its entry function directly.
//
// The entry function runs on the current thread's stack (recursive call).
// When the next thread yields, it calls Yield() again, which picks another
// thread. Eventually, a thread completes (reaches Phase 2), returns, and
// Yield() returns to the calling thread's entry function.
//
// The entry function must check iteration_count to determine which phase
// to run, and must return immediately after Yield() returns (don't
// continue to Phase 2 after yielding from Phase 1).
//
// @sideEffects Modifies global queue, thread iteration counts, and
//              runs the next thread's entry function on the current stack.
void GreenScheduler::Yield() {
  GreenThread* current = CurrentGreenThread();
  if (current == nullptr) return;

  // Item 1 (invariant I3): before publishing this GT's registers, poll for a
  // forced stop-the-world GC request. If one is in progress, this publishes the
  // GT's roots and parks the worker here (returning true) so the collector sees
  // only quiescent mutators. This is the single existing green-thread safepoint
  // where a worker regains control between native calls; reusing it (rather than
  // inventing a new bytecode probe) means every worker that yields stops for GC.
  if (GreenThreadCow::PollForGCRequest(current, isolate_)) {
    return;  // Parked: the coordinator resumes this worker later.
  }

  GreenProcessor* p = current->processor();
  if (p != nullptr) {
    // Save the green thread's general-purpose registers into its spill
    // buffer before we switch away. This is used by GC root visiting
    // and COW reference tracking.
    green_thread_save_registers_arm64(current->regs());
  }
  current->set_register_mask(kGreenRegMaskRoot | kGreenRegMaskCage |
                             kGreenRegMaskBytecode | kGreenRegMaskDispatch);

  // Re-enqueue the yielding thread into the global queue.
  current->set_state(GreenThreadState::kRunnable);
  EnqueueGlobal(current);

  // Pick the next thread from the global queue.
  GreenThread* next = DequeueGlobal();
  if (next == nullptr) return;

  // Flush any pending COW writes before switching.
  GreenThreadCow::FlushGreenThreadDirtySet(current);

  // Increment the next thread's iteration count so its entry function
  // knows which phase to run. This replaces the traditional
  // RunGreenThread increment, since entry() is called directly here.
  next->increment_iteration_count();
  next->set_register_mask(0);
  SetCurrentGreenThread(next);

  // Call the next thread's entry function directly.
  // This is the concurrency mechanism: current's entry is paused,
  // next runs to completion or yields.
  typedef void (*EntryFunc)(Isolate*, void*);
  EntryFunc entry = reinterpret_cast<EntryFunc>(next->pc_raw());
  entry(current->isolate(), next);
  // When this returns, the next thread has either completed or yielded.
  // If it yielded, another thread was picked and ran recursively.
}

// Main scheduling loop — runs until all green threads are complete.
// For each iteration:
//   1. If no current thread, dequeue one from the global queue.
//   2. If the current thread is complete, skip and pick a new one.
//   3. Otherwise, run the green thread until it yields or completes.
void GreenScheduler::ScheduleLoop() {
  while (true) {
    GreenThread* current = CurrentGreenThread();
    if (current == nullptr) {
      // Look for a runnable thread in the global queue.
      GreenThread* next = DequeueGlobal();
      if (next == nullptr) {
        // No more threads to run.
#ifdef V8_ENABLE_GREEN_THREADS
        if (::getenv("GT_CRASH_DEBUG"))
          ::fprintf(stderr, "[GTDBG-SCHED-EXIT] worker scheduler empty, exiting\n");
#endif
        break;
      }

      GreenProcessor* p = GetOrCreateProcessor();
      p->set_current_thread(next);
      SetCurrentGreenThread(next);
      current = next;
    }

    // Execute the green thread via RunGreenThread.
    // If the thread is complete, move on.
    if (current->state() == GreenThreadState::kComplete) {
      total_threads_completed_++;
      // Clear the current thread so we pick the next one.
      SetCurrentGreenThread(nullptr);
      continue;
    }

#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_CRASH_DEBUG"))
      ::fprintf(stderr, "[GTDBG-SCHED] worker running gt=%p iter_before=%u\n",
                (void*)current, (unsigned)current->iteration_count());
#endif
    // Run the green thread (this calls entry() directly; if it yields,
    // Yield() recursively calls entry(next) and returns when done).
    RunGreenThread(current);

    // If we get here, the green thread completed.
    if (current->state() == GreenThreadState::kComplete) {
      total_threads_completed_++;
    }

    // Clear current so the next iteration picks a new thread.
    SetCurrentGreenThread(nullptr);
  }

  // The embedding runner frees every green thread once this drain returns, so
  // detach from the last running thread here while it is still allocated.
  // Otherwise GreenProcessor::current_thread_ would dangle: when a later spawn
  // reuses this persistent scheduler, GreenProcessor::set_current_thread would
  // dereference the freed thread in its "detach from old" step (a
  // heap-use-after-free). Detaching now is safe because the last thread has
  // only completed scheduling, not been destroyed.
  if (GreenProcessor* p = GetOrCreateProcessor()) {
    p->set_current_thread(nullptr);
  }
}

Address GreenScheduler::GetCowViewAddress(Address original_addr) {
  GreenThread* current = CurrentGreenThread();
  if (current == nullptr || !current->cow_enabled()) {
    return original_addr;
  }

  return current->cow_ref_table()->Lookup(original_addr);
}

void GreenScheduler::ApplyCow(GreenThread* gt, Address obj_addr) {
  if (!gt->cow_enabled()) return;

  HeapRefTable* ref_table = gt->cow_ref_table();
  if (ref_table == nullptr) return;

  // Check if this object is already tracked.
  Address view_addr = ref_table->GetView(obj_addr);
  if (view_addr != kNullAddress) {
    // Already cloned, just update the reference.
    return;
  }

  // For v1: mark the object as needing COW clone.
  // In a full implementation, this would deep-clone the object into the
  // green thread's heap view.
}

void GreenScheduler::EnqueueGlobal(GreenThread* gt) {
  gt->set_next(nullptr);
  gt->set_prev(global_queue_tail_);

  if (global_queue_tail_ != nullptr) {
    global_queue_tail_->set_next(gt);
  } else {
    global_queue_head_ = gt;
  }
  global_queue_tail_ = gt;
}

GreenThread* GreenScheduler::DequeueGlobal() {
  if (global_queue_head_ == nullptr) {
    return nullptr;
  }

  GreenThread* gt = global_queue_head_;
  global_queue_head_ = gt->next();

  if (global_queue_head_ != nullptr) {
    global_queue_head_->set_prev(nullptr);
  } else {
    global_queue_tail_ = nullptr;
  }

  gt->set_next(nullptr);
  gt->set_prev(nullptr);
  gt->set_state(GreenThreadState::kRunnable);

  return gt;
}

// --- Cooperative I/O block / reactivate ---
//
// Let a green body park on an external event (I/O) and free its OS worker for
// other bodies. BlockGreenThread runs on the body's owning worker thread;
// ReactivateGreenThread may run on a libuv pool thread, so the only cross-thread
// hand-off is the mutex-protected react_queue_ + react_cv_. The blocked list and
// the global runnable queue stay worker-only, so no two threads mutate them.

void GreenScheduler::BlockGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> lk(react_mutex_);
  // Stamp the body's architectural registers so a GC root visit during the
  // parked window sees its live state (mirrors the register save in Yield).
  // No-op for a green-async descriptor (processor()==0): it holds no arch stack,
  // and the drive holds AlwaysAllocate so no collection parks against it.
  GreenProcessor* p = gt->processor();
  if (p != nullptr) {
    green_thread_save_registers_arm64(gt->regs());
  }
  gt->set_register_mask(kGreenRegMaskRoot | kGreenRegMaskCage |
                        kGreenRegMaskBytecode | kGreenRegMaskDispatch);
  // Reset the reactivation gate so the next I/O completion can wake this body
  // again. Guarded here too (not just by the owning worker) because a single
  // shared scheduler serves every drive worker, so Block can run on any of them.
  gt->set_react_pending(false);
  gt->set_state(GreenThreadState::kBlocked);
  // Head-insert into the (now mutex-guarded) blocked list.
  gt->set_blocked_next(blocked_list_head_);
  gt->set_blocked_prev(nullptr);
  if (blocked_list_head_ != nullptr) {
    blocked_list_head_->set_blocked_prev(gt);
  }
  blocked_list_head_ = gt;
}

void GreenScheduler::ReactivateGreenThread(GreenThread* gt) {
  std::lock_guard<std::mutex> lk(react_mutex_);
  // Skip if the body was already scheduled for reactivation, or if the scheduler
  // is tearing down -- a completion racing Shutdown must not resurrect a worker.
  if (gt->react_pending() || stopping_) return;
  gt->set_react_pending(true);
  react_queue_.push_back(gt);
  react_cv_.notify_all();
}

// Move every reactivated body from the blocked list into the global runnable
// queue. Worker thread only. Returns true if at least one body was reactivated.
bool GreenScheduler::DrainReactivations() {
  std::vector<GreenThread*> reactivated;
  {
    std::lock_guard<std::mutex> lk(react_mutex_);
    while (!react_queue_.empty()) {
      GreenThread* r = react_queue_.front();
      react_queue_.pop_front();
      r->set_react_pending(false);  // gate so Reactivate cannot re-add it
      reactivated.push_back(r);
    }
  }
  for (GreenThread* r : reactivated) {
    // Unlink from the (mutex-guarded) blocked list. Track the previous node
    // explicitly (blocked_next() returns by value, so we cannot take the
    // address of the link slot); set_blocked_next is the muting accessor.
    GreenThread* prev = nullptr;
    GreenThread* node = blocked_list_head_;
    while (node != nullptr && node != r) {
      prev = node;
      node = node->blocked_next();
    }
    if (node == r) {
      GreenThread* nx = r->blocked_next();
      if (prev == nullptr) blocked_list_head_ = nx;
      else prev->set_blocked_next(nx);
      if (nx != nullptr) nx->set_blocked_prev(prev);
    }
    r->set_blocked_next(nullptr);
    r->set_blocked_prev(nullptr);
    r->set_state(GreenThreadState::kRunnable);
    EnqueueGlobal(r);
  }
  return !reactivated.empty();
}

// Park the worker until `pred` becomes true (an I/O this worker OWNS has
// completed) or the scheduler is stopping. The worker does not spin here -- that
// is what frees the core while its bodies are blocked on I/O.
//
// We wait on a per-worker predicate rather than the shared react_queue_: under a
// single drive scheduler, ReactivateGreenThread's notify_all wakes EVERY parked
// worker, and DrainReactivations (run at the top of each worker's loop) drains the
// shared queue, so another worker can pop-and-clear THIS worker's reactivation
// before its owner wakes. The owner then re-parks on an empty queue, while its own
// body still has a completed read it can no longer be signalled to drain -- a
// deadlock. A predicate that checks THIS worker's own finished reads is steal-proof:
// the completion's effect is always visible to the owner regardless of who drained
// the shared queue. Evaluated under react_mutex_ (as all cv.wait predicates are),
// it must touch only the worker's own state -- no other lock, no cross-thread
// queue. Reactivate still supplies the notify_all that wakes the parked worker to
// re-check; the predicate decides whether it actually returns.
void GreenScheduler::WaitForReactivationWithPred(const std::function<bool()>& pred) {
  std::unique_lock<std::mutex> lk(react_mutex_);
  react_cv_.wait(lk, [&] { return stopping_ || pred(); });
}

int64_t GreenScheduler::blocked_count() const {
  int64_t count = 0;
  for (GreenThread* t = blocked_list_head_; t != nullptr; t = t->blocked_next()) {
    count++;
  }
  return count;
}

}  // namespace internal
}  // namespace v8

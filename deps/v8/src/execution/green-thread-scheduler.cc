// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread-scheduler.h"

#include <sys/mman.h>

#include <cstdlib>
#include <pthread.h>

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

}  // namespace internal
}  // namespace v8

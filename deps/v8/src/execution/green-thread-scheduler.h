// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_EXECUTION_GREEN_THREAD_SCHEDULER_H_
#define V8_EXECUTION_GREEN_THREAD_SCHEDULER_H_

#include <atomic>
#include <cstdint>

#include "src/execution/green-processor.h"
#include "src/execution/green-thread.h"

namespace v8 {

namespace internal {

class GreenProcessor;
class GreenThread;
class Isolate;

// GreenScheduler — manages the global runqueue and processor pool.
//
// The scheduler handles:
// 1. Creating and destroying processors (P's)
// 2. Assigning green threads to processors
// 3. Global runqueue for load balancing
// 4. Context switching between green threads
// 5. COW heap view management
//
// Model A execution model (single-threaded):
// All context switches happen on the embedding thread. The scheduler
// provides the glue between V8's execution and green thread scheduling.
//
// Important invariants for Model A:
// - Global queue (global_queue_head_/tail_) is accessed only from the
//   embedding thread, so no mutex is needed. EnqueueGlobal/DequeueGlobal
//   operate on a singly-linked list without synchronization.
// - TLS (g_current_green_thread_) is preserved across setjmp/longjmp
//   because those do not switch OS threads. The scheduler sets TLS
//   before each longjmp, and the green thread's code sees the correct
//   value via CurrentGreenThread().

class GreenScheduler {
 public:
  explicit GreenScheduler(Isolate* isolate);
  ~GreenScheduler();

  // Delete copy/assign.
  GreenScheduler(const GreenScheduler&) = delete;
  GreenScheduler& operator=(const GreenScheduler&) = delete;

  // --- Initialization ---

  // Initialize the scheduler with a given number of processors.
  void Initialize(int num_processors = 1);

  // Shutdown the scheduler (cleanup all processors and threads).
  void Shutdown();

  // Clear the processor's runqueue without deleting threads.
  // Call this before manually deleting threads to avoid double-free.
  void ClearProcessorQueue();

  // --- Processor management ---

  int num_processors() const { return num_processors_; }
  GreenProcessor* processor(int id) const {
    if (id >= 0 && id < num_processors_) {
      return processors_[id];
    }
    return nullptr;
  }

  // Get a processor for the current OS thread.
  // Creates one if it doesn't exist.
  GreenProcessor* GetOrCreateProcessor();

  // --- Thread management ---

  // Create a new green thread.
  GreenThread* CreateThread(Isolate* isolate, void* entry, void* arg);

  // Schedule a green thread to run (assign to a processor and start).
  void Schedule(GreenThread* gt);

  // Yield the current green thread (switch to the next one).
  void Yield();

  // Run the scheduler loop — processes green threads until none are left.
  void ScheduleLoop();

  // --- COW support ---

  // Get the active COW heap view for the current green thread.
  Address GetCowViewAddress(Address original_addr);

  // Apply COW: clone the object if needed and update references.
  void ApplyCow(GreenThread* gt, Address obj_addr);

  // --- Statistics ---

  // Enqueue a thread into the global runqueue (alternative to Schedule).
  // Used when threads should be balanced across processors.
  void EnqueueGlobal(GreenThread* gt);

  // Dequeue a thread from the global runqueue.
  GreenThread* DequeueGlobal();

  int64_t total_threads_created() const {
    return total_threads_created_.load();
  }
  int64_t total_threads_completed() const {
    return total_threads_completed_.load();
  }

 private:
  Isolate* isolate_;
  int num_processors_;
  GreenProcessor* processors_[kMaxGreenProcessors];

  // Global runqueue for load balancing.
  GreenThread* global_queue_head_;
  GreenThread* global_queue_tail_;

  // Thread-local processor mapping (OS thread -> GreenProcessor*).
  static thread_local GreenProcessor* g_processor_;

  // Statistics.
  std::atomic<int64_t> total_threads_created_;
  std::atomic<int64_t> total_threads_completed_;
};

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_GREEN_THREAD_SCHEDULER_H_

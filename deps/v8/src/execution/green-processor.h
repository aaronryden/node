// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_EXECUTION_GREEN_PROCESSOR_H_
#define V8_EXECUTION_GREEN_PROCESSOR_H_

#include <atomic>
#include <cstdint>

#include "src/execution/green-thread.h"

namespace v8 {

namespace internal {

class GreenScheduler;
class Isolate;

// Maximum number of processors (P's) per green scheduler.
constexpr int kMaxGreenProcessors = 64;

// GreenProcessor — maps green threads to OS threads (the "P" in Go terminology).
//
// Each processor has a local runqueue of green threads ready to run.
// When the current green thread yields, the processor picks the next one
// from the queue and performs a context switch.
//
// In Model A execution (all GT switches on embedding thread only), the
// processor's stack_switch_fn is called on the embedding thread to perform
// the actual stack switch.
class GreenProcessor {
 public:
  explicit GreenProcessor(GreenScheduler* scheduler, int id = 0);
  ~GreenProcessor();

  // Delete copy/assign.
  GreenProcessor(const GreenProcessor&) = delete;
  GreenProcessor& operator=(const GreenProcessor&) = delete;

  // --- Processor state ---

  GreenScheduler* scheduler() const { return scheduler_; }
  int id() const { return id_; }
  bool is_active() const { return is_active_; }
  void set_active(bool active) { is_active_ = active; }

  // The green thread currently executing on this processor.
  GreenThread* current_thread() const { return current_thread_; }
  void set_current_thread(GreenThread* gt);

  // --- Runqueue ---

  // Push a green thread onto the processor's local runqueue.
  void Enqueue(GreenThread* gt);

  // Pop a green thread from the processor's local runqueue.
  // Returns nullptr if the queue is empty.
  GreenThread* Dequeue();

  // Check if the runqueue is empty.
  bool is_queue_empty() const { return queue_head_ == nullptr; }

  // Clear the runqueue without deleting threads (threads are externally managed).
  void clear_queue() {
    queue_head_ = nullptr;
    queue_tail_ = nullptr;
  }

  // Get the number of green threads in the runqueue.
  int queue_length() const;

  // --- Statistics ---

  int64_t total_switches() const { return total_switches_.load(); }
  void increment_switches() { total_switches_++; }

 private:
  friend class GreenScheduler;

  GreenScheduler* scheduler_;
  int id_;

  // Flag to track if this processor is active.
  bool is_active_;

  // The green thread currently executing on this processor.
  GreenThread* current_thread_;

  // Local runqueue (linked list).
  GreenThread* queue_head_;
  GreenThread* queue_tail_;

  // Statistics.
  std::atomic<int64_t> total_switches_;
};

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_GREEN_PROCESSOR_H_

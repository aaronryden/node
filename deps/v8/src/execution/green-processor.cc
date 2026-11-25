// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-processor.h"

namespace v8 {
namespace internal {

GreenProcessor::GreenProcessor(GreenScheduler* scheduler, int id)
    : scheduler_(scheduler),
      id_(id),
      is_active_(false),
      current_thread_(nullptr),
      queue_head_(nullptr),
      queue_tail_(nullptr),
      total_switches_(0) {}

GreenProcessor::~GreenProcessor() {
  // Delete any remaining threads in the queue.
  GreenThread* gt = queue_head_;
  while (gt != nullptr) {
    GreenThread* next = gt->next();
    delete gt;
    gt = next;
  }
  queue_head_ = nullptr;
  queue_tail_ = nullptr;
}

void GreenProcessor::set_current_thread(GreenThread* gt) {
  // Detach from old processor.
  if (current_thread_ != nullptr) {
    current_thread_->set_processor(nullptr);
  }

  current_thread_ = gt;

  // Attach to this processor.
  if (gt != nullptr) {
    gt->set_processor(this);
    gt->set_state(GreenThreadState::kRunning);
  }
}

void GreenProcessor::Enqueue(GreenThread* gt) {
  gt->set_next(nullptr);
  gt->set_prev(queue_tail_);

  if (queue_tail_ != nullptr) {
    queue_tail_->set_next(gt);
  } else {
    queue_head_ = gt;
  }
  queue_tail_ = gt;
}

GreenThread* GreenProcessor::Dequeue() {
  if (queue_head_ == nullptr) {
    return nullptr;
  }

  GreenThread* gt = queue_head_;
  queue_head_ = gt->next();

  if (queue_head_ != nullptr) {
    queue_head_->set_prev(nullptr);
  } else {
    queue_tail_ = nullptr;
  }

  gt->set_next(nullptr);
  gt->set_prev(nullptr);
  gt->set_state(GreenThreadState::kRunnable);

  return gt;
}

int GreenProcessor::queue_length() const {
  int count = 0;
  GreenThread* gt = queue_head_;
  while (gt != nullptr) {
    count++;
    gt = gt->next();
  }
  return count;
}

}  // namespace internal
}  // namespace v8

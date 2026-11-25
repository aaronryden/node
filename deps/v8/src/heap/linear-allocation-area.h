// Copyright 2021 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_HEAP_LINEAR_ALLOCATION_AREA_H_
#define V8_HEAP_LINEAR_ALLOCATION_AREA_H_

// This header file is included outside of src/heap/.
// Avoid including src/heap/ internals.
#include <atomic>

#include "include/v8-internal.h"
#include "src/common/checks.h"

namespace v8::internal {

// Single process-wide instance for the concurrent young-gen allocation gate.
// Must be an inline variable (C++17), NOT a function-local static inside an
// inline function: a function-local static is permitted to have a distinct
// instance per translation unit that instantiates the inline function, so the
// green-threads setter (green-thread-cow.cc) armed one copy while IncrementTop
// and AllocateFastAligned (other objects in libv8_base_without_compiler) kept
// reading their own always-false copy -- the parallel young-gen bump then raced
// the LAB reset. An inline variable has external/weak linkage and the linker
// deduplicates it to exactly one object across the whole program, so every
// reader and the single writer always touch the same memory. Zero-overhead for
// the common (flag-off) case: a relaxed load that the compiler lowers to a
// plain load on every target.
inline std::atomic<bool> g_concurrent_allocation_gate{false};

// Reference accessor keeps every existing call site (IncrementTop,
// AllocateFastAligned, the green setter) unchanged while reading g inline.
inline std::atomic<bool>& ConcurrentLinearAllocationGate() {
  return g_concurrent_allocation_gate;
}

// A linear allocation area to allocate objects from.
//
// Invariant that must hold at all times:
//   start <= top <= limit
class LinearAllocationArea final {
 public:
  LinearAllocationArea() = default;
  LinearAllocationArea(Address top, Address limit)
      : start_(top), top_(top), limit_(limit) {
    Verify();
  }

  void Reset(Address top, Address limit) {
    start_ = top;
    top_ = top;
    limit_ = limit;
    Verify();
  }

  void ResetStart() { start_ = top_; }

  V8_INLINE bool CanIncrementTop(size_t bytes) const {
    Verify();
    return (top_ + bytes) <= limit_;
  }

  V8_INLINE Address IncrementTop(size_t bytes) {
    if (V8_UNLIKELY(
            ConcurrentLinearAllocationGate().load(std::memory_order_relaxed))) {
      return TryIncrementTopAtomic(bytes);
    }
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_GT_DUMP")) {
      ::fprintf(stderr, "[LAB] serial top=%p limit=%p bytes=%zu -> %p\n",
                (void*)top_, (void*)limit_, bytes, (void*)top_);
      if (V8_UNLIKELY(top_ < 0x100000)) {
        ::fprintf(stderr,
                  "[LAB-SMALL] reserved=%p start=%p limit=%p bytes=%zu "
                  "top_after=%p\n",
                  (void*)top_, (void*)start_, (void*)limit_, bytes,
                  (void*)(top_ + bytes));
      }
    }
#endif
    Address old_top = top_;
    top_ += bytes;
    Verify();
    return old_top;
  }

  // Lock-free reservation within [top_, limit_). Returns the reserved old top,
  // or kNullAddress if the region is exhausted, so the caller falls back to
  // refreshing the LAB / the slow path. CAS retries while another mutator
  // advances top_; a plain (relaxed) load of limit_ is safe because only the
  // main thread/refill changes it. top_ stays a plain member (the class must
  // remain standard-layout for offsetof / IsolateData serialization); the
  // atomic CAS operates on it through std::atomic_ref.
  V8_INLINE Address TryIncrementTopAtomic(size_t bytes) {
    std::atomic_ref<Address> ref(top_);
    Address current = ref.load(std::memory_order_relaxed);
#ifdef V8_ENABLE_GREEN_THREADS
    if (::getenv("GT_GT_DUMP")) {
      ::fprintf(stderr, "[CAS] current=%p limit=%p bytes=%zu\n", (void*)current,
                (void*)limit_, bytes);
    }
#endif
    for (;;) {
      if (V8_UNLIKELY(current + bytes > limit_)) return kNullAddress;
      if (ref.compare_exchange_weak(current, current + bytes,
                                    std::memory_order_relaxed)) {
        Verify();
#ifdef V8_ENABLE_GREEN_THREADS
        if (::getenv("GT_GT_DUMP") && V8_UNLIKELY(current < 0x100000)) {
          ::fprintf(stderr,
                    "[CAS-SMALL] reserved=%p start=%p top=%p limit=%p "
                    "bytes=%zu\n",
                    (void*)current, (void*)start_, (void*)top_,
                    (void*)limit_, bytes);
        }
#endif
        return current;
      }
    }
  }

  V8_INLINE bool DecrementTopIfAdjacent(Address new_top, size_t bytes) {
    Verify();
    if ((new_top + bytes) == top_) {
      top_ = new_top;
      if (start_ > top_) {
        ResetStart();
      }
      Verify();
      return true;
    }
    return false;
  }

  V8_INLINE void SetLimit(Address limit) {
    limit_ = limit;
    Verify();
  }

  V8_INLINE Address start() const {
    Verify();
    return start_;
  }
  V8_INLINE Address top() const {
    Verify();
    return top_;
  }
  V8_INLINE Address limit() const {
    Verify();
    return limit_;
  }
  const Address* top_address() const { return &top_; }
  Address* top_address() { return &top_; }
  const Address* limit_address() const { return &limit_; }
  Address* limit_address() { return &limit_; }

  void Verify() const {
#ifdef DEBUG
    SLOW_DCHECK(start_ <= top_);
    SLOW_DCHECK(top_ <= limit_);
    if (V8_COMPRESS_POINTERS_8GB_BOOL) {
      SLOW_DCHECK(IsAligned(top_, kObjectAlignment8GbHeap));
    } else {
      SLOW_DCHECK(IsAligned(top_, kObjectAlignment));
    }
#endif  // DEBUG
  }

 private:
  // Private offset accessors. Friend classes are allowed to access them.
  friend class IsolateData;

  static constexpr int StartOffset() {
    return offsetof(LinearAllocationArea, start_);
  }
  static constexpr int TopOffset() {
    return offsetof(LinearAllocationArea, top_);
  }
  static constexpr int LimitOffset() {
    return offsetof(LinearAllocationArea, limit_);
  }
  static constexpr int Size() {
    static_assert(sizeof(LinearAllocationArea) == 3 * kSystemPointerSize,
                  "LinearAllocationArea's size must be small because it "
                  "is included in IsolateData.");

    return sizeof(LinearAllocationArea);
  }

  // The start of the LAB. Initially coincides with `top_`. As top is moved
  // ahead, the area [start_, top_[ denotes a range of new objects. This range
  // is reset with `ResetStart()`.
  Address start_ = kNullAddress;
  // The top of the LAB that is used for allocation.
  Address top_ = kNullAddress;
  // Limit of the LAB the denotes the end of the valid range for allocation.
  Address limit_ = kNullAddress;
};

static_assert(std::is_standard_layout_v<LinearAllocationArea>,
              "LinearAllocationArea must be standard layout in order for "
              "offsetof to be defined");

}  // namespace v8::internal

#endif  // V8_HEAP_LINEAR_ALLOCATION_AREA_H_

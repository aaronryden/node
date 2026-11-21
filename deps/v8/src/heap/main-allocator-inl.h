// Copyright 2023 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_HEAP_MAIN_ALLOCATOR_INL_H_
#define V8_HEAP_MAIN_ALLOCATOR_INL_H_

#include "src/heap/main-allocator.h"
// Include the non-inl header before the rest of the headers.

#include "src/flags/flags.h"
#include "src/heap/heap-inl.h"
#include "src/heap/marking-state-inl.h"

namespace v8 {
namespace internal {

AllocationResult MainAllocator::AllocateRaw(int size_in_bytes,
                                            AllocationAlignment alignment,
                                            AllocationOrigin origin,
                                            AllocationHint hint) {
  size_in_bytes = ALIGN_TO_ALLOCATION_ALIGNMENT(size_in_bytes);

  DCHECK_EQ(in_gc(), origin == AllocationOrigin::kGC);
  DCHECK_EQ(in_gc(), isolate_heap()->IsInGC());

  AllocationResult result;

  if (alignment != kTaggedAligned) [[unlikely]] {
    result = AllocateFastAligned(size_in_bytes, nullptr, alignment, origin);
  } else {
    result = AllocateFastUnaligned(size_in_bytes, origin);
  }

  return result.IsFailure() ? AllocateRawSlow(size_in_bytes, alignment, origin)
                            : result;
}

AllocationResult MainAllocator::AllocateFastUnaligned(int size_in_bytes,
                                                      AllocationOrigin origin) {
  size_in_bytes = ALIGN_TO_ALLOCATION_ALIGNMENT(size_in_bytes);
  if (!allocation_info().CanIncrementTop(size_in_bytes)) {
    return AllocationResult::Failure();
  }
  // When the concurrent-allocation gate is set, IncrementTop reserves through a
  // lock-free CAS and reports exhaustion with kNullAddress (the CanIncrementTop
  // pre-check above is racy across workers, so the CAS is the authoritative
  // reservation). The serial path never returns kNullAddress here.
  Address reserved = allocation_info().IncrementTop(size_in_bytes);
  if (V8_UNLIKELY(reserved == kNullAddress)) {
    return AllocationResult::Failure();
  }
  Tagged<HeapObject> obj = HeapObject::FromAddress(reserved);

  MSAN_ALLOCATED_UNINITIALIZED_MEMORY(obj.address(), size_in_bytes);

  DCHECK_IMPLIES(black_allocation_ == BlackAllocation::kAlwaysEnabled,
                 space_heap()->marking_state()->IsMarked(obj));

  return AllocationResult::FromObject(obj);
}

AllocationResult MainAllocator::AllocateFastAligned(
    int size_in_bytes, int* result_aligned_size_in_bytes,
    AllocationAlignment alignment, AllocationOrigin origin) {
  if (V8_UNLIKELY(
          ConcurrentLinearAllocationGate().load(std::memory_order_relaxed))) {
    // Lock-free aligned reservation: CAS the top forward by (size + filler),
    // recomputing filler from the value the CAS actually observes so the
    // reserved region is self-consistent and the object stays aligned. With
    // allocation serialized (the green-mutator lock) this takes one iteration.
    for (;;) {
      Address top = allocation_info().top();
      int filler_size = Heap::GetFillToAlign(top, alignment);
      int aligned_size_in_bytes = size_in_bytes + filler_size;
      Address reserved =
          allocation_info().TryIncrementTopAtomic(aligned_size_in_bytes);
      if (V8_UNLIKELY(reserved == kNullAddress)) {
        return AllocationResult::Failure();
      }
      if (reserved == top) {
        Tagged<HeapObject> obj = HeapObject::FromAddress(reserved);
        if (result_aligned_size_in_bytes)
          *result_aligned_size_in_bytes = aligned_size_in_bytes;
        if (filler_size > 0) {
          obj = space_heap()->PrecedeWithFiller(obj, filler_size);
        }
        MSAN_ALLOCATED_UNINITIALIZED_MEMORY(obj.address(), size_in_bytes);
        DCHECK_IMPLIES(black_allocation_ == BlackAllocation::kAlwaysEnabled,
                       space_heap()->marking_state()->IsMarked(obj));
        return AllocationResult::FromObject(obj);
      }
      // A concurrent worker advanced top_: retry with a freshly poked top so
      // the filler (and therefore the object alignment) matches the base that
      // the CAS actually reserved.
    }
  }

  Address top = allocation_info().top();
  int filler_size = Heap::GetFillToAlign(top, alignment);
  int aligned_size_in_bytes = size_in_bytes + filler_size;

  if (!allocation_info().CanIncrementTop(aligned_size_in_bytes)) {
    return AllocationResult::Failure();
  }
  Tagged<HeapObject> obj = HeapObject::FromAddress(
      allocation_info().IncrementTop(aligned_size_in_bytes));
  if (result_aligned_size_in_bytes)
    *result_aligned_size_in_bytes = aligned_size_in_bytes;

  if (filler_size > 0) {
    obj = space_heap()->PrecedeWithFiller(obj, filler_size);
  }

  MSAN_ALLOCATED_UNINITIALIZED_MEMORY(obj.address(), size_in_bytes);

  DCHECK_IMPLIES(black_allocation_ == BlackAllocation::kAlwaysEnabled,
                 space_heap()->marking_state()->IsMarked(obj));

  return AllocationResult::FromObject(obj);
}

bool MainAllocator::TryFreeLast(Address object_address, int object_size) {
  if (top() != kNullAddress) {
    return allocation_info().DecrementTopIfAdjacent(object_address,
                                                    object_size);
  }
  return false;
}

}  // namespace internal
}  // namespace v8

#endif  // V8_HEAP_MAIN_ALLOCATOR_INL_H_

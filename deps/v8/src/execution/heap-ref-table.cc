// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/heap-ref-table.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <utility>

#include "src/common/assert-scope.h"
#include "src/execution/green-thread.h"
#include "src/execution/isolate.h"
#include "src/handles/handles-inl.h"
#include "src/heap/allocation-result.h"
#include "src/heap/factory.h"
#include "src/heap/heap.h"
#include "src/objects/fixed-array.h"
#include "src/objects/js-objects.h"
#include "src/objects/js-objects-inl.h"
#include "src/objects/map.h"
#include "src/objects/objects-inl.h"
#include "src/objects/slots.h"
#include "src/roots/roots.h"

#include <cstdio>

namespace v8 {
namespace internal {

// ================================================================
// HeapRefTable — hash table management
//
// Table keys and values are *untagged* object addresses, i.e. exactly what
// `HeapObject::address()` returns. Pointers coming from generated code are
// tagged and must be normalized before they are used as keys.
// ================================================================

HeapRefTable::HeapRefTable()
    : entries_(new Entry[kInitialCapacity]),
      capacity_(kInitialCapacity),
      count_(0) {}

HeapRefTable::~HeapRefTable() { delete[] entries_; }

// Returns the slot that either already holds `main_ptr` or is the first free
// slot probe sequence lands on. Must be called with mutex_ held. Because the
// table is resized at a 3/4 load factor there is always a free slot, so the
// probe loop terminates.
int HeapRefTable::FindSlotLocked(Address main_ptr) const {
  uint32_t mask = static_cast<uint32_t>(capacity_ - 1);
  // Objects are at least kTaggedSize aligned, so the low bits hold no
  // information; shift them out before masking.
  int start = static_cast<int>((main_ptr >> 3) & mask);
  for (int step = 0; step < capacity_; step++) {
    int index = (start + step) & mask;
    if (!entries_[index].occupied ||
        entries_[index].main_ptr == main_ptr) {
      return index;
    }
  }
  UNREACHABLE();  // Unreachable while the load factor stays below 1.
}

void HeapRefTable::InsertLocked(Address main_ptr, Address copy_ptr) {
  // Grow before inserting so that a free slot always exists.
  if ((count_ + 1) * 4 >= capacity_ * 3) {
    ResizeLocked();
  }
  int index = FindSlotLocked(main_ptr);
  if (!entries_[index].occupied) {
    entries_[index].occupied = true;
    count_++;
  }
  entries_[index].main_ptr = main_ptr;
  entries_[index].copy_ptr = copy_ptr;
}

void HeapRefTable::ResizeLocked() {
  Entry* old_entries = entries_;
  int old_capacity = capacity_;
  capacity_ *= 2;
  count_ = 0;
  entries_ = new Entry[capacity_];
  resize_count_++;
  for (int i = 0; i < old_capacity; i++) {
    if (old_entries[i].occupied) {
      InsertLocked(old_entries[i].main_ptr, old_entries[i].copy_ptr);
    }
  }
  delete[] old_entries;
}

// Open addressing ties an entry's slot to its key's hash. GC root visiting
// updates keys in place, so after a collection the entries must be rehashed
// or lookups by the relocated key would probe the wrong slot.
void HeapRefTable::RehashLocked() {
  Entry* old_entries = entries_;
  int old_capacity = capacity_;
  count_ = 0;
  entries_ = new Entry[old_capacity];
  for (int i = 0; i < old_capacity; i++) {
    if (old_entries[i].occupied) {
      InsertLocked(old_entries[i].main_ptr, old_entries[i].copy_ptr);
    }
  }
  delete[] old_entries;
}

bool HeapRefTable::FindCopy(Address main_ptr, Address* copy_ptr) {
  base::RecursiveMutexGuard guard(&mutex_);
  int index = FindSlotLocked(main_ptr);
  if (!entries_[index].occupied ||
      entries_[index].main_ptr != main_ptr) {
    return false;
  }
  if (entries_[index].copy_ptr == main_ptr) return false;
  if (copy_ptr) *copy_ptr = entries_[index].copy_ptr;
  return true;
}

Address HeapRefTable::Resolve(Address main_ptr) {
  base::RecursiveMutexGuard guard(&mutex_);
  int index = FindSlotLocked(main_ptr);
  if (!entries_[index].occupied ||
      entries_[index].main_ptr != main_ptr) {
    return main_ptr;
  }
  return entries_[index].copy_ptr;
}

// Lookup — alias for Resolve.
Address HeapRefTable::Lookup(Address main_ptr) { return Resolve(main_ptr); }

// GetView — returns the copy address if cloned, kNullAddress otherwise.
Address HeapRefTable::GetView(Address main_ptr) {
  base::RecursiveMutexGuard guard(&mutex_);
  int index = FindSlotLocked(main_ptr);
  if (!entries_[index].occupied ||
      entries_[index].main_ptr != main_ptr) {
    return kNullAddress;
  }
  return entries_[index].copy_ptr;
}

bool HeapRefTable::IsCloned(Address main_ptr) {
  base::RecursiveMutexGuard guard(&mutex_);
  int index = FindSlotLocked(main_ptr);
  if (!entries_[index].occupied ||
      entries_[index].main_ptr != main_ptr) {
    return false;
  }
  return entries_[index].copy_ptr != main_ptr;
}

void HeapRefTable::RecordCopy(Address main_ptr, Address copy_ptr) {
  base::RecursiveMutexGuard guard(&mutex_);
  InsertLocked(main_ptr, copy_ptr);
}

int HeapRefTable::Count() {
  base::RecursiveMutexGuard guard(&mutex_);
  return count_;
}

int HeapRefTable::ClonedCount() {
  base::RecursiveMutexGuard guard(&mutex_);
  int result = 0;
  for (int i = 0; i < capacity_; i++) {
    if (entries_[i].occupied &&
        entries_[i].main_ptr != entries_[i].copy_ptr) {
      result++;
    }
  }
  return result;
}

// ================================================================
// GC root visiting
//
// The GT-local copies are only reachable through this table, so they must be
// reported as GC roots. During marking, we visit both the main-heap pointers
// (to keep the originals alive) and the copy pointers (to mark the clones).
// If GC moves any object, we update both the entry key and value in-place and
// also scan the owning green thread's stack/registers for references.
// ================================================================

void HeapRefTable::VisitAsRoots(RootVisitor* visitor) {
  base::RecursiveMutexGuard guard(&mutex_);
  bool relocated = false;

  // Old->new address pairs for objects that compaction relocated. Both the
  // main-heap originals and the GT-local copies are visited as roots, so
  // either side may move; we stash the pairs and rewrite the owning green
  // thread's stack and saved registers afterwards. Kept small because it only
  // accumulates entries for objects that actually moved.
  std::vector<std::pair<Address, Address>> main_moves;
  std::vector<std::pair<Address, Address>> copy_moves;

  for (int i = 0; i < capacity_; i++) {
    if (!entries_[i].occupied) continue;

    // Visit the main-heap original: keep it alive while a GT holds a mapping.
    // Visiting updates the slot (compaction moves the object); write the
    // possibly updated value back, normalized to an untagged address.
    if (entries_[i].main_ptr != kNullAddress) {
      Address old_main = entries_[i].main_ptr;
      Address tagged_main = old_main + kHeapObjectTag;
      visitor->VisitRootPointer(Root::kStackRoots, nullptr,
                                FullObjectSlot(&tagged_main));
      Address new_main = tagged_main - kHeapObjectTag;
      if (new_main != old_main) {
        main_moves.emplace_back(old_main, new_main);
        relocated = true;
      }
      entries_[i].main_ptr = new_main;
    }

    // Visit the GT-local copy: it's only reachable through this table.
    Address old_copy = entries_[i].copy_ptr;
    Address tagged_copy = old_copy + kHeapObjectTag;
    visitor->VisitRootPointer(Root::kStackRoots, nullptr,
                              FullObjectSlot(&tagged_copy));
    Address new_copy = tagged_copy - kHeapObjectTag;
    if (new_copy != old_copy) {
      copy_moves.emplace_back(old_copy, new_copy);
      relocated = true;
    }
    entries_[i].copy_ptr = new_copy;
  }

  // The owning green thread is suspended while its roots are visited (Model A
  // runs green threads cooperatively on the embedding thread, so GC never
  // preempts a running body). Rewrite any references in its stack and saved
  // registers that still point at a relocated object, keeping the GT's
  // redirected views consistent with the compacted heap. Only the moved
  // objects are scanned for, so an untouched collection is a no-op here.
  if (owner_ != nullptr) {
    for (const std::pair<Address, Address>& move : main_moves) {
      RedirectWordsLocked(owner_, move.first, move.second);
    }
    for (const std::pair<Address, Address>& move : copy_moves) {
      RedirectWordsLocked(owner_, move.first, move.second);
    }
  }

  // Keys changed by a collection invalidate their slot positions.
  if (relocated) RehashLocked();
}

// ================================================================
// Stack/register redirection
//
// After cloning, all places in the green thread that hold a reference
// to the original must be updated to point to the copy.
// ================================================================

void HeapRefTable::RedirectReferences(GreenThread* gt, Address old_ptr,
                                      Address new_ptr) {
  if (!gt || old_ptr == new_ptr) return;
  base::RecursiveMutexGuard guard(&mutex_);
  RedirectWordsLocked(gt, old_ptr, new_ptr);
}

// Redirect every word of `gt`'s stack range, saved registers, and the
// heap-object descriptor fields that equal `from` - both in untagged
// (HeapObject::address() convention) and tagged form - to `to`.
void HeapRefTable::RedirectWordsLocked(GreenThread* gt, Address from,
                                       Address to) {
  if (!gt || from == to) return;

  // Scan the green thread's stack for slots holding old_ptr.
  const Address stack_bottom =
      gt->stack_limit() & ~static_cast<Address>(kSystemPointerSize - 1);
  const Address stack_top =
      gt->stack_top() & ~static_cast<Address>(kSystemPointerSize - 1);
  const Address from_tagged = from + kHeapObjectTag;
  const Address to_tagged = to + kHeapObjectTag;

  if (stack_bottom != kNullAddress && stack_top != kNullAddress &&
      stack_bottom < stack_top) {
    auto* slot = reinterpret_cast<Address*>(stack_bottom);
    auto* end = reinterpret_cast<Address*>(stack_top);
    for (; slot < end; slot++) {
      if (*slot == from) *slot = to;
      if (*slot == from_tagged) *slot = to_tagged;
    }
  }

  // Scan the saved register buffer.
  uint32_t reg_mask = gt->register_mask();
  constexpr int kRegisterSlots = kGreenRegsSize / kSystemPointerSize;
  for (int i = 0; i < kRegisterSlots; i++) {
    if (reg_mask != 0 && !(reg_mask & (1u << i))) continue;
    auto* reg_ptr =
        reinterpret_cast<Address*>(gt->regs() + i * kSystemPointerSize);
    if (*reg_ptr == from) *reg_ptr = to;
    if (*reg_ptr == from_tagged) *reg_ptr = to_tagged;
  }

  // Redirect the GT descriptor fields that hold heap objects directly.
  if (gt->awaited_promise() == from) {
    gt->set_awaited_promise(to);
  } else if (gt->awaited_promise() == from_tagged) {
    gt->set_awaited_promise(to_tagged);
  }
  if (gt->resolved_await_value() == from) {
    gt->set_resolved_await_value(to);
  } else if (gt->resolved_await_value() == from_tagged) {
    gt->set_resolved_await_value(to_tagged);
  }
}

// ================================================================
// Copy-on-write clone
//
// Objects reachable only through raw addresses are invisible to the GC, and
// allocating a copy can itself trigger a collection that moves objects. The
// clone therefore keeps every object it touches alive through handles
// (handles are GC roots and are updated when an object moves) and only
// touches raw object slots when no allocation can happen in between.
// ================================================================

namespace {

// Upper bound on the objects cloned because of a single write. Beyond this
// budget the remaining references stay shared, so a huge or cyclic object
// graph cannot stall the green thread.
constexpr int kMaxCowCloneNodes = 1024;

// Root data types a green thread may own a private deep copy of: plain object
// and array wrappers. Functions, proxies, Maps, Sets and every other exotic
// type report their own distinct instance type and must be shared, not copied
// — re-creating them with NewWithMap breaks heap invariants (see
// Factory::IsCloneable). JS_OBJECT_TYPE is the *exact* instance type of a plain
// object, not the base of a range shared with functions/proxies.
bool IsCowCloneableRootType(uint16_t instance_type) {
  switch (instance_type) {
    case JS_OBJECT_TYPE:
    case JS_ARRAY_TYPE:
      return true;
    default:
      return false;
  }
}

// Reference types followed by the transitive deep copy: value types (heap
// numbers) and nested object/array wrappers. Everything else — strings, Maps,
// Sets, functions, proxies, BigInts — is immutable from the writer's viewpoint
// and shared.
bool IsCowCloneableNestedType(uint16_t instance_type) {
  switch (instance_type) {
    case JS_OBJECT_TYPE:
    case JS_ARRAY_TYPE:
    case HEAP_NUMBER_TYPE:
      return true;
    default:
      return false;
  }
}

// Fixed-size backing stores whose slots are written in place. They are
// byte-copied rather than deep-cloned.
bool IsCowBackingStoreType(uint16_t instance_type) {
  switch (instance_type) {
    case FIXED_ARRAY_TYPE:
    case PROPERTY_ARRAY_TYPE:
    case FIXED_DOUBLE_ARRAY_TYPE:
      return true;
    default:
      return false;
  }
}

// Collects the offsets of the pointer slots of a copy that reference objects
// the green thread should own privately. It performs no allocation.
class CowReferenceScanner final : public ObjectVisitorWithCageBases {
 public:
  struct Reference {
    int offset;      // Slot offset from the start of the object.
    Address target;   // Untagged address of the referenced object.
  };

  explicit CowReferenceScanner(Isolate* isolate)
      : ObjectVisitorWithCageBases(isolate) {}

  void VisitPointers(Tagged<HeapObject> host, ObjectSlot start,
                     ObjectSlot end) final {
    Address base = host.address();
    for (ObjectSlot slot = start; slot < end; slot++) {
      Tagged<Object> value = slot.load(cage_base());
      if (!IsHeapObject(value)) continue;
      Tagged<HeapObject> ref = Cast<HeapObject>(value);
      // Only follow references into deep-cloneable value types and object/array
      // wrappers. References to Maps, Sets, functions, proxies, strings, etc.
      // are left pointing at the shared objects; their own writes are covered by
      // the per-slot COW barrier.
      if (!IsCowCloneableNestedType(ref->map()->instance_type())) continue;
      references_.push_back(
          {static_cast<int>(slot.address() - base), ref.address()});
    }
  }

  void VisitPointers(Tagged<HeapObject> host, MaybeObjectSlot start,
                     MaybeObjectSlot end) final {
    // Weak slots keep pointing at the shared object.
  }

  void VisitInstructionStreamPointer(Tagged<Code> host,
                                     InstructionStreamSlot slot) final {}

  void VisitMapPointer(Tagged<HeapObject> host) final {}

  std::vector<Reference>& references() { return references_; }

 private:
  std::vector<Reference> references_;
};

// Allocates a copy of a fixed-array-like backing store and byte-copies its
// slots. The slot values themselves stay shared; writes to them are covered
// by their own COW check.
IndirectHandle<HeapObject> CloneCowBackingStore(Isolate* isolate,
                                                IndirectHandle<HeapObject> src,
                                                std::vector<IndirectHandle<HeapObject>>* keep_alive) {
  Tagged<Map> map = src->map();
  uint16_t instance_type = map->instance_type();

  // FixedArrays are variable-size: their map's instance_size() reports the
  // kVariableSizeSentinel (0), so the slot count is read from the object
  // itself — exactly what HeapObject::SizeFromMap does internally for these
  // maps — rather than derived from the map.
  const int src_length =
      UncheckedCast<FixedArrayBase>(
          HeapObject::FromAddress(src->address()))->length();
  const int header = FixedArrayBase::kHeaderSize;
  const int payload = src_length * kTaggedSize;
  if (src_length <= 0 || payload > kMaxRegularHeapObjectSize) {
    return IndirectHandle<HeapObject>();
  }

  Factory* factory = isolate->factory();
  IndirectHandle<HeapObject> result;

  if (instance_type == FIXED_DOUBLE_ARRAY_TYPE) {
    IndirectHandle<FixedArrayBase> copy =
        indirect_handle(factory->NewFixedDoubleArrayWithHoles(src_length));
    keep_alive->push_back(copy);
    result = IndirectHandle<HeapObject>(copy);
  } else if (instance_type == PROPERTY_ARRAY_TYPE) {
    DirectHandle<PropertyArray> direct_copy =
        factory->NewPropertyArray(src_length);
    IndirectHandle<PropertyArray> copy = indirect_handle(direct_copy, isolate);
    keep_alive->push_back(copy);
    result = IndirectHandle<HeapObject>(copy);
  } else {
    MaybeHandle<FixedArray> try_copy =
        factory->TryNewFixedArray(src_length, AllocationType::kYoung);
    if (try_copy.is_null()) return IndirectHandle<HeapObject>();
    Tagged<FixedArray> fixed_array_obj = *try_copy.ToHandleChecked();
    IndirectHandle<FixedArray> copy =
        indirect_handle(DirectHandle<FixedArray>(fixed_array_obj, isolate));
    keep_alive->push_back(copy);
    result = IndirectHandle<HeapObject>(copy);
  }

  // Byte-copy the payload. Both copies have identical length, so the source
  // payload and the destination capacity match exactly. Allocation above may
  // have moved either object, so addresses come from the handles.
  if (payload > 0) {
    std::memcpy(reinterpret_cast<void*>(result->address() + header),
                reinterpret_cast<const void*>(src->address() + header),
                static_cast<size_t>(payload));
  }
  return result;
}

// Allocates the green-thread-local copy of `src` and byte-copies its payload.
// Returns a null handle when the object is not cloneable.
IndirectHandle<HeapObject> AllocateCowCopy(Isolate* isolate,
                                           IndirectHandle<HeapObject> src,
                                           std::vector<IndirectHandle<HeapObject>>* keep_alive) {
  Tagged<Map> map = src->map();
  uint16_t instance_type = map->instance_type();

  if (instance_type == HEAP_NUMBER_TYPE) {
    DirectHandle<HeapNumber> direct_copy =
        isolate->factory()->NewHeapNumberForCodeAssembler(
            UncheckedCast<HeapNumber>(*src)->value());
    IndirectHandle<HeapNumber> copy = indirect_handle(direct_copy, isolate);
    keep_alive->push_back(copy);
    return IndirectHandle<HeapObject>(copy);
  }

  if (IsCowBackingStoreType(instance_type)) {
    return CloneCowBackingStore(isolate, src, keep_alive);
  }

  if (!IsCowCloneableRootType(instance_type)) {
    return IndirectHandle<HeapObject>();
  }

  // Dictionary-mode objects (slow-mode objects) are shared: copying their
  // backing dictionary is not supported yet.
  if (map->is_dictionary_map()) return IndirectHandle<HeapObject>();

  // Allocate a fresh JSObject with the same map.
  DirectHandle<Map> map_handle(map, isolate);
  DirectHandle<AllocationSite> site;
  MaybeDirectHandle<JSObject> maybe_copy =
      JSObject::NewWithMap(isolate, map_handle, site);
  if (maybe_copy.is_null()) return IndirectHandle<HeapObject>();

  IndirectHandle<JSObject> copy(indirect_handle(maybe_copy.ToHandleChecked(),
                                                isolate));
  keep_alive->push_back(copy);

  // Copy the payload (in-object fields). The map, properties and elements
  // words are already set up by the allocator.
  const int header = JSObject::kHeaderSize;
  int src_size = src->SizeFromMap(src->map());
  int copy_size = copy->map()->instance_size();
  int bytes = std::min(src_size, copy_size) - header;
  if (bytes > 0) {
    // Read through the handles: allocating the copy may have collected and
    // moved either object.
    std::memcpy(reinterpret_cast<void*>(copy->address() + header),
                reinterpret_cast<const void*>(src->address() + header),
                static_cast<size_t>(bytes));
  }
  return IndirectHandle<HeapObject>(copy);
}

// Clones `src` into a green-thread-local copy, recursively cloning the
// objects it refers to. Returns `src` unchanged when the object stays shared.
IndirectHandle<HeapObject> CloneObjectInternal(
    Isolate* isolate, HeapRefTable* table,
    IndirectHandle<HeapObject> src,
    std::unordered_set<Address>& visited,
    std::vector<IndirectHandle<HeapObject>>& keep_alive,
    int& budget, bool& out_of_budget) {
  Address src_addr = src->address();

  Tagged<Map> m = src->map();
  uint16_t it = m->instance_type();

  if (budget <= 0) {
    out_of_budget = true;
    return src;
  }

  if (visited.find(src_addr) != visited.end()) {
    // Already cloned (or being cloned): reuse existing copy.
    Address resolved = table->Resolve(src_addr);
    if (resolved == src_addr) return src;
    return IndirectHandle<HeapObject>(
        HeapObject::FromAddress(resolved), isolate);
  }
  visited.insert(src_addr);
  budget--;

  Tagged<Map> map = src->map();
  uint16_t instance_type = map->instance_type();
  const bool is_backing_store = IsCowBackingStoreType(instance_type);

  // Plain object and array wrappers are deep-cloned; their map matches
  // JS_OBJECT_TYPE / JS_ARRAY_TYPE exactly, which excludes functions, proxies,
  // Maps and Sets (they report their own distinct instance types). Heap numbers
  // are value types cloned as nested references. Everything else is immutable
  // from the writer's viewpoint and shared.
  const bool is_data_root = IsCowCloneableRootType(instance_type);
  // The root's in-object fields, elements and properties stores are scanned
  // only for plain objects and arrays.
  const bool is_js_root = is_data_root;

  if (!is_data_root && !is_backing_store &&
      !IsCowCloneableNestedType(instance_type)) {
    return src;  // Immutable or unsupported: share it.
  }

  // Keep the source reachable while allocating.
  keep_alive.push_back(src);
  IndirectHandle<HeapObject> copy =
      AllocateCowCopy(isolate, src, &keep_alive);
  if (copy.is_null()) return src;

  // Backing stores are cloned for write isolation but don't get a table entry.
  const bool record_entry = !is_backing_store;
  if (record_entry) table->RecordCopy(src_addr, copy->address());

  // ---- Reference collection: performs no allocation ----
  struct PendingRedirect {
    int offset;
    Address original;
    IndirectHandle<HeapObject> result;
  };
  bool aborted = false;
  std::vector<PendingRedirect> redirects;

  // For JS objects, scan in-object fields and elements/properties stores.
  if (is_js_root) {
    Tagged<JSObject> src_js = UncheckedCast<JSObject>(*src);

    // Scan elements backing store.
    Tagged<FixedArrayBase> elements = src_js->elements();
    IndirectHandle<HeapObject> elements_src;
    if (elements.address() != kNullAddress) {
      IndirectHandle<HeapObject> elem_ref(elements, isolate);
      if (elem_ref->map()->instance_type() == FIXED_ARRAY_TYPE ||
          elem_ref->map()->instance_type() == PROPERTY_ARRAY_TYPE ||
          elem_ref->map()->instance_type() == FIXED_DOUBLE_ARRAY_TYPE) {
        elements_src = elem_ref;
      }
    }

    // Scan properties PropertyArray.
    Tagged<Object> raw_props = src_js->raw_properties_or_hash();
    IndirectHandle<HeapObject> properties_src;
    if (IsHeapObject(raw_props)) {
      IndirectHandle<HeapObject> prop_ref(
          UncheckedCast<HeapObject>(raw_props), isolate);
      if (prop_ref->map()->instance_type() == PROPERTY_ARRAY_TYPE) {
        properties_src = prop_ref;
      }
    }

    // In-object fields.
    CowReferenceScanner scanner(isolate);
    Address base = copy->address();
    int start = JSObject::kHeaderSize;
    int end = start + copy->map()->GetInObjectProperties() * kTaggedSize;
    if (end > start) {
      scanner.VisitPointers(*copy, ObjectSlot(base + start),
                            ObjectSlot(base + end));
    }

    // Process in-object references.
    for (const auto& ref : scanner.references()) {
      IndirectHandle<HeapObject> target(
          HeapObject::FromAddress(ref.target), isolate);
      bool child_aborted = false;
      IndirectHandle<HeapObject> result = CloneObjectInternal(
          isolate, table, target, visited, keep_alive, budget, child_aborted);
      if (child_aborted) aborted = true;
      redirects.push_back({ref.offset, ref.target, std::move(result)});
    }

    // Process elements reference.
    if (!elements_src.is_null()) {
      bool child_aborted = false;
      IndirectHandle<HeapObject> result = CloneObjectInternal(
          isolate, table, elements_src, visited, keep_alive, budget, child_aborted);
      if (child_aborted) aborted = true;
      redirects.push_back(
          {JSObject::kElementsOffset, elements_src->address(), std::move(result)});
    }

    // Process properties reference.
    if (!properties_src.is_null()) {
      bool child_aborted = false;
      IndirectHandle<HeapObject> result = CloneObjectInternal(
          isolate, table, properties_src, visited, keep_alive, budget, child_aborted);
      if (child_aborted) aborted = true;
      redirects.push_back(
          {JSObject::kPropertiesOrHashOffset, properties_src->address(), std::move(result)});
    }
  } else if (is_backing_store && instance_type != FIXED_DOUBLE_ARRAY_TYPE) {
    // Scan backing store payload slots.
    CowReferenceScanner scanner(isolate);
    Address base = copy->address();
    int start = FixedArrayBase::kHeaderSize;
    int end = copy->SizeFromMap(copy->map());
    if (end > start) {
      scanner.VisitPointers(*copy, ObjectSlot(base + start),
                            ObjectSlot(base + end));
    }
    for (const auto& ref : scanner.references()) {
      IndirectHandle<HeapObject> target(
          HeapObject::FromAddress(ref.target), isolate);
      bool child_aborted = false;
      IndirectHandle<HeapObject> result = CloneObjectInternal(
          isolate, table, target, visited, keep_alive, budget, child_aborted);
      if (child_aborted) aborted = true;
      redirects.push_back({ref.offset, ref.target, std::move(result)});
    }
  }

  if (aborted) {
    out_of_budget = true;
    return src;
  }

  // ---- Final phase: no allocation from here on, so no object can move ----
  // Backing stores (FixedArray/PropertyArray) reference cloned objects from
  // their payload slots; the recorded offsets are measured from the copy's own
  // address, so write them back here. JS objects handle their in-object fields
  // and elements/properties in the block below.
  if (is_backing_store) {
    for (const auto& entry : redirects) {
      const IndirectHandle<HeapObject>& clone = entry.result;
      if (clone.is_null()) continue;
      if (clone->address() == entry.original) continue;
      // Use the same slot type the scanner recorded (ObjectSlot adapts to the
      // pointer-compression width), so the write matches the slot size. A
      // FullObjectSlot would store kSystemPointerSize bytes into kTaggedSize
      // slots, overflowing into the next object's header.
      ObjectSlot(entry.offset + copy->address()).store(*clone);
    }
  }

  if (is_js_root) {
    IndirectHandle<JSObject> copy_js = UncheckedCast<JSObject>(copy);
    for (const auto& entry : redirects) {
      const IndirectHandle<HeapObject>& clone = entry.result;
      if (clone.is_null()) continue;
      if (clone->address() == entry.original) continue;
      // Use the same slot type the scanner recorded (ObjectSlot adapts to the
      // pointer-compression width), so the write matches the slot size. A
      // FullObjectSlot would store kSystemPointerSize bytes into kTaggedSize
      // slots, overflowing into the next object's header.
      ObjectSlot(entry.offset + copy->address()).store(*clone);
    }

    // Re-set elements and properties from redirect results.
    for (const auto& entry : redirects) {
      if (entry.offset == JSObject::kElementsOffset) {
        if (!entry.result.is_null() && entry.result->address() != entry.original) {
          copy_js->set_elements(UncheckedCast<FixedArrayBase>(*entry.result));
        }
      } else if (entry.offset == JSObject::kPropertiesOrHashOffset) {
        if (!entry.result.is_null() && entry.result->address() != entry.original) {
          copy_js->set_raw_properties_or_hash(*entry.result);
        }
      }
    }
  }

  // Re-record: allocation during children may have moved either side.
  if (record_entry) {
    table->RecordCopy(src->address(), copy->address());
  }
  return copy;
}

}  // namespace

Address HeapRefTable::CloneObject(Isolate* isolate, Address obj_ptr) {
  // Every object touched by the clone is held through a handle for the
  // duration of the operation, so a collection triggered by the allocations
  // below neither frees nor loses track of them.
  std::vector<IndirectHandle<HeapObject>> keep_alive;
  std::unordered_set<Address> visited;
  int budget = kMaxCowCloneNodes;
  bool out_of_budget = false;

  IndirectHandle<HeapObject> src(
      HeapObject::FromAddress(obj_ptr), isolate);

  IndirectHandle<HeapObject> copy = CloneObjectInternal(
      isolate, this, src, visited, keep_alive, budget, out_of_budget);

  if (copy.is_null() || *copy == *src) return obj_ptr;
  return copy->address();
}

}  // namespace internal
}  // namespace v8

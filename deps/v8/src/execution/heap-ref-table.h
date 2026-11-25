// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef V8_EXECUTION_HEAP_REF_TABLE_H_
#define V8_EXECUTION_HEAP_REF_TABLE_H_

#include <atomic>
#include <memory>
#include <unordered_set>
#include <vector>

#include "src/base/platform/mutex.h"
#include "src/common/globals.h"
#include "src/objects/objects.h"

namespace v8 {

namespace internal {

class GreenThread;
class Isolate;
class RootVisitor;

// HeapRefTable — per-green-thread mapping from main-heap pointers to
// green-thread-local copies. When an object has not been cloned, the object
// is simply shared and has no entry in the table. Once cloned on write, the
// GT's references are redirected to the copy.
//
// All table operations take `mutex_`: the green thread that owns the table
// mutates it while running on its OS worker thread, while GC root visiting
// (VisitAsRoots) reads it from a mutator thread during concurrent marking.
// The lock is only held for O(1) table primitives, never across a clone.
//
// Uses open-addressing with linear probing. The table grows when it becomes
// 3/4 full, so FindSlotLocked always terminates on a free slot.
class HeapRefTable {
 public:
  HeapRefTable();
  ~HeapRefTable();

  // Delete copy/assign.
  HeapRefTable(const HeapRefTable&) = delete;
  HeapRefTable& operator=(const HeapRefTable&) = delete;

  // Resolve a main-heap pointer to the GT-local copy.
  // If the object hasn't been cloned, returns the original pointer.
  Address Resolve(Address main_ptr);

  // Alias for Resolve — returns the GT-local copy address.
  Address Lookup(Address main_ptr);

  // Get the GT-local copy address for a main-heap pointer.
  // Returns kNullAddress if the object has not been cloned.
  Address GetView(Address main_ptr);

  // Look up an existing clone. Returns true and writes the copy pointer to
  // `*copy_ptr` when `main_ptr` has a distinct GT-local copy. Returns false
  // (leaving `*copy_ptr` untouched) when the object is not cloned, which
  // covers both "no entry" and "entry maps to itself".
  bool FindCopy(Address main_ptr, Address* copy_ptr);

  // Check if a main-heap pointer has a distinct GT-local copy.
  bool IsCloned(Address main_ptr);

  // Record that a GT-local copy exists for a main-heap pointer.
  void RecordCopy(Address main_ptr, Address copy_ptr);

  // The owning green thread (set once by GreenThreadCow::Initialize). The
  // table uses it during GC to keep the GT's raw stack/register references
  // and its dirty set consistent with relocated keys.
  void set_owner(GreenThread* owner) { owner_ = owner; }

  // Clone an object (and, transitively, the heap objects it refers to that
  // are themselves cloneable) into a GT-local copy. Returns the address of
  // the copy, or `obj_ptr` unchanged when the object is not cloneable or
  // allocation fails. Updates the table with every copy that is made.
  //
  // Untagged object addresses, i.e. `HeapObject::address()`. The caller must
  // provide an active handle scope: every object touched by the clone is held
  // through a handle while the copy is being built.
  Address CloneObject(Isolate* isolate, Address obj_ptr);

  // Redirect all GT stack/register references from old_ptr to new_ptr.
  // Called after cloning: finds all places in the GT that hold a reference
  // to old_ptr and updates them to new_ptr. Only scans registers whose index
  // is set in the register mask.
  void RedirectReferences(GreenThread* gt, Address old_ptr, Address new_ptr);

  // Scan all copied objects and visit them as GC roots.
  // Used during V8's GC mark phase.
  void VisitAsRoots(RootVisitor* visitor);

  // Get the number of entries (for debugging/stats).
  int Count();

  // Number of entries that actually hold a distinct clone.
  int ClonedCount();

  // Number of times the table grew (stats).
  int resize_count() const { return resize_count_; }

  // Test accessor for owner_ (exposed for unit testing only).
  GreenThread* owner_for_test() const { return owner_; }

  mutable base::RecursiveMutex mutex_;

 private:
  // Entry for the open-addressing hash table.
  struct Entry {
    Address main_ptr;  // Original main-heap pointer (tagged address)
    Address copy_ptr;  // GT-local copy (tagged address)
    bool occupied = false;
  };

  // Helpers that must be called with `mutex_` held.
  int FindSlotLocked(Address main_ptr) const;
  void InsertLocked(Address main_ptr, Address copy_ptr);
  void ResizeLocked();
  // Scan `gt`'s stack words, saved registers, and descriptor fields,
  // rewriting words equal to `from` (untagged or tagged) to the
  // corresponding form of `to`. Must be called with `mutex_` held.
  // Never allocates.
  void RedirectWordsLocked(GreenThread* gt, Address from, Address to);
  // Re-insert every entry at the current capacity so that each key sits in
  // the slot its (possibly GC-relocated) value now hashes to.
  void RehashLocked();

  // Linear probing hash table.
  static constexpr int kInitialCapacity = 256;
  Entry* entries_;
  int capacity_;
  int count_;
  int resize_count_ = 0;
  GreenThread* owner_ = nullptr;
};

}  // namespace internal
}  // namespace v8

#endif  // V8_EXECUTION_HEAP_REF_TABLE_H_

// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// cctest coverage for the green-thread copy-on-write (CoW) heap isolation.
//
// Each test runs against real heap objects compiled from JS in a real
// isolate, driving the internal CoW API directly:
//
//   * GreenThreadCow::Clone          — deep clone of a heap-object tree
//   * GreenThreadCow::CowCheckBeforeStore — clone-on-store + dirty bookkeeping
//   * GreenThreadCow::COWCheckAndRedirect — allocation-free copy lookup
//   * GreenThreadCow::FlushGreenThreadDirtySet — stack/register redirect
//   * HeapRefTable — clone bookkeeping, growth, self-entry marker
//
// Green-thread copies are registered as GC roots through the global
// GreenThreadGCRootsProvider, so a test that keeps the main object reachable
// can force GC and still resolve a relocated copy through the ref table.

#include "src/execution/green-thread.h"
#include "src/execution/green-thread-cow.h"
#include "src/execution/heap-ref-table.h"
#include "src/handles/handles.h"
#include "src/objects/fixed-array.h"
#include "src/objects/heap-number.h"
#include "src/objects/heap-object.h"
#include "src/objects/js-array.h"
#include "src/objects/smi.h"

#include <sys/mman.h>

#include "test/cctest/cctest.h"
#include "test/cctest/heap/heap-utils.h"

namespace v8 {
namespace internal {

namespace {

// Private stack handed to each green thread. Clone uses the isolate heap;
// the stack is what FlushGreenThreadDirtySet scans when redirecting a GT's
// captured references, so it must be a real, addressable region.
constexpr int kGreenStackSize = 4096 * 16;

// Re-open a compiled value as a DirectHandle. The v8::Local is handle-zone
// rooted, so it survives past the compile; the direct handle cannot be, which
// is exactly what CoW relies on (it re-derives the tagged address).
DirectHandle<JSObject> OpenRecv(v8::Local<v8::Value> value) {
  v8::Local<v8::Object> object = v8::Local<v8::Object>::Cast(value);
  Handle<JSReceiver> recv = v8::Utils::OpenHandle(*object);
  return DirectHandle<JSObject>(UncheckedCast<JSObject>(*recv),
                                CcTest::i_isolate());
}

DirectHandle<JSObject> CopyHandle(Isolate* isolate, Address untagged_addr) {
  return DirectHandle<JSObject>(
      UncheckedCast<JSObject>(HeapObject::FromAddress(untagged_addr)), isolate);
}

DirectHandle<FixedArray> ElementsOf(Isolate* isolate, DirectHandle<JSObject> o) {
  return DirectHandle<FixedArray>(
      UncheckedCast<FixedArray>(UncheckedCast<JSArray>(*o)->elements()),
      isolate);
}

// The GT-local copy for a main-heap pointer, per the ref table.
Address ExistingCopy(HeapRefTable* table, Address main_addr) {
  Address copy = kNullAddress;
  CHECK(table->FindCopy(main_addr, &copy));
  return copy;
}

// Allocate a COW-enabled green thread on a fresh private stack. The returned
// stack pointer must be munmap()'d by the caller once the thread is torn down.
GreenThread* MakeCowThread(Isolate* isolate, void** stack_out) {
  void* stack = mmap(nullptr, kGreenStackSize, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  GreenThread* gt = new GreenThread(isolate);
  gt->InitStack(reinterpret_cast<Address>(stack), kGreenStackSize);
  GreenThreadCow::Initialize(gt);
  *stack_out = stack;
  return gt;
}

// Clone an object tree and verify the copy is a structurally independent view:
// distinct objects with shared maps, nested objects boxed as fresh copies, and
// scalar boxed doubles preserved by value.
TEST(GreenThreadCow_DeepCloneCopiesObjectTree) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[{a: 1}, {b: 2}, 3.5]");
  DirectHandle<JSObject> orig = OpenRecv(recv);

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address orig_addr = orig->address();
  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);

  // The store hook and the allocation-free redirect both find the copy.
  CHECK_EQ(GreenThreadCow::CowCheckBeforeStore(iso, orig_addr), copy_addr);
  CHECK_EQ(GreenThreadCow::COWCheckAndRedirect(orig_addr), copy_addr);

  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK(copy->map() == orig->map());  // maps stay shared
  CHECK_NE(copy->address(), orig_addr);

  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);
  CHECK_NE(copy_elems->address(), orig_elems->address());  // private store
  CHECK_EQ(copy_elems->length(), orig_elems->length());

  // Nested objects are cloned too: new address, shared map.
  DirectHandle<JSObject> nested_o(
      UncheckedCast<JSObject>(orig_elems->get(0)), iso);
  DirectHandle<JSObject> nested_c(
      UncheckedCast<JSObject>(copy_elems->get(0)), iso);
  CHECK_NE(nested_c->address(), nested_o->address());
  CHECK(nested_c->map() == nested_o->map());
  CHECK(ExistingCopy(gt->cow_ref_table(), nested_o->address()) ==
        nested_c->address());

  // A boxed double becomes a fresh HeapNumber with the same value.
  DirectHandle<HeapNumber> hn_o(
      UncheckedCast<HeapNumber>(orig_elems->get(2)), iso);
  DirectHandle<HeapNumber> hn_c(
      UncheckedCast<HeapNumber>(copy_elems->get(2)), iso);
  CHECK_NE(hn_c->address(), hn_o->address());
  CHECK_EQ(hn_c->value(), hn_o->value());

  // array + 2 nested objects + 1 boxed heap number.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 4);

  // Resolving an uncloned pointer returns it unchanged.
  Address fake = orig_addr + 8 * kSystemPointerSize;
  CHECK_EQ(gt->cow_ref_table()->Resolve(fake), fake);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Writing through the green-thread copy leaves the main heap untouched, and
// flushing the dirty set drains the queue and keeps both objects intact.
TEST(GreenThreadCow_WritesStayPrivateAndFlushClears) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  // First store clone marks exactly one object dirty.
  Address copy_addr =
      GreenThreadCow::CowCheckBeforeStore(iso, orig->address());
  CHECK_NE(copy_addr, orig->address());
  CHECK(gt->cow_dirty_objects_ref().size() == 1u);
  CHECK(gt->cow_dirty_flag());

  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);
  CHECK(orig_elems->get(0) == Smi::FromInt(10));

  // The green thread mutates its own private copy.
  copy_elems->set(0, Smi::FromInt(0));
  CHECK(copy_elems->get(0) == Smi::FromInt(0));

  GreenThreadCow::FlushGreenThreadDirtySet(gt);
  CHECK(!gt->cow_dirty_flag());
  CHECK(gt->cow_dirty_objects_ref().empty());
  CHECK_GT(GreenThreadCow::ClonedObjectCount(gt), 0);

  // Main kept the original value; the copy kept the new one.
  CHECK(orig_elems->get(0) == Smi::FromInt(10));
  CHECK(copy_elems->get(0) == Smi::FromInt(0));
  CHECK_NE(copy_elems->address(), orig_elems->address());

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Two green threads cloning the same object get separate copies whose writes
// cannot interfere, and neither write reaches the main heap.
TEST(GreenThreadCow_TwoGreenThreadsGetPrivateCopies) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);

  void* stack1 = nullptr;
  void* stack2 = nullptr;
  GreenThread* gt1 = MakeCowThread(iso, &stack1);
  GreenThread* gt2 = MakeCowThread(iso, &stack2);

  Address a1 = GreenThreadCow::Clone(iso, gt1, orig->address());
  Address a2 = GreenThreadCow::Clone(iso, gt2, orig->address());
  CHECK_NE(a1, orig->address());
  CHECK_NE(a2, orig->address());
  CHECK_NE(a1, a2);  // independent private stores

  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<FixedArray> e1 = ElementsOf(iso, CopyHandle(iso, a1));
  DirectHandle<FixedArray> e2 = ElementsOf(iso, CopyHandle(iso, a2));

  e1->set(0, Smi::FromInt(111));
  e2->set(0, Smi::FromInt(222));

  SetCurrentGreenThread(gt1);
  GreenThreadCow::FlushGreenThreadDirtySet(gt1);
  SetCurrentGreenThread(gt2);
  GreenThreadCow::FlushGreenThreadDirtySet(gt2);

  CHECK(orig_elems->get(0) == Smi::FromInt(10));
  CHECK(e1->get(0) == Smi::FromInt(111));
  CHECK(e2->get(0) == Smi::FromInt(222));

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt1);
  GreenThreadCow::Teardown(gt2);
  delete gt1;
  delete gt2;
  munmap(stack1, kGreenStackSize);
  munmap(stack2, kGreenStackSize);
}

// Copies are GC roots, so both the table keys (main objects) and values (GT
// copies) must survive relocation. Verify across a minor and a major GC.
TEST(GreenThreadCow_ClonesSurviveGarbageCollection) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[{a: 1}]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);

  // Evacuate young space, then mark-compact everything.
  heap::InvokeAtomicMinorGC(CcTest::heap());
  heap::InvokeAtomicMajorGC(CcTest::heap());

  // The main object may have moved; re-open through the still-rooted local.
  DirectHandle<JSObject> moved = OpenRecv(recv);
  Address copy1 = ExistingCopy(gt->cow_ref_table(), moved->address());
  CHECK_NE(copy1, moved->address());

  DirectHandle<JSObject> copy = CopyHandle(iso, copy1);
  CHECK(copy->map() == moved->map());
  CHECK_NE(ElementsOf(iso, copy)->address(),
           ElementsOf(iso, moved)->address());
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 2);  // array + nested object

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A self-referencing array is cloned once: the copy's element points at the
// copy itself, not the original.
TEST(GreenThreadCow_CyclesClonedOnce) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(
      pub, "(() => { const a = []; a.push(a); return a; })()");
  DirectHandle<JSObject> orig = OpenRecv(recv);

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig->address());
  CHECK_NE(copy_addr, orig->address());
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);

  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, CopyHandle(iso, copy_addr));
  Tagged<Object> self_tag = UncheckedCast<Object>(HeapObject::FromAddress(copy_addr));
  CHECK(copy_elems->get(0) == self_tag);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Flush redirects a captured untagged slot that held the main-heap address so
// it ends up pointing at the private copy.
TEST(GreenThreadCow_FlushRedirectsStackSlot) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr =
      GreenThreadCow::CowCheckBeforeStore(iso, orig->address());
  CHECK_NE(copy_addr, orig->address());

  // Simulate a captured slot: a stack word holding the main-heap pointer.
  Address* sp = reinterpret_cast<Address*>(gt->stack_top()) - 2;
  sp[0] = orig->address();
  sp[1] = 0x1234567;  // non-object word must be left alone

  gt->add_cow_dirty_object(orig->address());
  gt->set_cow_dirty_flag(true);
  GreenThreadCow::FlushGreenThreadDirtySet(gt);

  CHECK(!gt->cow_dirty_flag());
  CHECK_EQ(sp[0], copy_addr);  // redirected to the copy
  CHECK_EQ(sp[1], 0x1234567);  // unrelated word untouched
  CHECK_EQ(gt->cow_ref_table()->Resolve(orig->address()), copy_addr);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Ref-table bookkeeping without touching the heap: growth past the initial
// capacity, lookup consistency, and the "entry maps to itself" (shared,
// uncloned) marker.
TEST(GreenThreadCow_TableGrowsAndResolves) {
  HeapRefTable table;
  CHECK_EQ(table.Count(), 0);
  CHECK_EQ(table.resize_count(), 0);

  for (int i = 0; i < 5000; i++) {
    Address key = 0x1000 + static_cast<Address>(i) * kSystemPointerSize;
    table.RecordCopy(key, key + kSystemPointerSize);
  }
  CHECK_EQ(table.Count(), 5000);
  CHECK_GT(table.resize_count(), 0);  // initial capacity is 256

  for (int i = 0; i < 5000; i++) {
    Address key = 0x1000 + static_cast<Address>(i) * kSystemPointerSize;
    Address copy = kNullAddress;
    CHECK(table.FindCopy(key, &copy));
    CHECK_EQ(copy, key + kSystemPointerSize);
    CHECK_EQ(table.Resolve(key), key + kSystemPointerSize);
  }

  // Self-entry: recorded but not cloned -> lookup reports "no copy".
  Address self = 0x90000;
  table.RecordCopy(self, self);
  Address unused_copy = kNullAddress;
  CHECK(!table.FindCopy(self, &unused_copy));
  CHECK(table.IsCloned(0x1000));
  CHECK(!table.IsCloned(self));
  CHECK_EQ(table.Resolve(self), self);
}

// CowCheckBeforeStore is idempotent: the second store to the same object
// returns the same copy without cloning again, and agrees with the
// allocation-free redirect.
TEST(GreenThreadCow_CowCheckBeforeStoreIdempotent) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[1, 2, 3]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address first = GreenThreadCow::CowCheckBeforeStore(iso, orig_addr);
  Address second = GreenThreadCow::CowCheckBeforeStore(iso, orig_addr);
  CHECK_NE(first, orig_addr);
  CHECK_EQ(first, second);  // no re-clone
  CHECK_EQ(GreenThreadCow::COWCheckAndRedirect(orig_addr), first);
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);  // cloned exactly once

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A plain-array clone deep-copies each nested plain object: every nested
// object becomes a distinct copy that shares its map, and the table records
// all of them.
TEST(GreenThreadCow_CloneDeepObjectsInArray) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[{x: 1}, {y: 2}]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK(copy->map() == orig->map());

  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<JSObject> on0(UncheckedCast<JSObject>(orig_elems->get(0)), iso);
  DirectHandle<JSObject> on1(UncheckedCast<JSObject>(orig_elems->get(1)), iso);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);
  DirectHandle<JSObject> n0(UncheckedCast<JSObject>(copy_elems->get(0)), iso);
  DirectHandle<JSObject> n1(UncheckedCast<JSObject>(copy_elems->get(1)), iso);
  // Each nested object is a fresh copy: distinct address, and its own map
  // (the object maps, not the array map, is shared with the original).
  CHECK_NE(n0->address(), on0->address());
  CHECK_NE(n0->address(), n1->address());
  CHECK(n0->map() == on0->map());
  CHECK(n1->map() == on1->map());

  // array + two nested objects; the integer literals stay shared smis.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 3);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Nested arrays are deep-copied level by level: each inner array is a fresh
// copy with its own elements backing store while sharing maps with the
// originals.
TEST(GreenThreadCow_CloneNestedArrays) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[[1, 2], [3, 4]]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  DirectHandle<JSObject> in0(UncheckedCast<JSObject>(copy_elems->get(0)), iso);
  DirectHandle<JSObject> in1(UncheckedCast<JSObject>(copy_elems->get(1)), iso);
  DirectHandle<FixedArray> e0 = ElementsOf(iso, in0);
  DirectHandle<FixedArray> e1 = ElementsOf(iso, in1);

  CHECK_NE(e0->address(), e1->address());
  CHECK_NE(e0->address(), copy_elems->address());
  CHECK_EQ(e0->length(), 2);
  CHECK_EQ(e1->length(), 2);
  CHECK(e0->get(0) == Smi::FromInt(1));
  CHECK(e1->get(1) == Smi::FromInt(4));
  // outer + two inner arrays (four smis stay shared).
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 3);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// When the same object is referenced twice from an array, it is cloned only
// once and both elements end up pointing at that single copy.
TEST(GreenThreadCow_CloneDeduplicatesSharedChild) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(
      pub, "(() => { const o = {s: 1}; return [o, o]; })()");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  Address c0 = UncheckedCast<HeapObject>(copy_elems->get(0)).address();
  Address c1 = UncheckedCast<HeapObject>(copy_elems->get(1)).address();
  CHECK_NE(c0, orig_addr);
  CHECK_EQ(c0, c1);  // one shared child, referenced by both slots

  // array + single shared child.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 2);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// An empty array still gets its own private wrapper; only its (zero-length,
// immutable) elements backing store stays shared.
TEST(GreenThreadCow_CloneEmptyArray) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK(copy->map() == orig->map());
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);
  CHECK_EQ(copy_elems->length(), 0);
  // Just the array wrapper; the empty elements store is shared.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Integer elements of an all-int array are stored as smis and are byte-copied
// verbatim, so they survive the clone unchanged.
TEST(GreenThreadCow_ClonePreservesScalarValues) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20, 30]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  CHECK(copy_elems->get(0) == Smi::FromInt(10));
  CHECK(copy_elems->get(1) == Smi::FromInt(20));
  CHECK(copy_elems->get(2) == Smi::FromInt(30));
  // Only the array wrapper is cloned; the integer literals stay shared smis.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A boxed double stored among objects becomes a fresh HeapNumber holding the
// same bit pattern.
TEST(GreenThreadCow_ClonePreservesBoxedDouble) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[{a: 1}, 3.5]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK_NE(copy_addr, orig_addr);
  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  // The trailing 3.5 is stored as a boxed HeapNumber (dictionary array) and is
  // cloned to a fresh HeapNumber with the same value.
  DirectHandle<HeapNumber> hn_o(
      UncheckedCast<HeapNumber>(orig_elems->get(1)), iso);
  DirectHandle<HeapNumber> hn_c(
      UncheckedCast<HeapNumber>(copy_elems->get(1)), iso);
  CHECK_NE(hn_c->address(), hn_o->address());
  CHECK_EQ(hn_c->value(), hn_o->value());
  CHECK_EQ(hn_c->value(), 3.5);
  // array + nested object + boxed double.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 3);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A three-deep array chain is replicated all the way down: the innermost
// scalar remains reachable and equal through the copies.
TEST(GreenThreadCow_CloneDeeplyNestedArrays) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[[[1]]]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);

  DirectHandle<FixedArray> e_outer = ElementsOf(iso, copy);
  DirectHandle<JSObject> e_mid(UncheckedCast<JSObject>(e_outer->get(0)), iso);
  DirectHandle<FixedArray> e_inner = ElementsOf(iso, e_mid);
  DirectHandle<JSObject> e_leaf(UncheckedCast<JSObject>(e_inner->get(0)), iso);
  DirectHandle<FixedArray> e_val = ElementsOf(iso, e_leaf);

  CHECK(e_val->length() == 1);
  CHECK(e_val->get(0) == Smi::FromInt(1));
  // three nested arrays (the smi stays shared).
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 3);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A large number of unique children each gets a distinct copy that the table
// can resolve individually.
TEST(GreenThreadCow_CloneManyUniqueChildren) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(
      pub, "(() => { const a = {v: 1}; const b = {v: 2}; const c = {v: 3};"
           " return [a, b, c]; })()");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  Address c0 = UncheckedCast<HeapObject>(copy_elems->get(0)).address();
  Address c1 = UncheckedCast<HeapObject>(copy_elems->get(1)).address();
  Address c2 = UncheckedCast<HeapObject>(copy_elems->get(2)).address();
  CHECK_NE(c0, c1);
  CHECK_NE(c1, c2);
  CHECK_NE(c0, c2);
  for (Address a : {c0, c1, c2}) {
    CHECK_EQ(gt->cow_ref_table()->Resolve(a), a);
  }
  // array + three unique children.
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 4);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A plain object root is cloned into a distinct copy that shares its map and
// in-object property count.
TEST(GreenThreadCow_ClonePlainObjectStructural) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "({a: 1})");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK(copy->map() == orig->map());
  CHECK_EQ(copy->map()->GetInObjectProperties(),
           orig->map()->GetInObjectProperties());
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A plain object with several in-object properties clones as a single entry.
TEST(GreenThreadCow_ClonePlainObjectMultipleProps) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "({a: 1, b: 2, c: 3})");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_NE(copy_addr, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  CHECK(copy->map() == orig->map());
  CHECK_EQ(copy->map()->GetInObjectProperties(),
           orig->map()->GetInObjectProperties());
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 1);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// The allocation-free read barrier returns the original untouched when no
// copy exists yet.
TEST(GreenThreadCow_COWCheckAndRedirectUnclonedReturnsOriginal) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  CHECK_EQ(GreenThreadCow::COWCheckAndRedirect(orig_addr), orig_addr);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// After a clone, the allocation-free read barrier redirects to the copy.
TEST(GreenThreadCow_COWCheckAndRedirectAfterCloneReturnsCopy) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_EQ(GreenThreadCow::COWCheckAndRedirect(orig_addr), copy_addr);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// IsCloned, FindCopy and GetView agree with each other for cloned and
// uncloned pointers alike.
TEST(GreenThreadCow_IsClonedFindCopyGetViewSemantics) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);
  HeapRefTable* table = gt->cow_ref_table();

  // Nothing cloned yet.
  CHECK(!table->IsCloned(orig_addr));
  Address none = kNullAddress;
  CHECK(!table->FindCopy(orig_addr, &none));
  CHECK_EQ(table->GetView(orig_addr), kNullAddress);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK(table->IsCloned(orig_addr));
  Address copy = kNullAddress;
  CHECK(table->FindCopy(orig_addr, &copy));
  CHECK_EQ(copy, copy_addr);
  CHECK_EQ(table->GetView(orig_addr), copy_addr);

  // A neighbour address is neither cloned nor mapped.
  Address neighbour = orig_addr + kSystemPointerSize;
  CHECK(!table->IsCloned(neighbour));
  CHECK_EQ(table->GetView(neighbour), kNullAddress);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Lookup is documented as an alias for Resolve; verify they agree.
TEST(GreenThreadCow_LookupIsAliasForResolve) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);
  HeapRefTable* table = gt->cow_ref_table();

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  CHECK_EQ(table->Lookup(orig_addr), copy_addr);
  CHECK_EQ(table->Resolve(orig_addr), copy_addr);
  CHECK_EQ(table->Lookup(orig_addr), table->Resolve(orig_addr));

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Several independent writes queue several dirty objects that the flush
// drains all at once, leaving both views intact and private.
TEST(GreenThreadCow_FlushMultipleWritesDrainsAll) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  v8::Local<v8::Value> recv2 = CompileRunChecked(pub, "[30, 40]");
  DirectHandle<JSObject> orig2 = OpenRecv(recv2);
  Address orig2_addr = orig2->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy1 = GreenThreadCow::CowCheckBeforeStore(iso, orig_addr);
  Address copy2 = GreenThreadCow::CowCheckBeforeStore(iso, orig2_addr);
  CHECK_NE(copy1, orig_addr);
  CHECK_NE(copy2, orig2_addr);
  CHECK_NE(copy1, copy2);
  CHECK_EQ(gt->cow_dirty_objects_ref().size(), 2u);
  CHECK(gt->cow_dirty_flag());

  GreenThreadCow::FlushGreenThreadDirtySet(gt);
  CHECK(!gt->cow_dirty_flag());
  CHECK(gt->cow_dirty_objects_ref().empty());
  // Both copies still resolve after the flush.
  CHECK_EQ(gt->cow_ref_table()->Resolve(orig_addr), copy1);
  CHECK_EQ(gt->cow_ref_table()->Resolve(orig2_addr), copy2);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Flushing when nothing was written is a no-op and never crashes.
TEST(GreenThreadCow_FlushNoopWhenNotDirty) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  CHECK(!gt->cow_dirty_flag());
  CHECK(gt->cow_dirty_objects_ref().empty());
  GreenThreadCow::FlushGreenThreadDirtySet(gt);  // must not crash
  CHECK(!gt->cow_dirty_flag());
  CHECK(gt->cow_dirty_objects_ref().empty());

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// SetIsolateCowCapable marks an isolate for the COW write barrier and the
// flag persists across green-thread teardown.
TEST(GreenThreadCow_SetIsolateCowCapablePersists) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  CHECK(!GreenThreadCow::IsolateCowCapable(iso));
  GreenThreadCow::SetIsolateCowCapable(iso);
  CHECK(GreenThreadCow::IsolateCowCapable(iso));

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);
  CHECK(GreenThreadCow::IsolateCowCapable(iso));

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);

  // The mark is purposefully permanent for the isolate's lifetime.
  CHECK(GreenThreadCow::IsolateCowCapable(iso));
}

// The COW active-count goes 0 -> 1 while a view is live and back to 0 after
// teardown.
TEST(GreenThreadCow_ActiveCountTracksLifecycle) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  uint32_t base = GreenThreadCow::ActiveCount();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  CHECK_EQ(GreenThreadCow::ActiveCount(), base + 1u);
  SetCurrentGreenThread(gt);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);

  CHECK_EQ(GreenThreadCow::ActiveCount(), base);
}

// Cloning through a null green thread returns the input unchanged.
TEST(GreenThreadCow_CloneReturnsOriginalForNullThread) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[10, 20]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  Address result = GreenThreadCow::Clone(iso, nullptr, orig_addr);
  CHECK_EQ(result, orig_addr);
}

// A mutual a<->b cycle is cloned once each, and each copy resolves to its own
// distinct address.
TEST(GreenThreadCow_CloneMutualCycleClonedOnce) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(
      pub, "(() => { const a = {b: 1}; const b = {a: 2}; a.b = b; b.a = a;"
           " return [a, b]; })()");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);
  HeapRefTable* table = gt->cow_ref_table();

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  Address a_copy = UncheckedCast<HeapObject>(copy_elems->get(0)).address();
  Address b_copy = UncheckedCast<HeapObject>(copy_elems->get(1)).address();
  CHECK_NE(a_copy, b_copy);
  CHECK_EQ(table->Resolve(a_copy), a_copy);
  CHECK_EQ(table->Resolve(b_copy), b_copy);
  // array + a + b.
  CHECK_EQ(table->ClonedCount(), 3);

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// Every object in a cloned tree shares the original's map, even deep ones.
TEST(GreenThreadCow_CloneSharedMapAcrossTree) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(pub, "[{a: 1}, {b: 2}]");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> orig_elems = ElementsOf(iso, orig);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  for (int i = 0; i < 2; i++) {
    DirectHandle<JSObject> on(
        UncheckedCast<JSObject>(orig_elems->get(i)), iso);
    DirectHandle<JSObject> n(UncheckedCast<JSObject>(copy_elems->get(i)), iso);
    CHECK_NE(n->address(), on->address());
    // The nested object maps match the nested originals (not the array map).
    CHECK(n->map() == on->map());
  }

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

// A shared object referenced from two nested slots is cloned only once, even
// when the reference lies one level deep.
TEST(GreenThreadCow_CloneDedupsDeepSharedChild) {
  v8::Isolate* pub = CcTest::isolate();
  v8::HandleScope scope(pub);
  v8::Local<v8::Context> ctx = v8::Context::New(pub);
  v8::Context::Scope ctx_scope(ctx);
  Isolate* iso = CcTest::i_isolate();

  v8::Local<v8::Value> recv = CompileRunChecked(
      pub, "(() => { const o = {s: 1}; return [{d: o}, {d: o}]; })()");
  DirectHandle<JSObject> orig = OpenRecv(recv);
  Address orig_addr = orig->address();

  void* stack = nullptr;
  GreenThread* gt = MakeCowThread(iso, &stack);
  SetCurrentGreenThread(gt);

  Address copy_addr = GreenThreadCow::Clone(iso, gt, orig_addr);
  DirectHandle<JSObject> copy = CopyHandle(iso, copy_addr);
  DirectHandle<FixedArray> copy_elems = ElementsOf(iso, copy);

  DirectHandle<JSObject> c0(UncheckedCast<JSObject>(copy_elems->get(0)), iso);
  DirectHandle<JSObject> c1(UncheckedCast<JSObject>(copy_elems->get(1)), iso);
  CHECK_NE(c0->address(), c1->address());  // the wrappers differ
  CHECK_EQ(gt->cow_ref_table()->ClonedCount(), 4);  // array + 2 wrappers + 1 o

  SetCurrentGreenThread(nullptr);
  GreenThreadCow::Teardown(gt);
  delete gt;
  munmap(stack, kGreenStackSize);
}

}  // namespace

}  // namespace internal
}  // namespace v8

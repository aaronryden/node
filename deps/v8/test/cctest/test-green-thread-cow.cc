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

}  // namespace

}  // namespace internal
}  // namespace v8

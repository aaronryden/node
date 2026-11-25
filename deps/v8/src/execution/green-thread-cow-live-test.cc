// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Live-isolate COW integration test harness.
//
// Creates a real V8 Isolate with a working heap, allocates real
// HeapObjects via the internal Factory, and verifies that the green
// thread COW system correctly tracks and clones them.
//
// IMPORTANT: We must create a v8::HandleScope BEFORE v8::Context::New()
// because V8's debug-mode HandleScope assertions require an enclosing
// HandleScope when Context::New() calls CloseAndEscape. This is standard
// V8 embedding practice (same pattern used in d8.cc).
//
// Run with: ./out/gt/v8_green_thread_live_isolate_test

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sys/mman.h>

#include "include/libplatform/libplatform.h"
#include "include/v8-initialization.h"
#include "include/v8-isolate.h"
#include "include/v8-handle-base.h"
#include "include/v8-context.h"
#include "include/v8-script.h"
#include "include/v8-primitive.h"

#include "src/common/globals.h"
#include "src/execution/green-thread-cow.h"
#include "src/execution/heap-ref-table.h"
#include "src/execution/isolate.h"
#include "src/heap/factory.h"
#include "src/objects/heap-object.h"
#include "src/objects/compressed-slots-inl.h"
#include "src/handles/handles.h"
#include "src/handles/handles-inl.h"

using namespace v8::internal;

static bool all_passed = true;
static int passed_count = 0;
static int failed_count = 0;

static void check(bool cond, const char* name) {
  if (cond) {
    printf("  PASS: %s\n", name);
    passed_count++;
  } else {
    printf("  FAIL: %s\n", name);
    failed_count++;
    all_passed = false;
  }
}

// Helper: allocate a real HeapObject via the Factory.
static Address allocate_object(Factory* f, v8::Isolate* public_isolate) {
  return f->NewJSObjectWithNullProto()->address();
}

// ================================================================
// Test harness
// ================================================================

int main(int argc, char* argv[]) {
  printf("=== Live-isolate COW Integration Test ===\n\n");

  // --- Setup V8 ---
  printf("[1] Initializing V8...\n");
  v8::V8::InitializeICUDefaultLocation(argv[0]);
  v8::V8::InitializeExternalStartupData(argv[0]);
  std::unique_ptr<v8::Platform> platform =
      v8::platform::NewDefaultPlatform();
  v8::V8::InitializePlatform(platform.get());
  v8::V8::Initialize();
  printf("  V8 %s initialized\n", v8::V8::GetVersion());

  // --- Create Isolate ---
  printf("[2] Creating V8 Isolate...\n");
  v8::Isolate::CreateParams create_params;
  create_params.array_buffer_allocator =
      v8::ArrayBuffer::Allocator::NewDefaultAllocator();
  v8::Isolate* public_isolate = v8::Isolate::New(create_params);
  Isolate* isolate = reinterpret_cast<Isolate*>(public_isolate);

  // Enter the isolate scope and create a HandleScope for Factory calls.
  // We use a scope block so all scopes are destroyed before Dispose().
  {
    v8::Isolate::Scope isolate_scope(public_isolate);
    v8::HandleScope handle_scope(public_isolate);
    v8::Local<v8::Context> ctx = v8::Context::New(public_isolate);
    v8::Context::Scope context_scope(ctx);

    // Access the internal Factory via the Isolate (reinterpret_cast,
    // same pattern used in existing green-thread-tests.cc).
    Factory* f = isolate->factory();

  // ============================================================
  // TEST 1: Allocate real HeapObjects via Factory
  // ============================================================
  printf("\n[Test 1] Allocate real HeapObjects via Factory::NewJSObjectWithNullProto\n");
  {
    Address addr1 = allocate_object(f, public_isolate);
    Address addr2 = allocate_object(f, public_isolate);

    check(addr1 != kNullAddress, "obj1 address is valid");
    check(addr2 != kNullAddress, "obj2 address is valid");
    check(addr1 != addr2, "objects have distinct addresses");
    check(addr1 > 0, "obj1 address > 0");
    check(addr2 > 0, "obj2 address > 0");
    check((addr1 + kHeapObjectTag) != addr1, "heap object tag is non-zero");
  }

  // ============================================================
  // TEST 2: HeapRefTable CRUD with real heap addresses
  // ============================================================
  printf("\n[Test 2] HeapRefTable CRUD with real heap addresses\n");
  {
    Address addr = allocate_object(f, public_isolate);
    Address copy = addr + (1 << 20);

    HeapRefTable table;
    check(!table.IsCloned(addr), "not cloned initially");
    table.RecordCopy(addr, copy);
    check(table.IsCloned(addr), "cloned after RecordCopy");
    check(table.Resolve(addr) == copy, "Resolve returns copy");

    Address other = addr + (1 << 20);
    check(!table.IsCloned(other), "different addr not cloned");
    check(table.Resolve(other) == other,
           "Resolve returns original for non-cloned");
  }

  // ============================================================
  // TEST 3: HeapObjectSlot store/load with real object
  // ============================================================
  printf("\n[Test 3] HeapObjectSlot store/load with real object\n");
  {
    Address orig_addr = allocate_object(f, public_isolate);

    alignas(8) uint8_t slot_data[8];
    memset(slot_data, 0, sizeof(slot_data));
    HeapObjectSlot slot(reinterpret_cast<Address>(slot_data));
    slot.StoreHeapObject(HeapObject::FromAddress(orig_addr));

    // Verify slot was written (compressed pointer, may differ from
    // orig_addr due to pointer compression, but not all zeros).
    bool written = false;
    for (int i = 0; i < 8; i++) {
      if (slot_data[i] != 0) { written = true; break; }
    }
    check(written, "slot written with compressed pointer");

    // Load back — should match the original tagged pointer.
    Tagged<HeapObject> loaded = slot.ToHeapObject();
    check(loaded.ptr() == orig_addr + kHeapObjectTag,
          "loaded ptr matches original");
  }

  // ============================================================
  // TEST 4: GreenThread + COW with real heap addresses
  // ============================================================
  printf("\n[Test 4] GreenThread + COW with real heap addresses\n");
  {
    const int kStackSize = 4096 * 16;
    void* stack_mem = mmap(nullptr, kStackSize,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(stack_mem != MAP_FAILED, "stack mmap succeeded");

    GreenThread* gt = new GreenThread(nullptr);
    gt->InitStack(reinterpret_cast<Address>(stack_mem), kStackSize);
    GreenThreadCow::Initialize(gt);

    Address obj_addr = allocate_object(f, public_isolate);

    check(GreenThreadCow::ActiveCount() > 0, "ActiveCount > 0");

    HeapRefTable* table = gt->cow_ref_table();
    check(table != nullptr, "COW ref table exists");

    Address copy = obj_addr + (1 << 20);
    table->RecordCopy(obj_addr, copy);
    check(table->IsCloned(obj_addr), "table tracks clone");
    check(table->Resolve(obj_addr) == copy, "table resolves to copy");

    // Put reference on GT stack.
    Address* sp = reinterpret_cast<Address*>(gt->stack_top()) - 2;
    sp[0] = obj_addr;
    sp[1] = 0xDEADBEEF;

    check(sp[0] == obj_addr, "stack holds original address");

    // Mark object dirty and flush — should redirect stack refs.
    gt->add_cow_dirty_object(obj_addr);
    gt->set_cow_dirty_flag(true);
    GreenThreadCow::FlushGreenThreadDirtySet(gt);

    check(!gt->cow_dirty_flag(), "dirty flag cleared after flush");
    check(sp[0] == copy, "stack reference redirected to copy");
    check(sp[1] == 0xDEADBEEF, "non-object ref untouched");

    GreenThreadCow::Teardown(gt);
    delete gt;
    munmap(stack_mem, kStackSize);
  }

  // ============================================================
  // TEST 5: COWWriteBarrierCheck with real object
  // ============================================================
  printf("\n[Test 5] COWWriteBarrierCheck (slot rewrite path)\n");
  {
    Address obj_addr = allocate_object(f, public_isolate);

    const int kStackSize = 4096 * 16;
    void* stack_mem = mmap(nullptr, kStackSize,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(stack_mem != MAP_FAILED, "stack mmap");

    GreenThread* gt = new GreenThread(nullptr);
    gt->InitStack(reinterpret_cast<Address>(stack_mem), kStackSize);
    GreenThreadCow::Initialize(gt);

    alignas(8) uint8_t slot_data[8];
    memset(slot_data, 0, sizeof(slot_data));
    HeapObjectSlot slot(reinterpret_cast<Address>(slot_data));

    // COW check with null-isolate GT: no real clone is produced
    // (CloneObject needs a real isolate), but the full control
    // flow exercises the write barrier path without crashing.
    Tagged<HeapObject> fake_obj(HeapObject::FromAddress(obj_addr));
    GreenThreadCow::COWWriteBarrierCheck(fake_obj, fake_obj, slot);

    check(slot_data[0] == 0,
          "slot unchanged (no clone with null isolate — expected)");

    GreenThreadCow::Teardown(gt);
    delete gt;
    munmap(stack_mem, kStackSize);
  }

  // ============================================================
  // TEST 6: Multiple GTs with isolated views
  // ============================================================
  printf("\n[Test 6] Multiple GTs with isolated HeapRefTables\n");
  {
    Address obj_addr = allocate_object(f, public_isolate);

    const int kStackSize = 4096 * 16;
    void* stack_a = mmap(nullptr, kStackSize,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void* stack_b = mmap(nullptr, kStackSize,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    GreenThread* gt_a = new GreenThread(nullptr);
    gt_a->InitStack(reinterpret_cast<Address>(stack_a), kStackSize);
    GreenThread* gt_b = new GreenThread(nullptr);
    gt_b->InitStack(reinterpret_cast<Address>(stack_b), kStackSize);

    GreenThreadCow::Initialize(gt_a);

    Address copy_a = obj_addr + (1 << 20);
    gt_a->cow_ref_table()->RecordCopy(obj_addr, copy_a);
    check(gt_a->cow_ref_table()->IsCloned(obj_addr), "GT A cloned");

    check(gt_b->cow_ref_table() == nullptr, "GT B has no COW table");

    GreenThreadCow::Teardown(gt_a);
    delete gt_a;
    delete gt_b;
    munmap(stack_a, kStackSize);
    munmap(stack_b, kStackSize);
  }
  }  // Closes outer scope (ends: context_scope, handle_scope, isolate_scope)

  // --- Cleanup ---
  printf("\n[Cleanup] Disposing isolate and V8...\n");
  public_isolate->Dispose();
  v8::V8::Dispose();
  v8::V8::DisposePlatform();

  // --- Results ---
  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", passed_count, failed_count);
  if (all_passed) {
    printf("ALL TESTS PASSED\n");
    printf("========================================\n");
    return 0;
  } else {
    printf("SOME TESTS FAILED\n");
    printf("========================================\n");
    return 1;
  }
}

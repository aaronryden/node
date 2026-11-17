// Copyright 2026 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Regression coverage for the green-thread COW external-reference
// registration. Adding an entry to EXTERNAL_REFERENCE_LIST grows
// ExternalReferenceTable::kSizeIsolateIndependent / kSize and the
// ref_name_ table. These tests assert that the entry is actually present
// with a valid address and that the static name table stays consistent
// with the (now larger) enum, which is exactly the kind of size/name
// mismatch that silently corrupts the isolate during mksnapshot setup.

#include "src/codegen/external-reference-table.h"

#include <cstring>

#include "src/codegen/external-reference.h"

#include "test/cctest/cctest.h"

#if V8_ENABLE_GREEN_THREADS
#include "src/execution/green-thread-cow.h"
#endif  // V8_ENABLE_GREEN_THREADS

namespace v8 {
namespace internal {

// The green-thread COW clone helper is the last entry of
// EXTERNAL_REFERENCE_LIST, so it must be the highest-indexed
// isolate-independent external reference. It has to resolve to the real
// (stable, non-null) function address.
TEST(ExternalReferenceRegistration_GreenThreadCowHelperIsRegistered) {
  constexpr const char* kExpectedName =
      "GreenThreadCow::CowCloneForGeneratedCode";

  int found = -1;
  ExternalReferenceTable table;
  for (int i = 0; i < ExternalReferenceTable::kSizeIsolateIndependent; ++i) {
    // Index 0 is the special "nullptr" reference; every other entry must
    // carry a non-null name. This catches a table that grew (or shrank)
    // out of sync with EXTERNAL_REFERENCE_LIST.
    CHECK_NE(i == 0, table.name(i) == nullptr);
    if (table.name(i) != nullptr &&
        std::strcmp(table.name(i), kExpectedName) == 0) {
      CHECK_EQ(-1, found);  // Exactly one occurrence.
      found = i;
    }
  }
  CHECK_NE(-1, found);
  // It sits at the end of the isolate-independent range.
  CHECK_EQ(ExternalReferenceTable::kSizeIsolateIndependent - 1, found);

#if V8_ENABLE_GREEN_THREADS
  // The FUNCTION_REFERENCE wiring resolves to the real function pointer.
  Address address =
      ExternalReference::green_thread_cow_clone_if_needed().address();
  CHECK_NE(kNullAddress, address);
  CHECK_EQ(reinterpret_cast<Address>(
               &GreenThreadCow::CowCloneForGeneratedCode),
           address);
#endif  // V8_ENABLE_GREEN_THREADS
}

// The reference must be encoded as an external function, not as an
// IsolateFieldId (which is a distinct, smaller-numbered encoding used for
// per-isolate field accesses). Mis-encoding would corrupt the table.
TEST(ExternalReferenceRegistration_GreenThreadCowHelperIsNotIsolateField) {
#if V8_ENABLE_GREEN_THREADS
  ExternalReference reference =
      ExternalReference::green_thread_cow_clone_if_needed();
  CHECK(!reference.IsIsolateFieldId());
#endif  // V8_ENABLE_GREEN_THREADS
}

// The total table must cover the isolate-independent range and still grow
// with the isolate-dependent entries; a regression in the table-size
// computation must not let the dependent range overlap the independent
// one.
TEST(ExternalReferenceRegistration_TableRangeIsStrictlyOrdered) {
  CHECK_GT(ExternalReferenceTable::kSizeIsolateIndependent,
           ExternalReferenceTable::kSpecialReferenceCount);
  CHECK_EQ(ExternalReferenceTable::kSize,
           ExternalReferenceTable::kSizeIsolateIndependent +
               ExternalReferenceTable::kExternalReferenceCountIsolateDependent +
               ExternalReferenceTable::kIsolateFieldReferenceCount +
               ExternalReferenceTable::kStubCacheReferenceCount +
               ExternalReferenceTable::kStatsCountersReferenceCount);
}

}  // namespace internal
}  // namespace v8

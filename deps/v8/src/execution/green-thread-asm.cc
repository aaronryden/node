// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "src/execution/green-thread.h"

namespace v8 {
namespace internal {

#if V8_TARGET_ARCH_ARM64

// ARM64 register save / restore layout (kGreenRegsSize = 696 bytes):
//   [0..167]     x0-x17, x29, sp, x30     168 bytes  (integer regs)
//   [168..175]   padding (16-byte aligned)    8 bytes
//   [176..687]   q0-q31 (NEON/SIMD)       512 bytes
//   [688..691]   FPSR                        4 bytes
//   [692..695]   FPCR                        4 bytes
//
// The q-pair stores require 16-byte-aligned offsets; the 8-byte padding
// after integer regs ensures q0-q31 starts at a 16-byte boundary.

// ARM64 register save: x0-x17, x29, sp, x30, q0-q31, FPSR, FPCR
extern "C" void green_thread_save_registers_arm64(uint8_t* dst) {
  // Save GPRs — use pre-indexed offsets (no pointer modification needed).
  asm volatile(
    "stp x0, x1, [%0, #0]\n"
    "stp x2, x3, [%0, #16]\n"
    "stp x4, x5, [%0, #32]\n"
    "stp x6, x7, [%0, #48]\n"
    "stp x8, x9, [%0, #64]\n"
    "stp x10, x11, [%0, #80]\n"
    "stp x12, x13, [%0, #96]\n"
    "stp x14, x15, [%0, #112]\n"
    "stp x16, x17, [%0, #128]\n"
    "str x29, [%0, #144]\n"
    "mov x17, sp\n"
    "str x17, [%0, #152]\n"
    "str x30, [%0, #160]\n"
    :
    : "r"(dst)
    : "x17", "memory");

  // Save NEON / SIMD registers (q0-q31) at dst + 176.
  uint8_t* n = dst + 176;
  asm volatile(
    "stp q0, q1, [%0, #0]\n"
    "stp q2, q3, [%0, #16]\n"
    "stp q4, q5, [%0, #32]\n"
    "stp q6, q7, [%0, #48]\n"
    "stp q8, q9, [%0, #64]\n"
    "stp q10, q11, [%0, #80]\n"
    "stp q12, q13, [%0, #96]\n"
    "stp q14, q15, [%0, #112]\n"
    "stp q16, q17, [%0, #128]\n"
    "stp q18, q19, [%0, #144]\n"
    "stp q20, q21, [%0, #160]\n"
    "stp q22, q23, [%0, #176]\n"
    "stp q24, q25, [%0, #192]\n"
    "stp q26, q27, [%0, #208]\n"
    "stp q28, q29, [%0, #224]\n"
    "stp q30, q31, [%0, #240]\n"
    :
    : "r"(n)
    : "memory");

  // Save FPSR and FPCR.
  uint64_t fpsr_val, fpcr_val;
  asm volatile("mrs %0, FPSR\n" : "=r"(fpsr_val));
  asm volatile("mrs %0, FPCR\n" : "=r"(fpcr_val));
  dst[688] = static_cast<uint8_t>(fpsr_val);
  dst[689] = static_cast<uint8_t>(fpsr_val >> 8);
  dst[690] = static_cast<uint8_t>(fpsr_val >> 16);
  dst[691] = static_cast<uint8_t>(fpsr_val >> 24);
  dst[692] = static_cast<uint8_t>(fpcr_val);
  dst[693] = static_cast<uint8_t>(fpcr_val >> 8);
  dst[694] = static_cast<uint8_t>(fpcr_val >> 16);
  dst[695] = static_cast<uint8_t>(fpcr_val >> 24);
}

// ARM64 register restore: x0-x17, x29, sp, x30, q0-q31, FPSR, FPCR
//
// Uses a single combined asm block with `ret` at the end. The `ret`
// returns to the value restored in x30 (link register from the save
// buffer). This is safe because the function is called as a tail-call
// from a context where the C return path is no longer needed — the
// scheduler has already longjmp'd here and the green thread's context
// is being restored to continue its entry function.
//
// All GPRs except x28 are listed as clobbered. The compiler therefore
// places input operands in x28, avoiding conflicts with the GPR
// restore writes.
extern "C" void green_thread_restore_registers_arm64(const uint8_t* src) {
  // Extract FPSR/FPCR values before the asm block is entered.
  uint64_t fpsr_val = static_cast<uint64_t>(src[688]) |
                      (static_cast<uint64_t>(src[689]) << 8) |
                      (static_cast<uint64_t>(src[690]) << 16) |
                      (static_cast<uint64_t>(src[691]) << 24);
  uint64_t fpcr_val = static_cast<uint64_t>(src[692]) |
                      (static_cast<uint64_t>(src[693]) << 8) |
                      (static_cast<uint64_t>(src[694]) << 16) |
                      (static_cast<uint64_t>(src[695]) << 24);

  uintptr_t gpr_ptr = reinterpret_cast<uintptr_t>(src);
  uintptr_t neon_ptr = gpr_ptr + 176;

  asm volatile(
    // --- Restore GPRs ---
    "ldp x0, x1, [%0, #0]\n"
    "ldp x2, x3, [%0, #16]\n"
    "ldp x4, x5, [%0, #32]\n"
    "ldp x6, x7, [%0, #48]\n"
    "ldp x8, x9, [%0, #64]\n"
    "ldp x10, x11, [%0, #80]\n"
    "ldp x12, x13, [%0, #96]\n"
    "ldp x14, x15, [%0, #112]\n"
    "ldp x16, x17, [%0, #128]\n"
    "ldr x29, [%0, #144]\n"
    "ldr x17, [%0, #152]\n"
    "mov sp, x17\n"
    "ldr x30, [%0, #160]\n"

    // --- Restore NEON (16 q-register pairs) ---
    "ldp q0, q1, [%1, #0]\n"
    "ldp q2, q3, [%1, #16]\n"
    "ldp q4, q5, [%1, #32]\n"
    "ldp q6, q7, [%1, #48]\n"
    "ldp q8, q9, [%1, #64]\n"
    "ldp q10, q11, [%1, #80]\n"
    "ldp q12, q13, [%1, #96]\n"
    "ldp q14, q15, [%1, #112]\n"
    "ldp q16, q17, [%1, #128]\n"
    "ldp q18, q19, [%1, #144]\n"
    "ldp q20, q21, [%1, #160]\n"
    "ldp q22, q23, [%1, #176]\n"
    "ldp q24, q25, [%1, #192]\n"
    "ldp q26, q27, [%1, #208]\n"
    "ldp q28, q29, [%1, #224]\n"
    "ldp q30, q31, [%1, #240]\n"

    // --- Restore FPSR/FPCR and return ---
    "msr FPSR, %2\n"
    "msr FPCR, %3\n"
    "ret\n"
    :
    : "r"(gpr_ptr), "r"(neon_ptr),
      "r"(fpsr_val), "r"(fpcr_val)
    : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7",
      "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
      "x16", "x17", "x30", "memory");
}

#endif  // V8_TARGET_ARCH_ARM64

}  // namespace internal
}  // namespace v8

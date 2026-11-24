// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the
// following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

'use strict';

// Phase-2 Stage 1 ABI proof. TEST-ONLY, NO V8 BUILD, NO serialization change.
//
// Proves the compiled-in per-worker execution-state routing (x25 ->
// execution_state_, keyed by isolate+OS-thread) is correct for a SINGLE worker
// executing a body, BEFORE any serializer is touched. Specifically:
//   1. the body's allocation-light integer math returns the exact sentinel
//      (x25 routing is consistent with inline-caches / deopt), and
//   2. a mid-body green_threads.yield() performs a real suspend -> resume that
//      re-arms x25 to the SAME per-worker block (not the shared
//      thread_local_top_) -- the body then returns a correct value instead of
//      segfaulting or returning garbage.
//
// Run with GT_NUM_WORKERS=1 so it cannot surface the multi-worker race; the
// guard is the ABI, not parallelism. A crash here (before Stage 2 proves
// per-worker scope) would be the forced-GC mirror race: x25 armed to the shared
// block -> context()==0 -> SEGV on resume.
//
// Green gate: exit 0 and the sentinel matches after the yield.

const g = require('green_threads');
const assert = require('node:assert/strict');

// Deterministic, allocation-light sentinel. Any hash collision here would be a
// code bug, not a flake -- the constant is opaque by construction.
function sentinel() {
  let x = 1;
  for (let i = 0; i < 999999; i++) x = (x * 1103515245 + 12345) & 0x7fffffff;
  return x;
}

// The body: compute the sentinel, yield once mid-body (forces suspend/resume on
// the per-worker block), then return. If the resume re-armed x25 to the wrong
// block, this throws / segfaults / returns a wrong value.
function body() {
  const r = sentinel();
  g.yield();
  return r;
}

const expected = sentinel();
const actual = g.spawn(body);

assert.equal(actual, expected,
  `Stage-1 ABI proof: body returned sentinel ${actual} after yield, expected ${expected}`);

console.error(`Stage-1 ABI proof OK: sentinel ${expected} correct after mid-body yield ` +
  `(GT_NUM_WORKERS=1, per-worker x25 resume verified)`);

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

// Regression test for the concurrent-GC / concurrent-compiler races against
// green-thread copy-on-write (COW) heap isolation.
//
// These two fixes superseded the single-execution-thread assumption baked into
// the CoW design (see deps/v8/src/execution/green-thread-cow.cc): the CoW model
// rewrites heap slots for a green thread's private view on the embedding
// thread and expects V8 to run its concurrent work there too. Node embeds V8
// with concurrent work enabled on separate OS worker threads, so two distinct
// races occurred while a CoW view was active.
//
//   Root cause 1 -- concurrent young-GC (scavenger) race.
//     The scavenger ran on a separate `node-V8Worker` OS thread (created under
//     Heap::ShouldUseBackgroundThreads()) and copied the set of young objects
//     pointed at by old-space. Racing the embedding thread's in-place slot
//     rewrites (CowCheckBeforeStore / COWWriteBarrierCheck), it dereferenced a
//     torn pointer and crashed with EXC_BAD_ACCESS inside
//     Scavenger::EvacuateObject <- Map::instance_size_in_words (confirmed via
//     lldb: a garbage map read off an object pulled from the remembered set).
//
//     Fix: Heap::ShouldUseBackgroundThreads() returns false while a CoW view is
//     active (via the GreenThreadCOWSuspendedConcurrentGC() hook), so every
//     concurrent-GC worker -- scavenger, mark-compact, and array-buffer
//     sweeper -- falls back to serial execution on the embedding thread.
//
//   Root cause 2 -- concurrent baseline (Sparkplug) compiler race.
//     With the GC race silenced, a second worker raced the same torn state: the
//     baseline compiler's worker threads read heap memory the embedding thread
//     was rewriting and produced a pointer that failed V8's is_int26
//     relocation invariant (Check failed: is_int26(x)) inside
//     RelocateFromDescWriteBarriers, aborting the process.
//
//     Fix: BaselineBatchCompiler::concurrent() and EnqueueSFI() return/skip
//     concurrent compilation while a CoW view is active, installing baseline
//     code serially on the embedding thread.
//
// Both crashes kill the process, so the harness fails simply: it drives high
// concurrent green-thread load over many rounds -- allocation heavy (young-GC
// pressure) and with writes to a shared captured object (CoW clones /
// remembered-set roots) and repeated compilation (baseline-compiler pressure)
// -- and requires the child process to survive every round with all returned
// Fibonacci values correct. Reverting either fix makes the child abort mid-run
// (non-zero / negative signal exit), which this test asserts against.
//
// Isolation note: green threads run under "Model A" recursive scheduling on the
// embedding thread inside this same isolate/context. That races with both
// node:test's subtest bookkeeping and the parent's exit handlers. As in
// test-green-threads-cow.js, the concurrent scenarios run in a *child* node
// process that owns the scheduler, and this test asserts on the child's exit
// code.

const common = require('../common');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// Stress parameters. CONC is the green-thread fan-out per round (a wider fan
// out means more V8 GC/compiler workers active at once and a smaller window to
// trip the races); ROUNDS repeats the fan-out so allocation churn repeatedly
// triggers young GC and compilation repeatedly drives the baseline compiler.
// These are deliberately heavy (the races are timing-probabilistic and, per the
// incident history, only reproduce under high concurrency). They are tunable
// through the environment so the suite default stays safe while operators can
// crank up exposure to chase the races.
const ROUNDS = Number(process.env.STRESS_ROUNDS || '40');
const CONCURRENCY = Number(process.env.STRESS_CONCURRENCY || '40');
const FIB_N = Number(process.env.STRESS_FIB || '20');

// Exercises the CoW write barrier (writes to a shared captured object), young
//-GC allocation pressure (a fresh 256-element object array per thread), and
// baseline-compiler pressure (repeated BigInt fib). Returns the correct fib
// value so the parent can verify every concurrent result.
const CHILD_STRESS = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');

function fib(n) {
  let a = 0n, b = 1n;
  for (let i = 0; i < n; i++) { const t = a + b; a = b; b = t; }
  return a;
}

function stressBody(parent, fibN) {
  for (let w = 0; w < 64; w++) {       // many COW writes -> many remembered-set roots for the scavenger
    parent.writes += 1;
    parent.values.push(parent.writes);
  }
  const arr = new Array(256);          // allocation pressure -> frequent young GC
  for (let i = 0; i < arr.length; i++) arr[i] = { idx: i, tag: 'gt-' + i };
  const value = fib(fibN);
  arr[0].idx = arr[0].idx + 1;         // CoW write to the local clone
  return { value };
}

const expected = fib(${FIB_N});
(async () => {
for (let round = 1; round <= ${ROUNDS}; round++) {
  const parent = { writes: 0, values: [] };
  const results = await g.spawnAll(
    Array.from({ length: ${CONCURRENCY} }, () => [stressBody, parent, ${FIB_N}]),
  );
  assert.equal(results.length, ${CONCURRENCY},
    'round ' + round + ': spawned ' + ${CONCURRENCY} + ' green threads');
  for (let i = 0; i < results.length; i++) {
    assert.equal(results[i].value, expected,
      'round ' + round + ': result #' + i + ' fib value correct');
  }
}
console.error('green_threads concurrent-race regression: all scenarios passed');
})();
`;

const res = spawnSync(process.execPath, ['-e', CHILD_STRESS], {
  encoding: 'utf8',
});

assert.equal(res.status, 0,
  `green-thread child process exited with status ${res.status}\n` +
  `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
assert.match(
  `${res.stdout}${res.stderr}`,
  /green_threads concurrent-race regression: all scenarios passed/,
);

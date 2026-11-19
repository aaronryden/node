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

// M1 architecture probe: proves that N OS worker threads can Enter the SAME
// green isolate *simultaneously*, each with a live JS frame. This exercises the
// green_thread_mode_ entry chain in Isolate::Enter/Exit -- the change that makes
// concurrent multi-mutator entry possible (see GT-OS-WORKER-POOL-PLAN.md).
//
// Each worker Enters the green isolate, runs a disjoint JS body that reaches a
// native rendezvous callback (a live JS frame), and waits there until all N
// workers have arrived. Serial (single-OS-thread) execution deadlocks at the
// rendezvous and the native side times out, so this fails fast on a build where
// concurrent entry is not actually concurrent. The probe runs in a child so a
// hang aborts the parent instead of stalling CI.
//
// The probe proves simultaneous ENTRY (the M1 milestone). Concurrent general-JS
// execution on the green heap is reserved for the all-mutator-GC work (M2); this
// bounded probe deliberately runs no heap-touching JS, it only proves entry.

const common = require('../common');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// Each scenario is a distinct worker count.
const scenarios = [4, 8, 16, 32];

// The child calls __concurrent_entry(n), which returns { count, concurrent,
// results_ok }. Exit 0 only when all N workers truly overlapped (concurrent).
const BODY = `
const g = require('green_threads');
// Under node -e the first CLI arg is process.argv[1], not process.argv[2].
const report = g.__concurrent_entry(Number(process.argv[1]));
process.exit(report.concurrent ? 0 : 1);
`;

for (const n of scenarios) {
  const res = spawnSync(process.execPath, [
    '-e',
    BODY,
    String(n),
  ], { encoding: 'utf8' });
  assert.equal(
    res.status,
    0,
    `n=${n}: ${n} OS threads did not all Enter the green isolate concurrently ` +
      `(exit ${res.status}); stderr:\n${res.stderr}`,
  );
}

console.error(
  `green_threads concurrent-entry probe: all ${scenarios.length} scenarios ` +
    `(4, 8, 16, 32 workers) passed -- concurrent multi-mutator entry proven`,
);

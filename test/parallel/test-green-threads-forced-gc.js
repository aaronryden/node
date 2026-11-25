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

// Acceptance gate for the all-mutator stop-the-world GC handshake (invariant
// I3) in src/green_threads.cc / deps/v8/src/execution/green-thread-cow.cc.
//
// The probe (__m2_probe) spawns N OS workers that share ONE green isolate and
// run a body each. With GT_M2_FORCE_GC=both, the coordinator announces a GC,
// every worker parks at its entry safepoint having published its roots, the
// coordinator runs a forced young + full collection while all N workers are
// quiescent, then resumes them. The report's post_gc_consistent flag proves
// every worker's per-rooted marker survived that collection intact.
//
// Green threads use "Model A" recursive scheduling on the embedding thread,
// which races node:test's subtest bookkeeping and the parent's exit handlers, so
// -- as in test-green-threads-cow.js, test-green-threads-shared-heap.js, and
// test-green-threads-concurrency.js -- the scenario runs in a *child* node
// process that owns the scheduler. The shared-green-isolate path and the
// forced-GC knob are pinned on the child explicitly rather than depending on
// the parent's environment.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// A pure body: allocate a pool of throwaway objects (enough real allocation to
// make the forced collection actually evacuate/compact), then return a stable
// value derived from the work. Bodies take no arguments (M2Probe invokes them
// with an empty argv) and must not close over anything -- RelocateFunction
// reparses each body's source onto the green isolate, so closures and even
// parameters are unavailable; only literal values survive. `seed` is therefore
// baked into the body source as a numeric literal at creation time, which is
// why each body stays distinct after reparsing.
function makeBody(seed) {
  const bodySrc = `
    const pool = new Array(64);
    for (let k = 0; k < pool.length; k++) {
      const value = ${seed} * 1000 + k;
      pool[k] = { seed: ${seed}, k, v: value };
    }
    let sum = 0;
    for (const e of pool) sum += e.v;
    return { seed: ${seed}, count: pool.length, sum, last: pool[pool.length - 1] };
  `;
  // `new Function` yields a body with NO closure and NO parameters -- exactly
  // what Reparse/RelocateFunction expects on the green isolate.
  return new Function(bodySrc);
}

const CHILD = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');

const bodies = [
  ${String(makeBody(1))},
  ${String(makeBody(2))},
  ${String(makeBody(3))},
  ${String(makeBody(4))},
];

const report = g.__m2_probe(4, ...bodies);

const results = report.results || [];
assert.equal(results.length, 4, 'probe returned one result per worker');
assert.ok(Array.isArray(results), 'results must be an array');

assert.ok(report.all_ok, 'all workers must run and return a value');

// The whole point of this test: the forced young + full collection ran while all
// four workers were parked with their roots published. If the handshake failed
// (a worker was mid-body, or roots were not published), the collection would
// either crash or reclaim a rooted marker; post_gc_consistent captures that.
assert.ok(report.post_gc_consistent,
  'every rooted per-worker marker must survive the forced GC');

console.error('green_threads forced-GC handshake: all workers active, collection survived, post_gc_consistent');
`;

const res = spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: {
    ...process.env,
    GT_SHARED_HEAP: '1',
    GT_M2_FORCE_GC: 'both',
  },
});

assert.equal(res.status, 0,
  `green-thread child process exited with status ${res.status}\n` +
  `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);

assert.match(
  `${res.stdout}${res.stderr}`,
  /green_threads forced-GC handshake: all workers active, collection survived, post_gc_consistent/,
);

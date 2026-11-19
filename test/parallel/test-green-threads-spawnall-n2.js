// Copyright 52 the Node.js authors. All rights reserved.
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
// The SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
// USE OR OTHER DEALINGS IN THE SOFTWARE.

'use strict';

// Regression test for the deterministic SIGSEGV that g.spawnAll() trips when it
// drives two or more green threads in a single ScheduleLoop window under
// GT_SHARED_HEAP=1. A single green thread (g.spawn, or spawnAll with one task)
// runs fine; the moment a second body is dispatched on the same shared green
// isolate the second body's fn->Call dereferences a torn green-isolate native
// context (execution.cc: native_context()->script_execution_callback) and the
// process is signal-killed (exit 139), with no stdout/stderr because a
// signal-killed child loses its buffers (the child-isolation trap seen in the
// CoW debugging playbook).
//
// Like test-green-threads-cow.js / test-green-threads-shared-heap.js the
// scenario runs in a *child* node process that owns the scheduler; the parent
// asserts strictly on the child's exit state, because the crash is deterministic
// (reproduced exit 139 on every run) and the red form is a non-zero / SIGSEGV
// status. After the native fix lands the child returns the computed results and
// exits 0, which this test also asserts so it guards correctness, not just
// "did not die".

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

const CHILD = `
'use strict';
const g = require('green_threads');
const add = (a, b) => a + b;
// Two DIFFERENT bodies in one spawnAll: if the fault were re-using a single
// JSFunction this would still differ, and it crashes either way -- the bug is
// in scheduling/driving 2+ green threads on the shared green isolate.
const mul = (a, b) => a * b;
const r = g.spawnAll([[add, 2, 3], [mul, 4, 5]]);
if (!Array.isArray(r) || r.length !== 2 || r[0] !== 5 || r[1] !== 20) {
  throw new Error('unexpected result ' + JSON.stringify(r));
}
console.log('n2 spawnAll ok: ' + JSON.stringify(r));
`;

const runOnce = () => spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: { ...process.env, GT_SHARED_HEAP: '1' },
});

// The crash is deterministic, but run the scenario a few times so a flaky
// relocation race (if one were hiding here) surfaces rather than masquerading
// as a pass on a lucky run.
let last;
for (let i = 0; i < 3; i++) {
  last = runOnce();
  assert.equal(last.signal, null,
    `green-thread spawnAll(n=2) child was signal-killed (${last.signal}) ` +
    `on run ${i + 1}\n` +
    `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
  assert.equal(last.status, 0,
    `green-thread spawnAll(n=2) child exited with status ${last.status}\n` +
    `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
}

assert.match(`${last.stdout}${last.stderr}`, /n2 spawnAll ok: \[5,20\]/);

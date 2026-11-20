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

// Closure-transfer regression test for green-thread shared-heap bodies.
//
// Before the fix, a green body that references captured module-scope state
// (`vals`, a module `const` array -> scenario 4) or the native `green_threads`
// binding (`g` -> scenario 5) threw ReferenceError at entry, because the body's
// source is reparsed onto the green isolate with an empty scope and the free
// variables no longer resolve. This asserts that captured values are lifted to
// the green global by name (scenario 4) and that the native binding is callable
// from green (scenario 5), plus CoW isolation and a locals-only regression guard.
//
// Green-thread scenarios run in a *child* node process that owns the scheduler
// (see test-green-threads-cow.js / test-green-threads-spawnall-n2.js): "Model A"
// recursive scheduling races with node:test subtest bookkeeping and the process
// exit handlers. The parent asserts on the child's exit state -- a body that
// throws exits non-zero (a clean ReferenceError, signal null), so the failure is
// visible here (not signal-killed and lost, the spawnSync child-isolation trap).

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

const CHILD = `
'use strict';
const g = require('green_threads');

// Captured module-scope state the green bodies must see (scenarios 4 & 5).
const vals = [2, 3];
let count = 0; count = 7;

// (A) scenario 4 core: read a captured module \`const\` array.
{
  const row = g.spawn(() => {
    const r = {};
    for (let i = 0; i < vals.length; i++) r['k' + i] = vals[i];
    return r;
  });
  if (row.k0 !== 2 || row.k1 !== 3) throw new Error('A: row = ' + JSON.stringify(row));
}

// (B) captured let binding: a fresh per-spawn snapshot sees the latest value.
{
  const v = g.spawn(() => count);
  if (v !== 7) throw new Error('B: got ' + v);
}

// (C) mutation inside the body works on the green view (CoW isolation): returns 8.
{
  const v = g.spawn(() => ++count);
  if (v !== 8) throw new Error('C: got ' + v);
}

// (D) scenario 5 core: the native binding is callable from green.
{
  const t = g.spawn(() => typeof g.addrOf);
  if (t !== 'function') throw new Error('D: got ' + t);
}

// (E) scenario 5 CoW isolation (hardened). A heap object created in the parent
// program is captured by the body as a live alias; a write through that alias
// inside the green view must be copy-on-write -- the embedding original stays
// immutable while the green view observes its own write. The isolation is
// checked in the PARENT (outside the green view), so a barrier that applied the
// write in place and mutated the shared object fails here.
//
// This replaces the prior shape, which compared addrOf against cowCloneOf. In
// the value-capture model captured/argument heap values are copied into the
// green view (there is no persistent embedding-to-green alias for the barrier
// to clone), so cowCloneOf returns undefined and s !== c passed trivially on
// NaN without ever exercising isolation. The assertion runs in the parent so
// assert -- an object this path does not lift -- is never captured by the body.
{
  const shared = { v: 1 };
  const greenSaw = g.spawn(() => {
    shared.v = 999; // write through the captured alias -> copy-on-write clone
    return shared.v; // the green view must observe its own write
  });
  if (greenSaw !== 999) {
    throw new Error('E: green view did not apply its write, got ' + greenSaw);
  }
  if (shared.v !== 1) {
    throw new Error('E: embedding object not isolated after green write, v=' + shared.v);
  }
}

// (F) regression guard: a locals-only body still works.
{
  const v = g.spawn(() => 42);
  if (v !== 42) throw new Error('F: got ' + v);
}

console.log('green_threads closure repro ok');
`;

const runOnce = () => spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: { ...process.env, GT_SHARED_HEAP: '1' },
});

// Run a few times so a flaky relocation race (if one were hiding here) surfaces
// rather than masquerading as a pass on a lucky run.
let last;
for (let i = 0; i < 3; i++) {
  last = runOnce();
  assert.equal(last.signal, null,
    `green_threads closure child was signal-killed (${last.signal}) ` +
    `on run ${i + 1}\n` +
    `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
  assert.equal(last.status, 0,
    `green_threads closure child exited with status ${last.status} ` +
    `(expected 0) on run ${i + 1}\n` +
    `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
}

assert.match(`${last.stdout}${last.stderr}`, /green_threads closure repro ok/);

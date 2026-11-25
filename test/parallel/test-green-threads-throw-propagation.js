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

// Regression gate for green-thread throw-mode exception propagation.
//
// A green body's uncaught exception must reach the enclosing JS try/catch
// (not be swallowed into an undefined spawn() result), without crashing the
// process, and without leaking into a subsequent task on a shared green
// isolate. It also runs serial (no GT_SHARED_HEAP) to cover both routing paths.

const { spawnSync } = require('node:child_process');
const assert = require('node:assert/strict');

const CHILD = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');

// spawn() ALWAYS returns a Promise. A body that throws yields a *rejected*
// Promise (not a synchronous throw); a returning body yields a *resolved* one.
// cap() awaits the spawn result, recording either the resolved value or the
// rejection error -- preserving the original "does the body's exception reach JS
// faithfully?" intent under the new Promise contract.
(async () => {
  let thrown = [];
  async function cap(label, fn) {
    try {
      const r = await fn();
      thrown.push([label, null, r]);
    } catch (e) {
      thrown.push([label, e, undefined]);
    }
  }

  // 1. An uncaught green throw propagates as a faithful Error (via rejection).
  await cap('throw-Error', () => g.spawn(() => { throw new Error('boom-from-green'); }));

  // 2. A body that throws keeps its error class on rejection.
  await cap('throw-TypeError', () => g.spawn(() => { throw new TypeError('bad-type'); }));

  // 3. A non-Error throw propagates faithfully (primitive).
  await cap('throw-string', () => g.spawn(() => { throw 'plain-string-throw'; }));

  // 4. A body that catches internally returns normally.
  await cap('catch-inside', () => g.spawn(() => {
    try { throw new Error('inner'); }
    catch (e) { return 'caught-inside:' + e.message; }
  }));

  // 5. A normal return is unaffected by surrounding throws.
  await cap('return-42', () => g.spawn(() => 42));

  // 6. No leakage: after throws that propagated to JS, the next task runs clean.
  const after = () => g.spawn(() => {
    return { nums: [1, 2, 3], s: 'hi', f: 2.5, n: 1000n };
  });
  await cap('after-leak-return', after);

  // Assertions all run on the embedding side (assert is not a green-global).
  for (const [label, err, val] of thrown) {
    if (label === 'throw-Error') {
      assert.ok(err instanceof Error, 'throw-Error: propagated as Error (rejection)');
      assert.equal(err.message, 'boom-from-green', 'throw-Error: message preserved');
    } else if (label === 'throw-TypeError') {
      assert.ok(err instanceof TypeError, 'throw-TypeError: class preserved');
      assert.equal(err.message, 'bad-type', 'throw-TypeError: message preserved');
    } else if (label === 'throw-string') {
      assert.equal(err, 'plain-string-throw', 'throw-string: primitive propagated');
      assert.ok(!(err instanceof Error), 'throw-string: not wrapped as Error');
    } else if (label === 'catch-inside') {
      assert.equal(val, 'caught-inside:inner', 'catch-inside: returned normally');
    } else if (label === 'return-42') {
      assert.equal(val, 42, 'return-42: returned normally');
    } else if (label === 'after-leak-return') {
      assert.deepEqual(val, { nums: [1, 2, 3], s: 'hi', f: 2.5, n: 1000n },
        'after-leak-return: no leaked exception, structured value intact');
    }
  }

  console.log('green_threads throw-propagation repro ok');
})();
`;

const runOnce = (env) => spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: { ...process.env, ...env },
});

// Shared-heap (isolated green isolate) AND serial, each a few times so a
// relocation / exception-capture race surfaces instead of masquerading as a
// pass on a lucky run.
const configs = [
  { label: 'shared-heap', env: { GT_SHARED_HEAP: '1' } },
  { label: 'serial', env: {} },
];

for (const { label, env } of configs) {
  let last;
  for (let i = 0; i < 3; i++) {
    last = runOnce(env);
    assert.equal(last.signal, null,
      `green_threads throw-propagation child was signal-killed (${last.signal}) ` +
      `${label} run ${i + 1}\n` +
      `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
    assert.equal(last.status, 0,
      `green_threads throw-propagation child exited with status ${last.status} ` +
      `${label} (expected 0) on run ${i + 1}\n` +
      `--- stdout ---\n${last.stdout}\n--- stderr ---\n${last.stderr}`);
  }
  assert.match(`${last.stdout}${last.stderr}`,
    /green_threads throw-propagation repro ok/);
}

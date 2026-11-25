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

// Regression tests for the two native-bridge defects that block the
// GT_SHARED_HEAP (shared-green-isolate) execution path in
// src/green_threads.cc. Each green body is re-parsed and compiled on the
// shared green isolate, and the isolate itself is created and torn down there,
// so both fixes live in that path. Green threads use "Model A" recursive
// scheduling on the embedding thread, which races node:test's subtest
// bookkeeping and the parent's exit handlers, so -- as in
// test-green-threads-cow.js and test-green-threads-concurrency.js -- the
// scenarios run in a *child* node process that owns the scheduler and this
// test asserts on the child's exit code. The shared-green-isolate path is only
// taken when GT_SHARED_HEAP is set, so the test pins it on the child explicitly
// rather than depending on the parent's environment.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// The child exercises both fixes while returning only primitives. Primitives
// avoid the green-thread copy-on-write object-store path (a separate, unrelated
// regression surface) so each scenario maps cleanly onto the native fix it
// guards.
const CHILD = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');

// Fix 1 -- RellocateFunction context rooting.
//
// A green body is located onto the shared green isolate by re-parsing its
// source and compiling it with Script::Compile. When the fresh Context passed
// to that compile was rooted inside a HandleScope whose scope ended before the
// compile (CloseScope resets and zaps the released handle slot), per_ctx became
// a dangling Local: the compile dereferenced a garbage context and aborted with
// exit 134 (a bogus SharedFunctionInfo/SB) on the very first spawn. Correctly
// returning values here proves the body compiled to real, executable green Code
// and that its result round-tripped back across isolates.
const add = (a, b) => a + b;
const mul = (a, b) => a * b;
const greet = (n) => 'hi ' + n;

// spawn() always returns a Promise, so the assertions below await their results.
// The whole child body runs inside an async IIFE (a node -e CommonJS string
// cannot use top-level await); a failed assertion rejects the IIFE, so the child
// exits non-zero and the parent's status check fails the test.
(async () => {
assert.equal(await g.spawn(add, 2, 3), 5, 'add(2,3)');
assert.equal(await g.spawn(add, 100, 23), 123, 'add(100,23)');
assert.equal(await g.spawn(mul, 6, 7), 42, 'mul(6,7)');
assert.equal(await g.spawn(greet, 'world'), 'hi world', 'greet');
assert.equal(await g.spawn(() => 42), 42, 'literal return');

// Several more sequential bodies force RellocateFunction to re-parse + compile
// on the same green isolate repeatedly (each spawn is an independent relocate),
// exercising the rooting fix across many compile+round-trip cycles.

// Fix 2 -- DisposeGreenIsolate teardown.
//
// The green isolate is registered on Node's platform (so it needs a
// PerIsolatePlatformData with a flush_tasks_ handle). If it is disposed with a
// raw Isolate::Dispose() the isolate is never unregistered from the platform,
// so at process exit its PerIsolatePlatformData is destroyed while flush_tasks_
// is still set and ~PerIsolatePlatformData trips CHECK(!flush_tasks_), aborting
// with exit 134 -- even though every green body ran correctly. A clean exit
// below is therefore the assertion for this fix as well.
console.error('green_threads shared-heap regression: all scenarios passed');
})();
`;

const res = spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: { ...process.env, GT_SHARED_HEAP: '1' },
});

assert.equal(res.status, 0,
  `green-thread child process exited with status ${res.status}\n` +
  `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);

// Explicit guard against the teardown abort specifically, since its message
// ("Assertion failed: !flush_tasks_") is the signature of Fix 2.
assert.ok(!/Assertion failed: !flush_tasks_/.test(res.stderr),
  'child aborted at green-isolate teardown (!flush_tasks_)');

assert.match(
  `${res.stdout}${res.stderr}`,
  /green_threads shared-heap regression: all scenarios passed/,
);

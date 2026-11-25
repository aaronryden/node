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

// Regression test for the shared-heap green-async context desync SIGSEGV.
//
// Under GT_SHARED_HEAP=1 with GT_NUM_WORKERS > 1, an async green body that
// `await`s real I/O and then returns an array literal crashes its worker:
//
//   GreenGtRead -> await parks the body -> the drive loop resumes it by running
//   the body's microtask queue on an OS worker. The body body is
//   `async (pth, ei) => { const c = await gtRead(pth, 'utf8'); return [c, ei]; }`
//   -- the `return [c, ei]` is constructed by
//   Factory::NewJSArrayWithUnverifiedElements, which calls
//   Isolate::raw_native_context -> Context::native_context(). Under the
//   desync, the shared IsolateData execution_state_base_ mirror (which the
//   JS-entry microtask runner re-arms the per-worker execution-state register,
//   x25, from) is overwritten mid-arm by a concurrent worker's
//   GTGreenThreadRefreshMirror(). So this worker's generated microtask code arms
//   x25 to a sibling's ExecutionState block while its own C++ path
//   (Isolate::context / raw_native_context) reads its own block: SetCurrentContext
//   writes the sibling's block and the context_ field this thread reads is
//   garbage (0xffffffffffffffff) -> SIGSEGV inside
//   PromiseFulfillReactionJob / AsyncFunctionAwaitResolveClosure on a
//   `node-V8Worker` thread.
//
// The fix holds the coarse green-mutator lock around the driver's per-body
// mirror-touch + microtask checkpoint so only one worker arms the shared mirror
// at a time (serialized worker execution -- the "safe correctness baseline N
// workers run behind" the true-parallel milestone). Reverting the fix makes the
// child abort mid-run (signal / non-zero exit); the test asserts that does not
// happen over many concurrent rounds.
//
// Isolation note: like the other green-threads acceptance tests, the load runs
// in a *child* process (Model A recursive scheduling races node:test bookkeeping),
// and this test asserts on the child's exit status.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// W: async bodies per round (== the number of OS workers, so every worker runs
// one body's checkpoint each round -- maximum contention on the shared mirror).
// ROUNDS: repeated concurrent rounds; the race is timing-probabilistic, so many
// rounds make the failure (when the fix is reverted) reproduce deterministically.
const W = Number(process.env.GT_ASYNC_W || '8');
const ROUNDS = Number(process.env.GT_ASYNC_ROUNDS || '40');
const PAYLEN = 256;
const PAYLOAD = (i) => 'payload-' + i + '-' + 'z'.repeat(PAYLEN);

const CHILD = [
  "'use strict';",
  "const g = require('green_threads');",
  "const fs = require('fs');",
  "const os = require('os');",
  "const path = require('path');",
  "const assert = require('assert/strict');",
  "",
  "const W = " + W + ";",
  "const ROUNDS = " + ROUNDS + ";",
  "const PAYLEN = " + PAYLEN + ";",
  "",
  "const dir = os.tmpdir();",
  "const pths = [];",
  "for (let i = 0; i < W; i++) {",
  "  const p = path.join(dir, 'gt-async-reg-' + process.pid + '-' + i + '.txt');",
  "  pths.push(p);",
  "  fs.writeFileSync(p, 'payload-' + i + '-' + 'z'.repeat(PAYLEN), 'utf8');",
  "}",
  "",
  "// An async body whose return value is an array literal built AFTER an await;",
  "// constructing that literal in the resumption microtask is the crash trigger.",
  "async function makeBody() {",
  "  return async (pth, ei) => {",
  "    const c = await gtRead(pth, 'utf8');",
  "    return [c, ei];",
  "  };",
  "}",
  "",
  "(async () => {",
  "  for (let round = 1; round <= ROUNDS; round++) {",
  "    const body = await makeBody();",
  "    const outer = await g.spawnAll(pths.map((p, i) => [body, p, i]));",
  "    const results = await Promise.all(outer);",
  "    for (let i = 0; i < W; i++) {",
  "      const r = results[i];",
  "      assert.ok(Array.isArray(r), 'round ' + round + ' result #' + i + ' is array literal');",
  "      assert.equal(r[1], i, 'round ' + round + ' closure index preserved (context not torn)');",
  "      assert.equal(r[0], 'payload-' + i + '-' + 'z'.repeat(PAYLEN), 'round ' + round + ' read value correct');",
  "    }",
  "  }",
  "  fs.unlinkSync(pths[0]);",
  "  console.error('green_threads async-context-regression: all ' + ROUNDS + ' rounds passed');",
  "})();",
].join('\n');

const res = spawnSync(process.execPath, ['-e', CHILD], {
  encoding: 'utf8',
  env: {
    ...process.env,
    GT_SHARED_HEAP: '1',
    GT_NUM_WORKERS: String(W),
  },
  timeout: 120000,
});

assert.equal(res.status, 0,
  `green-async shared-heap child process exited with status ${res.status} ` +
  `(SIGSEGV/context-desync regression) -- ` +
  `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);

assert.match(
  `${res.stdout}${res.stderr}`,
  /green_threads async-context-regression: all [0-9]+ rounds passed/,
);

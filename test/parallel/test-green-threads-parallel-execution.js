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

// Acceptance gate for TRUE parallel green-thread execution on parallel OS
// threads (M:N like Go's scheduler). Green threads must not merely interleave
// cooperatively on a single real thread -- N independent bodies on N OS workers
// must finish in roughly 1/N of the single-thread wall time.
//
// Today this fails: bodies run on one shared green isolate under the
// GreenThreadCowSharedWriterScope, so green wall time ~= serial baseline
// (speedup ~1.0). Once each worker runs its body on its own execution state
// with the serializer released, speedup climbs above the threshold below.
//
// Bodies are allocation-light integer math so they stay clear of the CoW
// young-GC movement race (allocation under parallel GC is what causes the
// intermittent SIGSEGVs) -- the gate measures CPU parallelism, not allocation.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// Child: N compute-heavy, allocation-light bodies on the OS-thread pool. Runs
// to completion in a standalone process (Model A recursive scheduling races the
// parent's node:test bookkeeping), printing one JSON object on stdout.
const CHILD = [
  "'use strict';",
  "const g = require('green_threads');",
  "const N = Number(process.env.GT_N || '4');",
  "const CHUNKS = 4;",
  "const ITERS = 120000000;",
  // Pure integer work: no object/BigInt/array allocation -> no CoW movement.
  "function work(n){",
  "  let x = 1;",
  "  for (let i = 0; i < n; i++) { x = (x * 1103515245 + 12345) & 0x7fffffff; }",
  "  return x;",
  "}",
  "function body(){",
  "  let a = 0;",
  "  for (let c = 0; c < CHUNKS; c++) a = work(ITERS);",
  "  return a;",
  "}",
  "// Serial baseline: N bodies * CHUNKS chunks, back-to-back on one thread.",
  "const t0 = performance.now();",
  "let base = 0;",
  "for (let j = 0; j < N; j++) { for (let c = 0; c < CHUNKS; c++) base = work(ITERS); }",
  "const baseMs = performance.now() - t0;",
  "// Green fan-out: N bodies on the OS-thread pool.",
  "const t1 = performance.now();",
  "const res = g.spawnAll(Array.from({ length: N }, () => [body]));",
  "const greenMs = performance.now() - t1;",
  "console.log(JSON.stringify({",
  "  N, CHUNKS, ITERS,",
  "  serial_ms: +baseMs.toFixed(1),",
  "  green_ms: +greenMs.toFixed(1),",
  "  speedup: +(baseMs / greenMs).toFixed(3),",
  "  acc_ok: res.every((a) => a === work(ITERS)),",
  "}));",
].join('\n');

function runChild(n) {
  const res = spawnSync(process.execPath, ['-e', CHILD], {
    encoding: 'utf8',
    env: { ...process.env, GT_SHARED_HEAP: '1', GT_NUM_WORKERS: String(n), GT_N: String(n) },
    timeout: 120000,
  });
  assert.equal(res.status, 0,
    `child exited with status ${res.status}\n` +
    `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
  const line = (res.stdout || '').trim().split('\n').find((l) => l.startsWith('{'));
  assert.ok(line, 'child printed a JSON summary');
  return JSON.parse(line);
}

// Small-N: correctness first, then require wall-clock parallelism.
{
  const r = runChild(4);
  assert.equal(r.N, 4, 'small-N run used 4 bodies');
  assert.ok(r.acc_ok, `all bodies must compute correct results (got speedup ${r.speedup}, green_ms ${r.green_ms})`);
  assert.ok(r.speedup > 1.4,
    `4 bodies on 4 OS threads must show >1.4x wall-clock speedup (got ${r.speedup}x, ` +
    `serial_ms=${r.serial_ms}, green_ms=${r.green_ms}) -- bodies are running serially`);
  console.error(`parallel-execution small-N: ${r.N} bodies, speedup=${r.speedup}x, green_ms=${r.green_ms}`);
}

// Big-N: scale the speedup toward N, proving the parallelism is real, not a fluke
// of a single warm worker.
{
  const r = runChild(12);
  assert.equal(r.N, 12, 'big-N run used 12 bodies');
  assert.ok(r.acc_ok, `all 12 bodies must compute correct results (speedup ${r.speedup})`);
  assert.ok(r.speedup > 1.8,
    `12 bodies on 12 OS threads must show >1.8x wall-clock speedup (got ${r.speedup}x, ` +
    `serial_ms=${r.serial_ms}, green_ms=${r.green_ms})`);
  console.error(`parallel-execution big-N: ${r.N} bodies, speedup=${r.speedup}x, green_ms=${r.green_ms}`);
}

console.error('green_threads parallel execution: N bodies finish ~1/N wall time on N OS threads');

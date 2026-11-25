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

// Acceptance gate for Item 4 (GT-OS-WORKER-POOL-PLAN.md): the atomic CAS
// bump-pointer (linear-allocation-area.h, enabled by
// SetGreenHeapConcurrentAllocationEnabled) is wired into a REAL unserialized
// worker path. With GT_SHARED_HEAP=1 GT_M2_PARALLEL=1, four OS workers claim
// bodies from the thread-safe M2ProbeWorkQueue and allocate from ONE green V8
// heap SIMULTANEOUSLY, with the green-mutator lock (g_green_mutator_lock)
// RELEASED. The CAS then hands each worker a DISJOINT young-gen bump pointer;
// before Item 4 the CAS was gated only behind the lock, so workers never truly
// allocated in parallel.
//
// Disjointness is proved structurally: each worker allocates a uniquely-shaped
// object and returns its heap address (via g.addrOf) as a decimal string. The
// four addresses must be pairwise distinct -- a broken CAS (a plain `top_ +=
// bytes` bump write) makes two workers reserve the same address, which the
// harness asserts against. The bodies use UNIQUE object-literal shapes (property
// count varies per worker) so no two workers share a cached initial Map; this
// isolates the CAS bump-pointer disjointness (Item 4) from the shared-map write
// race (Item 3), which is validated separately in the expanded test matrix.
//
// Two variants are run and BOTH must be green:
//   - CONTROL  (GT_SHARED_HEAP=1, no GT_M2_PARALLEL): CAS enabled but the
//     serializer held -> serial Model-A baseline.
//   - PARALLEL (GT_SHARED_HEAP=1 GT_M2_PARALLEL=1 GT_NUM_WORKERS=4): CAS
//     enabled AND serializer released -> true parallel CAS.
// If parallel is the only green one, CAS + Item 3 do the work; if the
// lock-held control is green and parallel is not, the shared-writer safety
// (Item 3) is incomplete. Keep both, as here.
//
// Green threads use "Model A" recursive scheduling on the embedding thread,
// which races node:test's subtest bookkeeping and the parent's exit handlers,
// so -- as in the other green-thread suites -- the scenario runs in a *child*
// node process that owns the scheduler and the parent asserts on the child's
// exit code.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// The child runs __m2_probe(4, ...bodies) and prints exactly one JSON object on
// stdout describing the four results (one entry per worker: its seed and the
// heap address of its allocation). Everything is returned as primitives (number
// + decimal string) so nothing crosses the isolate boundary through the CoW
// object-store path. Each body's object literal has a UNIQUE property count
// (seed + 2 fields), so no two workers share the same cached initial Map.
const CHILD = [
  "'use strict';",
  "const g = require('green_threads');",
  "function makeBody(seed) {",
  "  const count = seed + 2;",
  "  let fields = '';",
  "  for (let k = 0; k < count; k++) {",
  "    fields += 'p' + k + ':' + ((seed + 1) * 1000 + k) + ',';",
  "  }",
  "  const bodySrc =",
  "    'const marker = {' + fields + '};' +",
  "    'const ptr = g.addrOf(marker).toString();' +",
  "    'let sum = 0;' +",
  "    'for (let k = 0; k < ' + count + '; k++) sum += ' + ((seed + 1) * 1000) + ' + k;' +",
  "    'return [' + seed + ', ptr, sum];';",
  "  return new Function(bodySrc);",
  "}",
  "const bodies = [1, 2, 3, 4].map(makeBody);",
  "const report = g.__m2_probe(4, ...bodies);",
  "const summary = (report.results || []).map(function(r) {",
  "  // r is [seed, ptr, sum]; coerce to a stable array of primitives.",
  "  return { seed: Array.isArray(r) ? r[0] : null, ptr: Array.isArray(r) ? r[1] : null };",
  "});",
  "console.log(JSON.stringify({ all_ok: !!report.all_ok, summary: summary }));",
].join('\n');

function runVariant(label, env) {
  const res = spawnSync(process.execPath, ['-e', CHILD], {
    encoding: 'utf8',
    env: { ...process.env, ...env },
    timeout: 30000,
  });
  assert.equal(res.status, 0,
    `${label}: child exited with status ${res.status}\n` +
    `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
  const line = (res.stdout || '').trim().split('\n').find((l) => l.startsWith('{'));
  assert.ok(line, `${label}: child printed a JSON summary\n${res.stdout}`);
  const parsed = JSON.parse(line);
  assert.ok(parsed.all_ok, `${label}: all workers must run and return a value`);
  const ptrs = parsed.summary.map((s) => s.ptr);
  assert.equal(ptrs.length, 4, `${label}: probe returned one address per worker`);
  assert.ok(ptrs.every((p) => typeof p === 'string' && p.length > 0),
    `${label}: every worker must return a heap address\n${JSON.stringify(ptrs)}`);
  const distinct = new Set(ptrs);
  assert.equal(distinct.size, 4,
    `${label}: four workers must reserve pairwise-disjoint heap addresses ` +
    `(got ${JSON.stringify(ptrs)})`);
  return parsed;
}

// Control: CAS enabled but the green-mutator serializer held (serial baseline).
const control = runVariant('CONTROL (lock held, serial baseline)', {
  GT_SHARED_HEAP: '1',
});

// Parallel: CAS enabled AND serializer released -> true concurrent allocation.
const parallel = runVariant('PARALLEL (GT_M2_PARALLEL=1, 4 unserialized workers)', {
  GT_SHARED_HEAP: '1',
  GT_M2_PARALLEL: '1',
  GT_NUM_WORKERS: '4',
});

const controlPtrs = control.summary.map((s) => s.ptr).join(',');
const parallelPtrs = parallel.summary.map((s) => s.ptr).join(',');
console.error(`disjointness CONTROL  addresses=[${controlPtrs}]`);
console.error(`disjointness PARALLEL addresses=[${parallelPtrs}]`);
console.error(
  'green_threads parallel disjointness: 4 workers reserved disjoint ' +
  'young-gen pointers from one green heap (lock released)');

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

// Cooperative I/O block/resume regression for green-thread shared-heap async
// bodies.
//
// A green-async body that awaits gtRead must, once its read is submitted, park
// cooperatively (GreenScheduler::BlockGreenThread -> kBlocked) so its drive
// worker frees the OS core instead of busy-polling, and be resumed the instant
// the read finishes (ReactivateGreenThread on the libuv pool thread). Before the
// fix the drive loop slept 50us and retried (burning a core per parked body); if
// Reactivate ever failed to wake the parked worker the body would stall forever,
// the drive would hit its stall bound, and its caller Promise would never
// settle -- the child hangs and spawnSync's timeout fails it.
//
// So exercising many block/reactivate cycles (a body doing many sequential
// gtReads) is a strong regression guard, and a compute-only body sharing a
// single drive worker with a parked I/O body proves the parked body freed its
// slot without stalling the worker.
//
// Like test-green-threads-cow.js / -concurrency.js, the scenarios run in a child
// node process that owns the scheduler (Model A recursive scheduling races with
// node:test bookkeeping); the parent asserts on the child's exit state. A
// signal-killed crash is invisible to spawnSync (res.status === null), so a hang
// is detected via spawnSync's timeout rather than a crash.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

// A scratch file each async body reads repeatedly. A sequential read -> park ->
// reactivate cycle, so SEQUENTIAL reads per body drives SEQUENTIAL reactivations.
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gt-coop-io-'));
const ioPath = path.join(dir, 'io.bin');
fs.writeFileSync(ioPath, Buffer.alloc(4096, 0x61)); // 4 KiB of 'a'

const SEQUENTIAL = Number(process.env.GT_COOP_READS || '40');
const BODIES = Number(process.env.GT_COOP_BODIES || '6');

const CHILD = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');
const ioPath = ${JSON.stringify(ioPath)};
const SEQUENTIAL = ${SEQUENTIAL};
const BODIES = ${BODIES};
const BYTE_COUNT = 4096;

// Reads the file SEQUENTIAL times in a row. Each await gtRead parks this body on
// I/O (its drive worker blocks it and frees the core); resolving the read
// reactivates the worker. Completing many sequential reads therefore exercises
// block -> reactivate -> block -> ... end to end.
async function readBody(tag) {
  let total = 0;
  for (let i = 0; i < SEQUENTIAL; i++) {
    const c = await gtRead(ioPath, 'utf8');
    total += c.length;
  }
  return [tag, total];
}

// A compute-only async body that does no I/O. It must settle even while a sibling
// body is parked on I/O on the SAME drive worker (when GT_NUM_WORKERS === 1): if
// a parked body held onto its worker, this body would never be driven and the
// child would hang.
async function computeBody() {
  let a = 0;
  for (let i = 0; i < 500000; i++) a ^= (i * 2654435761) >>> 0;
  return a;
}

(async () => {
  const expected = SEQUENTIAL * BYTE_COUNT;

  // (1) Heavy block/reactivate cycling across the pool: BODIES async bodies, each
  // doing SEQUENTIAL sequential reads. Every body must settle with the correct
  // byte sum.
  {
    const rows = await g.spawnAll(
      Array.from({ length: BODIES }, (_, i) => [readBody, 'b' + i]),
    );
    const settled = await Promise.all(rows);
    assert.equal(settled.length, BODIES, 'all bodies returned');
    for (let i = 0; i < settled.length; i++) {
      assert.deepEqual(settled[i], ['b' + i, expected],
        'body b' + i + ' settled with the correct byte sum');
    }
  }

  // (2) Cooperative freeing on a single drive worker: a parked long-IO body must
  // not stop a compute-only sibling on that worker. This run is exercised with
  // GT_NUM_WORKERS=1 (see the parent); the same block also passes under more
  // workers.
  {
    const [ioRes, cmpRes] = await Promise.all([
      g.spawn(readBody, 'io'),
      g.spawn(computeBody),
    ]);
    assert.deepEqual(ioRes, ['io', expected],
      'long-IO body settled on a single worker');
    assert.equal(typeof cmpRes, 'number',
      'compute body settled on the same single worker as a parked I/O body');
  }

  console.log('green_threads cooperative-io repro ok');
})();
`;

function runChild() {
  const res = spawnSync(process.execPath, ['-e', CHILD], {
    encoding: 'utf8',
    timeout: 60000,
    env: {
      ...process.env,
      GT_SHARED_HEAP: '1',
      GT_NUM_WORKERS: String(process.env.GT_NUM_WORKERS || '4'),
      GT_COOP_READS: String(SEQUENTIAL),
      GT_COOP_BODIES: String(BODIES),
    },
  });
  assert.equal(res.status, 0,
    `child exited status ${res.status} (signal ${res.signal})\n` +
    `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
  return res;
}

// Run once at the default worker count (multi-worker correctness + no cross
// worker corruption), then once at a single worker (a parked I/O body must not
// stall a compute-only sibling on the same drive worker).
let last = runChild();
assert.match(`${last.stdout}${last.stderr}`, /green_threads cooperative-io repro ok/);

process.env.GT_NUM_WORKERS = '1';
last = runChild();
assert.match(`${last.stdout}${last.stderr}`, /green_threads cooperative-io repro ok/);

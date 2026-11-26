// Copyright Joyent, Inc. and other Node contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
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

// Regression guard for the drive-loop hang introduced while replacing the
// per-worker uv_run(NOWAIT) with a dedicated libuv pump thread.
//
// DriveGreenAsyncBodiesParallel partitions async bodies across an OS worker
// pool sharing ONE GreenScheduler. Each body blocks via GreenScheduler::
// BlockGreenThread (head-insert into blocked_list_) and is unblocked by
// DrainReactivations (scan + unlink). GreenScheduler::ReactivateGreenThread,
// called from the libuv pool thread on read completion, also holds the shared
// scheduler mutex.
//
// The bug: BlockGreenThread was called a second time on a gt whose prior
// Reactivate had already been drained+unlinked by ANOTHER worker, so its state
// was still kBlocked. Head-inserting an already-blocked gt makes blocked_list_
// head point at itself (a self-cycle). Every subsequent DrainReactivations scan
// for a DIFFERENT body then loops forever on that self-cycle, holding the
// scheduler mutex -- which starves the libuv pool threads (ReactivateGreenThread
// blocks on the mutex) so reads never complete, and the other drive worker too.
// The caller Promise never settles; spawnSync's timeout fails the child.
//
// This test reproduces it with GT_NUM_WORKERS=2 (where the multi-worker
// drain/block race is most likely) plus GT=1 (sanity): a heavy block/reactivate
// workload -- many bodies each doing many sequential gtReads -- so the shared
// scheduler sees thousands of head-insert / scan / unlink cycles. Any self-cycle
// formed by a double-block stalls the drive and the child times out.
//
// Like cooperative-io.js, scenarios run in a child node process that owns the
// scheduler; a hang is detected via spawnSync's timeout rather than a crash.

const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gt-pump-drive-'));
const ioPath = path.join(dir, 'io.bin');
fs.writeFileSync(ioPath, Buffer.alloc(4096, 0x61)); // 4 KiB of 'a'

// Tunables. BODY_COUNT * SEQUENTIAL reads drives block/reactivate cycles; under
// GT=2 with several workers these cycles contend on the shared scheduler, which
// is exactly what triggers the double-block self-cycle.
const BODY_COUNT = Number(process.env.GT_PUMP_BODIES || '6');
const SEQUENTIAL = Number(process.env.GT_PUMP_READS || '60');
const BYTE_COUNT = 4096;

const CHILD = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');
const ioPath = ${JSON.stringify(ioPath)};
const BODY_COUNT = ${BODY_COUNT};
const SEQUENTIAL = ${SEQUENTIAL};
const BYTE_COUNT = ${BYTE_COUNT};

async function readBody(tag) {
  let total = 0;
  for (let i = 0; i < SEQUENTIAL; i++) {
    const c = await gtRead(ioPath, 'utf8');
    total += c.length;
  }
  return [tag, total];
}

(async () => {
  const expected = SEQUENTIAL * BYTE_COUNT;
  // All bodies run concurrently across the worker pool; every one must settle
  // with the correct byte sum. A blocked_list_ self-cycle stalls the drive and
  // the caller Promise never settles -> child hangs -> parent times out.
  const rows = await g.spawnAll(
    Array.from({ length: BODY_COUNT }, (_, i) => [readBody, 'b' + i]),
  );
  const settled = await Promise.all(rows);
  assert.equal(settled.length, BODY_COUNT, 'all bodies returned');
  for (let i = 0; i < settled.length; i++) {
    assert.deepEqual(settled[i], ['b' + i, expected],
      'body b' + i + ' settled with the correct byte sum');
  }
  console.log('green_threads async-drive pump repro ok');
})();
`;

function runChild() {
  const res = spawnSync(process.execPath, ['-e', CHILD], {
    encoding: 'utf8',
    timeout: 30000,
    env: {
      ...process.env,
      GT_SHARED_HEAP: '1',
      GT_NUM_WORKERS: String(process.env.GT_NUM_WORKERS || '2'),
      GT_PUMP_READS: String(SEQUENTIAL),
    },
  });
  assert.equal(res.status, 0,
    `child exited status ${res.status} (signal ${res.signal})\n` +
    `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
  return res;
}

// Primary guard: GT_NUM_WORKERS=2 is where the drain/block race forms the
// self-cycle. Run it at GT=1 too (cooperative-io already covers GT=1 heavily,
// but a single pass here keeps the two tests independent).
const counts = [
  Number(process.env.GT_WORKER_SWEEP_1) || 2,
  Number(process.env.GT_WORKER_SWEEP_2) || 1,
];
for (const w of counts) {
  process.env.GT_NUM_WORKERS = String(w);
  runChild();
}

// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Example: green threads with copy-on-write (COW) heap isolation.
//
//   node examples/green_threads.js
//
// Green threads run cooperatively on the embedding thread (recursive entry
// calls) and each owns an isolated view of the heap. This example shows both:
//   1. Cooperative scheduling  - several bodies interleave via yield().
//   2. Copy-on-write isolation  - a clone is created lazily on first write.

const green_threads = require('green_threads');

// ---------------------------------------------------------------------------
// 1. Cooperative scheduling
// ---------------------------------------------------------------------------
// A body runs until it returns or calls green_threads.yield(). When it yields,
// the scheduler runs the next body on the same thread; control returns here
// only after that other body yields or completes. Because every body runs on
// this one real thread, the ordering of the prints below is the scheduling
// order. Each body pushes a marker per step and yields so the others interleave.
function heartbeat(label) {
  for (let step = 0; step < 3; step++) {
    process.stdout.write(`${label} step ${step}  `);
    green_threads.yield();
  }
  return `${label} done`;
}

console.log('== cooperative scheduling ==');
const order = green_threads.spawnAll([
  [heartbeat, 'A'],
  [heartbeat, 'B'],
  [heartbeat, 'C'],
]);
console.log();
console.log('results (in task order):', order.join(', '));

// ---------------------------------------------------------------------------
// 2. Copy-on-write heap isolation
// ---------------------------------------------------------------------------
// A green thread starts with a shared view of the heap. The first time it
// stores a shared object into one of its own containers, the COW write barrier
// clones it: writes to the clone are invisible to the main thread (and to
// other threads), and writes the main thread does stay shared. addrOf reports
// the shared (main-heap) address; cowCloneOf reports the private copy, or
// undefined while the object is still shared.
function cowBody(shared) {
  // Nothing written yet -> no private copy.
  const clonedBeforeWrite = green_threads.cowCloneOf(shared);
  console.log('\n== copy-on-write isolation ==');
  console.log('cowCloneOf (before write):', clonedBeforeWrite);

  // Store the shared object into our own container. This triggers the clone:
  // from here on, `own.payload` is this thread's private copy.
  const own = { payload: shared };

  // Mutate the private copy only.
  own.payload.value += 1000;

  const sharedAddr = green_threads.addrOf(shared);
  const privateAddr = green_threads.cowCloneOf(shared);
  console.log('addrOf (shared main-heap):      ', sharedAddr);
  console.log('cowCloneOf (private copy):      ', privateAddr);
  console.log('isolated (cloned != shared):    ', privateAddr !== sharedAddr);
  console.log('thread sees own value:          ', own.payload.value);

  return { sharedAddr, privateAddr };
}

const shared = { value: 42 };
const cw = green_threads.spawn(cowBody, shared);

// The main thread never wrote to `shared.value`, so it still sees 42.
console.log('main thread still sees value:   ', shared.value);
console.log('done.');

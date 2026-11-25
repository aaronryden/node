// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Green threads integration.
//
// This module exposes the V8 green-thread scheduler together with its
// copy-on-write (COW) heap isolation. It is a thin, documented wrapper over
// the native binding, which you can also reach directly via
// `internalBinding('green_threads')`.
//
// Execution model
// ---------------
// Green threads are cooperative and run on the embedding thread using recursive
// entry calls (no setjmp/longjmp, no OS threads). A body runs until it either
// returns or calls `green_threads.yield()`. When a body yields, the scheduler
// runs the next green thread on the same C++ stack; control returns to the
// yielding body only after that other thread yields or completes. This lets
// many threads interleave their work while sharing a single real thread.
//
// Copy-on-write isolation
// -----------------------
// Every green thread also owns an isolated view of the heap. Objects that a
// thread never writes to remain shared with the main thread and with other
// threads. The moment a thread stores a shared object into one of its own
// containers, that object is cloned for the thread: subsequent writes to the
// clone are invisible to the main thread, and writes the main thread does are
// invisible to the thread. Use `addrOf` and `cowCloneOf` to observe this:
// `addrOf` reports the shared (main-heap) address, while `cowCloneOf` reports
// the private copy a thread holds once (and if) it has cloned it.
//
// The public surface below maps directly onto the native functions.

'use strict';

const native = internalBinding('green_threads');


/**
 * Run a single green thread body to completion and return a Promise.
 *
 * spawn() ALWAYS returns a Promise, regardless of body type:
 *  - A sync body runs green (on the green isolate/heap); its Promise resolves
 *    with the body's return value (or rejects if the body throws).
 *  - An async body (an async function) runs GREEN on the green isolate in
 *    shared-heap mode (its `await` parks that green thread cooperatively while
 *    it drives real embedding I/O through the green global `gtRead`), or on the
 *    embedding isolate in serial mode (where Node's async I/O bindings are
 *    directly reachable). Either way its Promise resolves/rejects when the body's
 *    own Promise settles.
 *
 * @param {(...args: any[]) => any} fn The body to run.
 * @param {...any} args Arguments passed to the body as its parameters.
 * @returns {Promise<any>} Resolves to the body's result; rejects on throw.
 */
function spawn(fn, ...args) {
  if (typeof fn !== 'function')
    throw new TypeError('green_threads.spawn(fn, ...args): fn must be a function');
  try {
    return Promise.resolve(native.spawn(fn, ...args));
  } catch (err) {
    // A sync body threw synchronously (the native binding re-throws); surface
    // it as a rejection so spawn() never throws from body execution.
    return Promise.reject(err);
  }
}

/**
 * Run several green threads cooperatively to completion.
 *
 * Returns a Promise that resolves to an array of each body's result, in
 * `tasks` order. Each element is:
 *  - the body's value for a sync body (Promise-resolved), or
 *  - a Promise for an async body, resolved/rejected when that body settles.
 * Awaiting the returned Promise plus each async row (Promise.all) yields all
 * values; awaiting the outer Promise alone leaves async rows pending.
 *
 * @param {Array<[(...args: any[]) => any, ...any[]]>} tasks
 *   An array of `[fn, ...args]` tuples. Each `fn` runs as its own green
 *   thread; bodies interleave via `green_threads.yield()`.
 * @returns {Promise<Array<any>>}
 */
function spawnAll(tasks) {
  if (!Array.isArray(tasks))
    throw new TypeError('green_threads.spawnAll(tasks): tasks must be an array of [fn, ...args]');
  try {
    return Promise.resolve(native.spawnAll(tasks));
  } catch (err) {
    return Promise.reject(err);
  }
}

/**
 * Cooperatively yield control to the next green thread, returning when the
 * scheduler resumes this one. Calling yield() from the main thread (outside a
 * green thread) is a no-op.
 * @returns {undefined}
 */
function yield_() {
  return native.yield();
}

/**
 * Report the untagged heap address of an object, i.e. the shared main-heap
 * view shared by the main thread and all green threads.
 * @param {object} obj A heap object.
 * @returns {bigint} The object's main-heap address.
 */
function addrOf(obj) {
  if (obj === null || typeof obj !== 'object')
    throw new TypeError('green_threads.addrOf(obj): obj must be an object');
  return native.addrOf(obj);
}

/**
 * Report this green thread's private copy of an object.
 *
 * Only meaningful while called from inside a green thread body. Returns the
 * address of the thread's own clone once it has cloned the object on write, or
 * `undefined` while the object is still shared. This is the primary tool for
 * proving copy-on-write isolation.
 * @param {object} obj A heap object.
 * @returns {bigint | undefined} The private copy's address, or undefined.
 */
function cowCloneOf(obj) {
  if (obj === null || typeof obj !== 'object')
    throw new TypeError('green_threads.cowCloneOf(obj): obj must be an object');
  return native.cowCloneOf(obj);
}

// Exposed as a plain string constant for feature detection, driven by the
// native binding so the two cannot drift apart.
const version = String(native.version());

// TEMP diagnostic passthrough (remove before finalize).
function __dbg(...a) {
  return native.__dbg(...a);
}

// TEMP M1 probe passthrough (remove before finalize). Calls into a native
// multi-mutator entry probe on the green isolate.
function __concurrent_entry(...a) {
  return native.__concurrent_entry(...a);
}

// TEMP M2 probe passthrough. Drives n OS workers to run n self-contained
// allocating CoW bodies simultaneously in one green isolate. Returns
// { results: [...], all_ok: bool }.
function __m2_probe(...a) {
  return native.__m2_probe(...a);
}

// Returns the OS thread ID (as a hash) for the current thread.
function threadId() {
  return native.threadId();
}

/** @type {{ (fn: (...args: any[]) => any, ...args: any[]): Promise<any>,
 *          spawnAll: (tasks: any[]) => Promise<any[]>,
 *          yield: () => undefined,
 *          addrOf: (obj: object) => bigint,
 *          cowCloneOf: (obj: object) => bigint | undefined,
 *          version: string,
 *          threadId: () => number,
 *          __concurrent_entry: (n: number, fibBase: number) =>
 *              { count: number, concurrent: boolean, results_ok: boolean,
 *                expected: number[], actual: number[] } }} */
module.exports = {
  spawn,
  spawnAll,
  yield: yield_,
  addrOf,
  cowCloneOf,
  version,
  threadId,
  __dbg,
  __concurrent_entry,
  __m2_probe,
};

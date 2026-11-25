// Copyright 2025 the V8 project authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Regression test for the green-thread copy-on-write (COW) heap-isolation
// root-cause fix.
//
// Root cause (fixed): the COW deep-clone machinery attempted to re-create
// arbitrary heap objects with JSObject::NewWithMap, including exotic objects
// such as JS Map instances (instance type MAP_TYPE). Cloning those objects is
// invalid and silently corrupted the heap, so concurrent green-thread work
// returned wrong files.
//
// The fix restricts the deep copy to value types (heap numbers) and
// object/array wrappers (plain JSObject, JSArray) — exactly the types
// HeapRefTable::IsCowCloneableRootType accepts and V8's Factory::IsCloneable
// allows. Every other type (functions, proxies, Maps, Sets, strings, BigInts,
// ...) is immutable from the writer's viewpoint and shared, never cloned.
//
// This file exercises those guarantees and exits non-zero on any failure:
//   node examples/green_threads_cow_regression.js
//
// Green threads run cooperatively on the embedding thread and complete via
// spawnAll(), which integrates with the event loop and does not expose the
// tight synchronous spawn-loop scheduler quirk.

const green_threads = require('green_threads');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

let failures = 0;
function check(condition, message) {
  if (condition) {
    console.log('  ok  -', message);
  } else {
    console.error('  FAIL-', message);
    failures++;
  }
}

function fib(n) {
  let a = 0n;
  let b = 1n;
  for (let i = 0; i < n; i++) {
    const t = a + b;
    a = b;
    b = t;
  }
  return a;
}

// ---------------------------------------------------------------------------
// 1. Plain-object copy-on-write isolation still works.
// ---------------------------------------------------------------------------
console.log('[1] plain-object COW isolation');
function isolateBody(shared) {
  const own = { payload: shared };
  own.payload.value += 1000;
  return {
    privateAddr: green_threads.cowCloneOf(shared),
    selfValue: own.payload.value,
  };
}
const sharedObj = { value: 42 };
const cw = green_threads.spawn(isolateBody, sharedObj);
check(cw.privateAddr !== undefined, 'private clone created on first write');
check(cw.selfValue === 1042, 'thread reads its own clone (42 + 1000 = 1042)');
check(sharedObj.value === 42, 'main thread view is untouched (42)');

// ---------------------------------------------------------------------------
// 2. Arrays are deep-cloned; the main-heap array stays shared.
// ---------------------------------------------------------------------------
console.log('[2] array COW isolation');
function arrayBody(arr) {
  // Storing the shared array into our own container triggers the clone; from
  // here on the push goes to the private copy.
  const own = { backing: arr };
  own.backing.push(999);
  return { length: own.backing.length, privateAddr: green_threads.cowCloneOf(arr) };
}
const sharedArr = [1, 2, 3];
const ar = green_threads.spawn(arrayBody, sharedArr);
check(ar.privateAddr !== undefined, 'array clone created when stored into own container');
check(ar.length === 4, 'thread sees its own extended array (4)');
check(sharedArr.length === 3, 'main-thread array unchanged (3)');

// ---------------------------------------------------------------------------
// 3. Exotic objects (Map/Set) are shared, never deep-cloned (root cause).
// ---------------------------------------------------------------------------
console.log('[3] exotic objects are shared, not cloned');
function exoticBody(map, set) {
  // Map/Set use their own native mutation path (their host type is not a COW
  // cloneable root), so they are never deep-cloned here.
  map.set('k', 'v');
  set.add(1);
  return {
    mapShared: green_threads.cowCloneOf(map) === undefined,
    mapSize: map.size,
    setSize: set.size,
  };
}
const sharedMap = new Map();
const sharedSet = new Set();
const ex = green_threads.spawn(exoticBody, sharedMap, sharedSet);
check(ex.mapShared, 'Map is shared (cowCloneOf undefined) — never deep-cloned');
check(ex.mapSize === 1, 'Map wrote one entry via its native path');
check(ex.setSize === 1, 'Set wrote one element via its native path');
check(sharedMap.size === 1, 'main-thread Map unchanged (shared)');

// ---------------------------------------------------------------------------
// 4. A plain object that holds a nested Map: the object is cloned, the nested
//    Map reference stays shared (the exact shape that corrupted before).
// ---------------------------------------------------------------------------
console.log('[4] cloning a plain object that references a Map shares the Map');
function refBody(shared) {
  // `shared` is a plain object that references a Map. Storing it into our own
  // container triggers a deep clone of the plain object; the nested Map must
  // remain SHARED (never deep-cloned) — this is the exact object shape that
  // corrupted the heap before the fix.
  const own = { child: shared };
  own.child.n += 1; // mutate the cloned object
  const cloneMap = own.child.map; // Map reachable through the clone
  return {
    childN: own.child.n,
    sameMap: cloneMap === shared.map, // clone references the very same Map
    mapShared: green_threads.cowCloneOf(shared.map) === undefined,
  };
}
const shared = { n: 5, map: new Map([['a', 1]]) };
const mixed = green_threads.spawn(refBody, shared);
check(mixed.childN === 6, 'cloned plain-object field mutated (5 + 1 = 6)');
check(mixed.sameMap, 'clone references the same (shared) Map');
check(mixed.mapShared, 'nested Map is never deep-cloned (cowCloneOf undefined)');
check(shared.n === 5, 'main-thread object unchanged (5)');

// ---------------------------------------------------------------------------
// 5. Concurrent load: N green threads each fib() + write a temp file; every
//    returned file must contain the correct value. This is the scenario that
//    crashed before the fix.
// ---------------------------------------------------------------------------
console.log('[5] concurrent fib -> file load');
function fibToFile(fibN) {
  const value = fib(fibN);
  const file = path.join(os.tmpdir(), `fib-${crypto.randomBytes(4).toString('hex')}.txt`);
  fs.writeFileSync(file, value.toString());
  return path.basename(file);
}
const COUNT = 20;
const fibN = 40;
const expected = fib(fibN).toString();
const files = green_threads.spawnAll(
  Array.from({ length: COUNT }, () => [fibToFile, fibN]),
);
let correct = 0;
for (const f of files) {
  const contents = fs.readFileSync(path.join(os.tmpdir(), f), 'utf8');
  if (contents === expected) correct++;
}
check(correct === COUNT, `all ${COUNT} concurrent files hold the correct value (${correct}/${COUNT})`);

console.log(failures === 0 ? '\nREGRESSION PASS' : `\nREGRESSION FAIL (${failures} checks)`);
process.exit(failures === 0 ? 0 : 1);

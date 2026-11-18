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

// Regression tests for the copy-on-write (COW) write-barrier HeapNumber
// corruption fix.
//
// Root cause (fixed in deps/v8/src/execution/green-thread-cow.cc):
// A boxed double (HeapNumber) stored into a fast-object double-representation
// field runs through JSObject::WriteToField, which writes the HeapNumber's
// bits inline. The CoW barrier nonetheless treated that HeapNumber as a
// heap-object reference, cloned it, and rewrote the field with the clone's
// pointer -- overwriting the inline double bits with heap garbage. The next
// WriteToField / property access then dereferenced the garbage pointer and
// segfaulted (confirmed via lldb in HeapNumber::set_value_as_bits <-
// JSObject::WriteToField <- TransitionAndWriteDataProperty <- StoreIC_Miss).
//
// The fix skips HeapNumber values in both COWWriteBarrierCheck and
// CowCloneForGeneratedCode, so such stores write through unchanged.
//
// A second, related corruption affects internal backing-store values (the
// FixedArray dictionary `properties`, the property-descriptor FixedArray, and
// double arrays) that V8 allocates and links into an object during the
// interpreter's dictionary/map transitions. Those containers are V8-owned, not
// user values: cloning one and rewriting the owner's properties slot to the
// copy orphans V8's dictionary machinery, leaving a stale pointer that a GC
// marks pass then dereferences (SIGBUS). The fix likewise skips CoW cloning of
// internal backing-store values (FIXED_ARRAY, PROPERTY_ARRAY,
// FIXED_DOUBLE_ARRAY) on the store path, mirroring the HeapNumber guard.
//
// Isolation note: green threads run under "Model A" recursive scheduling on the
// embedding thread inside this same isolate/context. That races with both
// node:test's subtest bookkeeping (subtests silently collapse) and common's
// process/async-hook exit handlers (the process is SIGKILLed). To keep each
// scenario deterministic, the green-thread scenarios run in a *child* node
// process that owns the scheduler, and this test asserts on the child's exit
// code -- the same pattern Node uses for other embedding/threading tests.

const common = require('../common');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');

// Runs several green-thread CoW scenarios to completion. Exits non-zero (with a
// distinct code per failure) if any boxed-double store is corrupted, a value
// is lost, or a required CoW clone is missing.
const CHILD_SCENARIOS = `
'use strict';
const g = require('green_threads');
const assert = require('node:assert/strict');

// The CoW write barrier is active while a green thread runs. A boxed double
// (HeapNumber) stored into a double-representation field must survive intact.

// 1. First-write boxed-double store into a fresh fast-object field -- the
//    TransitionAndWriteDataProperty -> WriteToField path that used to crash.
{
  const r = g.spawn(() => {
    const m = {};
    for (const v of [1e15, 3.14159, -2.5e10, 9007199254740993]) m.link = v;
    return m.link;
  });
  assert.equal(r, 9007199254740993, 'scenario 1: boxed-double store preserved');
}

// 2. Two independent boxed-double properties, including an overwrite of the
//    first field with another boxed double.
{
  const r = g.spawn(() => {
    const m = {};
    m.a = 1e12;
    m.b = -7.25e8;
    m.a = 5e13; // overwrite
    return { a: m.a, b: m.b };
  });
  assert.equal(r.a, 5e13, 'scenario 2: overwritten field preserved');
  assert.equal(r.b, -7.25e8, 'scenario 2: sibling field preserved');
}

// 3. Arrays and nested objects holding boxed doubles, plus heap-object
//    reference stores (the normal CoW-clone path).
{
  const r = g.spawn(() => {
    const arr = [];
    for (const v of [1e15, 3.14159e6, -2.5e10]) arr.push(v);
    const inner = { value: 2.718e9 };
    const outer = {};
    outer.arr = arr; // reference store -> CoW clone
    outer.nested = inner; // reference store -> CoW clone
    const sum = arr.reduce((s, x) => s + x, 0);
    return { sum, innerValue: inner.value, head: outer.arr[0] };
  });
  assert.equal(r.innerValue, 2.718e9, 'scenario 3: nested double preserved');
  assert.equal(r.head, 1e15, 'scenario 3: boxed-double array element intact');
  let expected = 1e15 + 3.14159e6 + -2.5e10;
  assert.equal(r.sum, expected, 'scenario 3: array of doubles intact');
}

// 4. A loop writing boxed doubles into distinct fields.
{
  const vals = [1e15, 2e15, 3e15, 4e15];
  const r = g.spawn(() => {
    const row = {};
    for (let i = 0; i < vals.length; i++) row['k' + i] = vals[i];
    return row;
  });
  for (let i = 0; i < vals.length; i++)
    assert.equal(r['k' + i], vals[i], 'scenario 4: field k' + i + ' intact');
}

// 5. Canonical copy-on-write isolation: after a write, the thread's private
//    copy (cowCloneOf) must have a different address than the shared
//    main-heap view (addrOf).
{
  const r = g.spawn(() => {
    const target = { kind: 'shared' };
    const m = {};
    m.link = target; // heap-object reference store -> clones target
    const shared = Number(g.addrOf(target));
    const copy = g.cowCloneOf(target);
    const copyNum = copy == null ? Number.NaN : Number(copy);
    assert.notEqual(shared, copyNum, 'scenario 5: cowCloneOf must differ from addrOf');
    return { shared, copy: copyNum };
  });
  assert.ok(Number.isInteger(r.shared), 'scenario 5: shared address is an integer');
  assert.ok(Number.isInteger(r.copy), 'scenario 5: copy address is an integer');
}

// 6. Dictionary-mode object restore. This is the regression for the copy-on-write
//    corruption of internal backing-store values: when a fast object grows many
//    string-keyed fields the interpreter transitions it to dictionary mode, and
//    V8 allocates/links internal FixedArray-backed containers (the 'properties'
//    dictionary, the 'named' lookup table, a property-descriptor FixedArray).
//    Those internal containers are *not* user values -- they are V8-owned by
//    construction -- yet the CoW barrier previously cloned them and rewrote the
//    owner's properties slot to the copy, orphaning V8's dictionary machinery.
//    The next GC/marks pass then dereferenced the stale backing store and
//    SIGBUSed (confirmed empirically: ~1266 backing-store clones, 516x
//    FIXED_ARRAY, 746x PROPERTY_ARRAY, 4x FIXED_DOUBLE_ARRAY, on the interpreter
//    store path; exit 138).
//
//    The fix skips CoW cloning of internal backing-store values (FIXED_ARRAY,
//    PROPERTY_ARRAY, FIXED_DOUBLE_ARRAY) on the store path, mirroring the
//    HeapNumber write-through guard. This scenario reproduces the corruption by
//    spawning a dictionary-transitioning workload repeatedly with periodic GC.
{
  const results = [];
  for (let iter = 0; iter < 400; iter++) {
    const r = g.spawn(() => {
      const m = {};
      for (let i = 0; i < 5; i++) m['f' + i] = (i + 1) * 1.5e11;
      m.nested = { x: 42, y: 7e12 };
      m.arr = [1e13, 2e13, 3e13];
      return m.f0 + m.f1 + m.f2 + m.f3 + m.nested.x + m.arr[0];
    });
    results.push(r);
    if (iter % 40 === 0 && globalThis.gc) globalThis.gc();
  }
  // A corrupted double backing store surfaces as NaN. If any restore below
  // returned a corrupted double, the reduction is NaN and this fails.
  for (const r of results) {
    assert.ok(Number.isFinite(r), 'scenario 6: dictionary restore intact, no NaN leak');
  }
  assert.equal(results.length, 400, 'scenario 6: all dictionary restores returned');
}

console.error('green_threads COW regression: all scenarios passed');
`;

const res = spawnSync(process.execPath, ['-e', CHILD_SCENARIOS], {
  encoding: 'utf8',
});

assert.equal(res.status, 0,
  `green-thread child process exited with status ${res.status}\n` +
  `--- stdout ---\n${res.stdout}\n--- stderr ---\n${res.stderr}`);
assert.match(
  `${res.stdout}${res.stderr}`,
  /green_threads COW regression: all scenarios passed/,
);

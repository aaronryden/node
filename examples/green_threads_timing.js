// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Green-thread timing harness.
//
//   ./out/Release/node examples/green_threads_timing.js
//
// Spawns N green threads, each doing a chunk of CPU work (iterative BigInt
// Fibonacci, the same CPU profile as the HTTP demo) and yielding between chunks
// so the scheduler interleaves them. It logs the wall-clock execution span of
// every green thread, then answers one question rigorously:
//
//   Do these N green threads do the work faster than one thread would?
//
// Proof: it runs the IDENTICAL total work (N * CHUNKS chunks) as a plain
// synchronous loop on the embedding thread and compares wall times. Because
// these green threads are cooperative coroutines running on the embedding
// thread via recursive entry (no setjmp/longjmp, no OS threads -- see
// src/green_threads.cc and lib/green_threads.js), the green-thread wall time is
// ~= the synchronous baseline. Green threads here give you heap ISOLATION
// (copy-on-write views), not multi-core speedup. A truly parallel run (e.g.
// worker_threads on N cores, see examples/green_threads_parallel.js) would
// finish in roughly ONE chunk's worth of time regardless of N.
//
// Tunables (env): GT_TIMING_THREADS, GT_TIMING_CHUNKS,
//   GT_TIMING_FIB_PER_CHUNK, GT_TIMING_YIELD_EVERY.

'use strict';

const green_threads = require('green_threads');
const { performance } = require('node:perf_hooks');

const NUM_THREADS = Number(process.env.GT_TIMING_THREADS || '12');
const CHUNKS = Number(process.env.GT_TIMING_CHUNKS || '6');
const FIB_PER_CHUNK = Number(process.env.GT_TIMING_FIB_PER_CHUNK || '4000');
const YIELD_EVERY = Number(process.env.GT_TIMING_YIELD_EVERY || '1');

// CPU work for one chunk. fib() keeps BigInt exact and grows the operand size,
// so the later iterations are genuinely expensive -- a fair analog to the fib
// HTTP demo.
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

// Runs as a green-thread body: CHUNKS chunks of fib, yielding between them.
function body(chunkSize, yieldEvery) {
  const start = performance.now();
  let last = 0n;
  for (let c = 0; c < chunkSize; c++) {
    last = fib(FIB_PER_CHUNK);
    if (yieldEvery > 0 && c % yieldEvery === 0) green_threads.yield();
  }
  // dur spans from this body's first resume to its completion; with cooperative
  // nesting these spans overlap, which is exactly the interleave we visualize.
  return { dur: performance.now() - start };
}

function timeWork(fn) {
  const t = performance.now();
  const value = fn();
  return { value, ms: performance.now() - t };
}

function main() {
  // Warm up the fib JIT so per-thread timings aren't skewed by first-call
  // compilation on the earliest-resuming body.
  fib(FIB_PER_CHUNK);

  // Baseline: the exact same total work (NUM_THREADS * CHUNKS chunks) as a
  // plain synchronous loop on the embedding thread.
  const baseline = timeWork(() => {
    let last = 0n;
    for (let j = 0; j < NUM_THREADS; j++)
      for (let c = 0; c < CHUNKS; c++) last = fib(FIB_PER_CHUNK);
    return last;
  });

  // Green threads: NUM_THREADS bodies, each doing CHUNKS chunks, interleaving
  // via yield(). Same total work, same embedding thread.
  const t0 = performance.now();
  const results = green_threads.spawnAll(
    Array.from({ length: NUM_THREADS }, () => [body, CHUNKS, YIELD_EVERY]),
  );
  const cow = performance.now() - t0;

  console.log(
    `\n=== green-thread timing: ${NUM_THREADS} threads, ${CHUNKS} chunks each`,
  );
  console.log(
    `   (each "span" is one body's wall time from first resume to completion;`,
  );
  console.log(
    `    they overlap because the scheduler runs them cooperatively on 1 thread)\n`,
  );
  for (let i = 0; i < results.length; i++)
    console.log(`  gt-${i}: ${results[i].dur.toFixed(1)} ms`);
  console.log('  ----------------------------------------');
  console.log(
    `  synchronous baseline (same work, 1 real thread): ${baseline.ms.toFixed(1)} ms`,
  );
  console.log(`  green_threads.spawnAll wall time              : ${cow.toFixed(1)} ms`);
  const ratio = baseline.ms > 0 ? cow / baseline.ms : 0;
  console.log(`  green / sync                                  : ${ratio.toFixed(2)}`);
  if (ratio < 1.35) {
    console.log(
      '\n  => SERIAL: green threads spent ~ the same wall time as doing the work',
    );
    console.log(
      '     on one thread. They interleave cooperatively but share ONE real OS',
    );
    console.log(
      '     thread, so there is no multi-core speedup. This is the CoW-isolation',
    );
    console.log(
      '     model. For real parallelism see examples/green_threads_parallel.js.',
    );
  } else {
    console.log(
      '\n  => FASTER than serial: the green threads exploited multiple cores.',
    );
  }
}

main();

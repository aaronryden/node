// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Load client for examples/green_threads_http.js.
//
//   ./out/Release/node examples/green_threads_load.js
//
// Fires 10 concurrent GET requests at /fib?n=<N> (default 30), reads each file
// name back from the server, and verifies every file was written with the
// correct Fibonacci value.

const fs = require('fs');
const os = require('os');
const path = require('path');

const PORT = Number(process.env.PORT) || 8124;
const N = Math.max(0, Math.min(80, Number.parseInt(process.env.N || '30', 10)));

// Expected fib(N) as BigInt, for an exact comparison.
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

const expected = fib(N);

// One request per green thread on the server; the client fires 10 at once.
async function request(i) {
  const url = `http://localhost:${PORT}/fib?n=${N}&i=${i}`;
  const started = process.hrtime.bigint();
  const res = await fetch(url);
  const fileName = (await res.text()).trim();
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;

  // Confirm the file the server returned actually exists and is correct.
  const contents = fs.readFileSync(path.join(os.tmpdir(), fileName), 'utf8');
  const ok = contents === expected.toString();

  return { i, status: res.status, fileName, got: BigInt(contents), ok, elapsedMs };
}

async function main() {
  const COUNT = 10;
  console.log(`firing ${COUNT} concurrent GET /fib?n=${N} against http://localhost:${PORT}`);

  const started = Date.now();
  // All 10 fetches are in flight at the same time (client-side concurrency).
  const results = await Promise.all(Array.from({ length: COUNT }, (_, i) => request(i)));
  const totalMs = Date.now() - started;

  results.sort((a, b) => a.i - b.i);
  let correct = 0;
  for (const r of results) {
    console.log(
      `  #${String(r.i).padStart(2)}  ${String(r.status).padStart(3)}  ` +
      `${r.elapsedMs.toFixed(1)}ms  ok=${r.ok ? 'yes' : 'NO '}  ` +
      `fib(${N})=${r.got}  ->  ${r.fileName}`,
    );
    if (r.ok) correct++;
  }

  console.log(`\n${correct}/${COUNT} files correct; ${COUNT} requests in ${totalMs}ms`);
  process.exit(correct === COUNT ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

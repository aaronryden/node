// Copyright 2025 the Node.js authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Green-threaded HTTP server.
//
//   ./out/Release/node examples/green_threads_http.js   (default port 8124)
//
// One GET endpoint, /fib?n=<0..80>:
//
//   GET /fib?n=30
//
// Each request is handled by a brand-new green thread. The green-thread body:
//   1. computes fib(n) synchronously (CPU work on the cooperative scheduler),
//   2. writes the fib value to a random file in the OS temp dir (I/O),
//   3. returns the file name to the HTTP caller.
//
// The green-thread model here runs a body to completion on the embedding
// thread via recursive entry, so the file write is done with fs.writeFileSync
// *inside* the body. That is the green-thread-appropriate pattern: it is not
// async, but it does not require any async/await ceremony, and it guarantees
// the file exists by the time the file name is returned.

const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');

const green_threads = require('green_threads');

const PORT = Number(process.env.PORT) || 8124;

// Iterative big-integer Fibonacci. BigInt keeps fib exact well past
// Number.MAX_SAFE_INTEGER (fib(82) is already too large for a double).
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

// Runs as a green thread. fibN is captured as the thread's argument.
function fibToFile(fibN) {
  const value = fib(fibN);
  const file = path.join(
    os.tmpdir(),
    `fib-${crypto.randomBytes(4).toString('hex')}.txt`,
  );
  // I/O on the cooperative thread (synchronous, so it finishes before we
  // return the file name).
  fs.writeFileSync(file, value.toString());
  return path.basename(file);
}

const server = http.createServer((req, res) => {
  if (req.method !== 'GET' || !req.url.startsWith('/fib')) {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end('usage: GET /fib?n=<0..80>\n');
    return;
  }

  const url = new URL(req.url, 'http://localhost');
  const n = Number.parseInt(url.searchParams.get('n') || '30', 10);
  const clamped = Number.isNaN(n) ? 30 : Math.max(0, Math.min(80, n));

  // One green thread per request. spawn runs it to completion and returns the
  // file name; the response is sent once the green thread has finished.
  const fileName = green_threads.spawn(fibToFile, clamped);

  res.writeHead(200, { 'content-type': 'text/plain' });
  res.end(`${fileName}\n`);
});

server.listen(PORT, () => {
  console.log(`green-threads server listening on http://localhost:${PORT}/fib?n=30`);
});

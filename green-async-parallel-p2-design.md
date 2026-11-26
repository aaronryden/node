# Phase 2 design — parallel async compute drive (no code)

> **Status: design only, no edits.** This is the "separate co-design" from
> `green-async-parallel-plan.md`. P1 (dedicated libuv pump) is done and does not
> parallelize async *compute*. P2 is what does. Nothing here is implemented.

## 1. Where the async drive currently serializes

The serial ceiling is a single lock held for the **whole per-body iteration** in
`DriveGreenAsyncBodiesParallel` (src/green_threads.cc):

```
line 2007: std::lock_guard<std::mutex> mutator_lock(g_green_mutator_lock);   // acquired
  line 2029-2035: drain FINISHED gtReads out of d->pending_reads
  line 2042-2069: resolve them (inline-resume reaction queued on body's own mq)
  line 2077-2110: resolve bridged emb-Promise settles
  line 2122-2124: bctx2->GetMicrotaskQueue()->PerformCheckpoint(green)  // <-- the hazard
  line 2131: restore g_green_driving_body
line 2007: lock released
```

`PerformCheckpoint` is the bad call. It runs the body's promise microtasks — i.e.
the body's awaited continuation — but V8's `MicrotaskQueue::PerformCheckpoint`
touches **isolate-global** state:

- **GlobalHandles table** — MakeWeak / Release of the green result roots
  (line 1997 comment: *"touch the ISOLATE-GLOBAL GlobalHandles table
  (MakeWeak/Release of the green result roots)"*). When a body's continuation
  produces/resolves a value, V8 roots it via a `WeakPersistent` (green
  `settle_value_p` at 339/1735) that lives in the isolate-global table. N workers
  doing MakeWeak/Release concurrently corrupt the global handle nodes.
- **Shared promise/task bookkeeping** — V8's promise-reaction + isolate global
  task-queue bookkeeping (line 2115-2117).

That corruption is what produces `Check failed: node->IsInUse()` FATAL
(reproduced ~100% under GT_NUM_WORKERS≥8 in the parallel gate, line 2001). So the
lock is not conservatism — it is the one thing keeping the global tables sane.

## 2. What is ALREADY worker-safe (the isolation exists)

The per-body queue isolation is already in place; P2 does not invent it, it
*extends* it:

- **Disjoint green microtask queue per body.** `data->body_mq_p`
  (unique_ptr<MicrotaskQueue>, 242/1779/1887). `Context::GetMicrotaskQueue`
  returns the body's own queue. Comment 1887-1900: *"each body owns a private
  green context + MicrotaskQueue … two steps MUST happen on the SAME thread, so
  each body is owned by exactly one drive worker that both resolves its finished
  reads and checkpoints its queue."*
- **Round-robin partition.** DriveGreenAsyncBodiesParallel assigns each body to
  exactly one OS worker (line 1990 `for (d : bodies)`, comment 1990). Body `d`
  is touched by worker `w` alone. So the per-body queue is **already** single-
  owner per worker — two workers never touch the same `body_mq_p`.
- **Proven per-worker V8 isolation.** GreenJSEntry (2291+) uses the disjoint
  true-parallel recipe: per-OS-thread `HandleScopeImplementer` (live handle
  cursor + handle blocks), per-worker `entered_contexts_`, per-worker
  `ThreadLocalTop::context_` via x25 (`set_context`). Comment 2315-2329 verifies:
  *"two workers can now Enter/Exit their own contexts without the
  'Cannot exit non-entered context' abort … the disjoint test runs 4 concurrent
  workers with the serializer released and is clean."*

**Bottom line of §2:** the *queue* and the *handle/context* routing are already
per-worker/per-body. The ONLY thing that is isolate-global and forces the lock is
the GlobalHandles + promise-reaction bookkeeping that `PerformCheckpoint(green)`
drags in.

## 3. Hazard map — the three buckets

| Work the continuation does | Scope | Worker-safe today? | Owner |
|---|---|---|---|
| Run the body's own queued reactions (JS resume at await point) | body's `body_mq_p` | **YES** (per-body mq, single owner) | the body's worker |
| Root the produced settle value | isolate-global **GlobalHandles** | **NO** — concurrent MakeWeak/Release corrupts the table | → must be per-worker or de-globalized |
| Queue any promise reactions / isolate tasks | isolate-global promise-task bookkeeping | **NO** | → must be per-worker or de-globalized |
| Compile / tier / feedback (map tiering, feedback vectors, JSDispatchTable) | process-global | **NO** — guarded by `GreenThreadCowSharedWriterScope` in GreenJSEntry (2340-2342) | stays serialized |

So P2's job narrows to **two** isolate-global hazards — the GlobalHandles table
and the promise/task bookkeeping. The compilation-state hazard is orthogonal and
stays serialized (it is the same wall sync bodies already hit).

## 4. Route analysis

### Route 1 — manual green-microtask drive, no `PerformCheckpoint(green)`
Drive **just this body's own queue** (iterate `body_mq_p` reactions on the
body's worker) instead of calling the isolate-global `PerformCheckpoint(green)`.

- **Pros:** touches the smallest surface — the per-body queue is already single-
  owner. Avoids the isolate global task queue.
- **Cons (blocking):** the reaction that runs the continuation STILL produces
  roots → still needs to root into the GlobalHandles table, and STILL runs
  compilation (map tiering / feedback vectors) for any non-trivial JS. Both are
  isolate/process-global and hit the same hazards. Merely swapping
  `PerformCheckpoint` for a manual queue walk does **not** remove the lock; it
  just moves the hazard from the V8 call to the JS the reaction runs. You'd
  still need to de-globalize routing (#5) to actually gain parallelism.
- **Verdict:** route 1 is the *mechanism* (drive the body's own queue per
  worker), but it is **necessary, not sufficient**. It pairs with #5.

### Route 2 — de-share GlobalHandles / microtask state per body
Give each body its own **private** global-handle root table + its own
microtask/promise-reaction queue, so the continuation roots/queues into per-body
state owned by the body's worker — then the isolation at §2 already makes the
drive parallel.

- **Pros:** the per-body isolation is the same shape already proven for handle
  scope and context (§2). If GlobalHandles + microtask queue become per-body,
  two workers root/queue into disjoint tables → no FATAL → the `mutator_lock`
  around PerformCheckpoint can be dropped for the continuation-drive portion.
- **Cons:** GlobalHandles is owned by the green isolate; making it per-body
  requires either (a) a body-private `GlobalHandles`-shim that the green roots
  register into instead of the isolate's, or (b) keeping the green isolate's
  global table but guarding only the root registration with a fine-grained lock.
  (a) is the clean win but the larger change; (b) is incremental.
- **Verdict:** route 2 is the *structural* fix for the two global hazards.

### Recommended: Route 1 mechanism + Route 2 isolation, smallest safe seam
Do not try to parallelize the whole body re-execution (that still hits
compilation-state, a separate wall). Parallelize **only the continuation-drive
loop** — drain + resolve + run reactions — by (1) driving each body's own queue
per worker (route 1) and (2) rooting the produced value into a **per-body**
handle table instead of the isolate-global one (route 2a/b), while leaving the
compilation-state guard (`GreenThreadCowSharedWriterScope`) and isolate
activation exactly as they are.

This keeps the change surface to the region 2007-2131 and touches no V8-internal
routing.

## 5. The concrete seam (where the code would go)

All changes live inside the worker that already owns body `d` (line 2007-2131);
no new threads, no new partitioning.

### 5.1 Root into a per-body handle table, not the isolate-global table
- **Introduce** `GreenHandleTable` (a body-private root registry; per-`GreenCallData`,
  declared near `body_mq_p` at 242). It holds the green result-root Persistents
  that currently go into `settle_value_p` / weak roots.
- **Replace** the MakeWeak-of-isolate-global path with: register the root in
  `d->handle_table` on the body's worker, and release it when the body settles.
  Because `d` is single-owned per worker, `handle_table` is single-threaded —
  **no lock**. This removes the GlobalHandles-table FATAL source.
- **Why it is safe:** §2 proves per-worker handle-scope routing already works
  (disjoint test, 4 workers clean); a per-body table is the same invariant one
  level tighter.

### 5.2 Drive the body's own queue per worker (route 1 mechanism)
- **Replace** line 2124 `bctx2->GetMicrotaskQueue()->PerformCheckpoint(green)`
  with a **per-body reaction driver** that runs only `d->body_mq_p`'s reactions
  on worker `w`. The driver reuses the existing per-worker scope recipe from
  GreenJSEntry (Isolate::Scope + HandleScope + Context::Scope, 2353-2356) so the
  resume stays in the body's own context.
- **Key invariant to preserve:** the same-thread microtask enqueue→run contract
  (1891-1895) — because the driver runs on `d`'s owner worker and the reaction
  may enqueue new tasks into `d`'s own queue, keep enqueue + run on the same
  worker. (This is already true — the worker owns `d`.)
- **Compilation-state stay:** any reaction that compiles/tiers JS still runs
  under the same compile-state guard used today (CowSharedWriterScope path). This
  is the one hazard we deliberately keep serialized; it does not need the
  coarse `g_green_mutator_lock`.

### 5.3 What does NOT change (explicitly out of scope)
- Isolate activation (`v8::Isolate::Scope`, 2353) and JS-invoke permission — still
  isolate-global, stay serialized; not touched.
- Compilation-state guard (`GreenThreadCowSharedWriterScope`, 2345) — stays.
- The always-allocate + CAS bump window (1899-1901) — stays; disjoint bodies
  already allocate disjoint fresh objects.
- Route 5.2-5.3 both run on the SAME worker that owns `d`, so no new partitioning
  or thread hand-off.

## 6. Verification plan (per plan-doc Proof strategy)

- **Correctness gate first:** keep `test-green-threads-cooperative-io.js`,
  `test-green-threads-async-drive-pump.js`, `test-green-threads-parallel-execution.js`
  green at GT=2/4/8/16, CPU_COUNT 4-8. The `IsInUse()` FATAL is the acceptance
  signal — if the lock is truly removable, GT=8 must stay green (currently it
  FATALs with the lock removed).
- **Wall-clock:** reuse the cooperative-io harness; parallel success = GT=4/8
  beats GT=1 materially at fixed body count, plateauing at high worker counts.
  (Async compute is the metric P1 did not improve; P2 must.)
- **OS-parallel / timing-independent:** extend `__concurrent_entry(N)` from
  test-green-threads-parallel-execution.js — all workers must cross the barrier
  simultaneously; a serial drive deadlocks/times out.
- **Negative check:** temporarily re-add the coarse lock and confirm the
  wall-clock regression reappears — proves the measurement is actually seeing
  the removed serialization, not noise.

## 7. Risk register

- **Reaction enqueue ordering** — a reaction enqueued during the per-body drive
  must run in the same worker's next pass, not leak to another worker's queue.
  Guarded by the single-owner invariant (5.2).
- **MakeWeak weak-root liveness** — per-body table must stay rooted until the
  body settles or is torn down; tie lifetime to `GreenCallData`
  (declared above `context_p` per 240-241 destruction order).
- **Compilation-state wall** — if a continuation does non-trivial JS, the
  compile-state guard serializes anyway; P2 parallelism is bounded by how much
  work is in the reaction vs the body. Expected: strong speedup on bodies whose
  await-return work is root-routing + small JS.
- **Promise-reaction global bookkeeping** — if reactions enqueue to the isolate
  global task queue (not just the body mq), that is still a hazard. Track
  whether the reaction enqueue can be redirected to the body queue; if not, this
  needs the microtask-route part of route 2.

## 8. Build order
P2 rides on P1 (P1 removed the pump as a lock-contention point and produced the
exact one-heap / one-uvloop shape). Build P2 top of the P1 commit. Do **not**
re-add the `HandleGCRequest` deferral (historical regression).

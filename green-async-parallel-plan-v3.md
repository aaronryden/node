# Green async-drive re-architecture: per-worker green isolates

> **Goal (user, 2026-11-26):** async green-thread continuations must run on the
> green OS *workers* (in parallel), not serialized on one shared green isolate.
> Re-architect so each async-drive worker owns its own green V8 isolate + drives
> its continuations green on that isolate.
>
> **Status: implementing (Stage 1).** Base 808489d. Committed: 4d3a1a5 (UAF fix).
> Build: `ninja -C out/Release node -j20`. Worktree: /Users/aryden/dev/node-v26,
> branch feature/green-threads.
>
> The earlier Route-1/Route-2 document (`green-async-parallel-p2-design.md`)
> explored making the *shared* isolate's runner state worker-safe from Node —
> Codex §4 killed that ("cannot honestly scope to green_threads.cc"). This doc is
> the realization of the user's steer: **give each worker its own isolate**, so
> the runner state is per-worker *by construction* (isolates own their
> GlobalHandles / promise bookkeeping / microtask runner).

## 1. Why the current async drive serializes (confirmed from source)

`DriveGreenAsyncBodiesParallel` (`src/green_threads.cc`): N OS workers each do
`v8::Isolate::Scope iso_scope(green)` with the **same** `green` (`g_green_isolate`,
`GetGreenIsolate` 769–848; registered on `uv_default_loop()`, 783; `SetGreenThreadMode(true)`
so EntryStackItem routes per-worker, 844–847). The per-body iteration holds a
**coarse** `g_green_mutator_lock` (declared :1918, held :2007) for the whole of:

- drain finished `gtRead`s out of `d->pending_reads` (:2029–2035),
- resolve them → inline-resume reaction on the body's own microtask queue (:2042–2076),
- resolve bridged emb-Promise settles (:2084–2117),
- `bctx2->GetMicrotaskQueue()->PerformCheckpoint(green)` (:2129–2131) — runs the
  awaited continuation.

`PerformCheckpoint` touches **isolate-global** state (GlobalHandles table,
promise reaction jobs, `kCurrentMicrotask`, `continuation_preserved_embedder_data_`…).
N workers driving different bodies' PerformCheckpoint on ONE isolate concurrently →
`Check failed: node->IsInUse()` FATAL (Codex §1, reproduced ~100% at GT≥8). So the
coarse lock is mandatory *while the isolate is shared*.

Baseline measurement (`/tmp/gt_p2_wallclock.js`, 8 bodies × 40 gtReads, each
continuation 5M-iter CPU loop): **GT=1 = 110 ms, GT=2 = 0.90×, GT=4 = 0.93×** —
async compute does NOT parallelize; more workers only add coarse-lock contention.

## 2. What is ALREADY worker-safe (so per-worker isolates are a small change)

- **Per-body microtask queue.** `data->body_mq_p` (unique_ptr<MicrotaskQueue>,
  created :1779 in `SetupAndRunGreenAsyncBody`). Each body's continuation is
  checkpointed on its own queue.
- **Per-worker handle-scope routing.** `SetIsolateHandleScopePerWorker(working_isolate,
  true)` (:2898) is armed for the whole drive window, so handle-scope + entered-
  context state is per-OS-thread.
- **Concurrent compilation suspended.** `SetConcurrentCompilationSuspended(true)`
  (:2969) is held around the async drive — the one *process-global* hazard (map
  tiering / feedback vectors / JSDispatchTable, per `GreenJSEntry` :2346) is
  already serialized by refusing to compile. So per-worker isolates need **no**
  extra compile guard.
- **libuv I/O plumbing.** `uv_queue_work` is thread-safe; the shared GreenIOLoop +
  dedicated pump drive completions; `ReactivateGreenThread` wakes a drive worker.
  All of this stays shared and correct — it is not the serialization point.

## 3. Target architecture

```
RunTasks (embedding thread):
  - create N per-worker green isolates  iso[w]   (w = 0..N-1)
  - arm AlwaysAllocate (GC-free window) on EACH iso[w]
  - for each async body i: setup on iso[i % N]   (body_ctx, relocate fn, first frame,
    settle reaction — all on iso[i%N])
  - DriveGreenAsyncBodiesParallel(datas, iso[], N)

DriveGreenAsyncBodiesParallel:
  - worker w  ->  iso[w]  (owns all bodies {w, w+N, ...}, all on iso[w])
  - each worker enters iso[w], drains + resolves + PerformCheckpoint ON iso[w]
  - NO coarse lock: each isolate's GlobalHandles / promise bookkeeping / task runner
    is touched only by its owning worker  => races gone by construction
  - shared GreenIOLoop + pump (I/O) unchanged; Reactivate scheduler narrow-locked
  - compilation stays suspended (caller holds it around the call)
```

**Isolation invariants that make the coarse lock removable:**

1. Each `iso[w]` is entered only by worker `w` → its GlobalHandles/promise/task
   state is single-owner. PerformCheckpoint no longer races.
2. Bodies are partitioned round-robin so worker `w`'s bodies all live on `iso[w]`
   (serial within a worker, parallel across workers).
3. Per-worker handle-scope routing already armed.
4. Compilation suspended process-wide.
5. libuv loop + Reactivate scheduler stay shared but are thread-safe / narrow-locked.

## 4. Per-worker isolate creation (reusing GetGreenIsolate)

`GetGreenIsolate` (769–848) creates ONE isolate: `Isolate::Allocate(embedding->
GetGroup())` (shares string table + group), registers on a loop, copies the
embedding V8 snapshot blob for the shared readonly heap (:788–805), own incremental
CppHeap (:827–832), `SetGreenThreadMode(true)`.

New `CreateWorkerGreenIsolate(embedding)`: same recipe, but
- `SetGreenThreadMode(false)` — a single worker owns it, no per-worker EntryStack
  routing needed,
- registered on `uv_default_loop()` (the async drive does not pump the isolate's
  own loop — shared GreenIOLoop does),
- **not** cached in the process-global `g_green_isolate` (each worker isolate is
  distinct); disposed after the drive.

Isolate creation must be single-threaded (V8 platform registration + snapshot), so
all N are created up front on the embedding thread in RunTasks before workers spawn.

## 5. GC-free window allocation

`node::GreenEnterAlwaysAllocation(iso)` (:2736 arms AlwaysAllocate on one isolate;
`GreenLeaveAlwaysAllocation` releases). Currently armed on `working_isolate`
(`g_green_isolate`) once. For per-worker isolates, arm it on **each** `iso[w]`
(each worker arms its own during the drive; the outer window on `g_green_isolate`
covers sync bodies + the async-drive setup cross-isolate allocations). Nested
AlwaysAllocate across different isolates is per-isolate state, so it composes.

## 6. Implementation stages (each: build `-j20`, test, then commit)

**STAGE 1 — scaffolding + fallback.** Add `GT_ASYNC_WORKER_ISOLATES` env gate
(default **off** → byte-identical to current shared-isolate behavior; the committed
baseline is never at risk). When ON: create N worker isolates up front, assign
bodies round-robin, drive per-worker, **coarse lock still held during integration**
(so Stage 1 proves isolation + relocation machinery without yet removing
serialization). Verify GT=1/2/4/8/16 no hang/crash with the flag both ways.

**STAGE 2 — remove the coarse lock.** With per-worker isolates, release
`g_green_mutator_lock` around the per-body iteration (keep a narrow lock only
around the shared Reactivate scheduler drain/reenqueue + block, which remain
shared state). Build + verify GT=8/16 no `IsInUse()` FATAL (the acceptance signal
that isolation is correct). Then measure `/tmp/gt_p2_wallclock.js` — expect
GT=4 speedup.

**STAGE 3 — polish + regression.** Tune worker count, consider caching worker
isolates process-wide to amortize creation cost across `g.spawnAll` calls, add a
parallel regression test mirroring `test-green-threads-async-drive-pump.js` at
GT=2/4/8, confirm the full green-thread suite + Codex re-review.

## 7. Risk register

- Isolate creation/disposal cost per `g.spawnAll` — measured in Stage 1; mitigated
  by process-wide caching if needed (Stage 3).
- `data->gt` (vi::GreenThread descriptor) is created on the embedding-side
  `vi_isolate_for_gt` today; must be created on each body's worker isolate.
- Mixing sync bodies (shared `g_green_isolate` + CAS bump) with async bodies
  (per-worker isolates) in one dispatch — sync path unchanged; async path guarded.
- Disposal order: isolate + CppHeap released before V8::Dispose; snapshot data is
  process-owned (not per-isolate), so no snapshot leak.

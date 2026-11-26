# Green Async-Thread Parallel Plan (Golang-style)

**Goal:** async green bodies run **in parallel across the GT OS worker pool**, proving
a wall-clock speedup + OS-parallel confirmation at `GT_NUM_WORKERS=4`. Target shape:
**one heap** (green isolate) + **one uvloop** (`GreenIOLoop`) + **parallel compute**
across `GT_NUM_WORKERS` OS threads — the Go `GOMAXPROCS` + single netpoller model.

**Worktree:** `/Users/aryden/dev/node-v26` · branch `feature/green-threads` · base `779e7b2`
**Build:** `ninja -C out/Release node -j20` (one build at a time; confirm `LINK node`).

## Task status
| Phase | State | Notes |
|-------|-------|-------|
| **P1** — dedicated libuv pump thread | ✅ **COMPLETED** | Verified green at GT=2/4/8/16. Drive-loop hang root-caused + fixed (see "P1 drive-loop hang — RESOLVED"). Regression test added + flip-verified. Committed. |
| **P2** — parallel async compute | ⬜ TODO | Separate co-design; P1 unblocks the pump-as-contention-point. |

## Status of the investigation (grounded)

### The measured serial ceiling
Wall-clock is flat at ~9.5 ms across `GT_NUM_WORKERS` = 1,2,4,8,16 and across
`UV_THREADPOOL_SIZE` = 1..16, scaling **linearly with body count at every worker count**.
→ the green-side `g_green_mutator_lock` serializes the async drive; the libuv pool is
*not* the binding constraint for these tiny reads. (Sync compute already parallelizes;
only the async drive does not.)

### Two independent hazards — same global lock
| Phase | Hazard | Current enforcer | V8/uv code |
|-------|--------|------------------|-----------|
| **P1** | **libuv rule:** one `GreenIOLoop()` may have only one `uv_run()` caller at a time (libvue tolerates one caller) | held inside `g_green_mutator_lock` per worker | `uv_run(GreenIOLoop(), UV_RUN_NOWAIT)` at `src/green_threads.cc:2128`; submit at `:1683` |
| **P2** | **V8 rule:** `PerformCheckpoint(green)` + isolate-global GlobalHandles + shared promise-reaction bookkeeping are NOT worker-safe; naive lock removal FATALs ~100% with `Check failed: node->IsInUse()` at `GT_NUM_WORKERS>=8` | held inside `g_green_mutator_lock` (`:2006`, span `:2006`–`:2140`); continuation via `GetMicrotaskQueue()->PerformCheckpoint(green)` at `:2120` | see comment `:1993`–`:2005` |

These hazards are independent but both currently ride on `g_green_mutator_lock`.

## Architecture facts (verified)
- **One heap:** all green bodies + COW clones run on a single green isolate; disjoint
  allocation is already armed (`CAS bump-pointer` + `AlwaysAllocate` GC-free window at
  `:2853`–`:2862`, re-armed per drive `:2919`–`:2936`).
- **Per-worker scheduler P:** `GreenThreadPool::Run` creates one `GreenScheduler`(1 P) per
  OS worker, round-robins `EnqueueGlobal` across them (`:2559`–`:2575`); sync bodies get
  real N-way parallel compute via `GT_TRUE_PARALLEL` (`GreenThreadCowSharedWriterScope`
  no-ops at `:222`–`:224`).
- **Async bodies run GREEN on the green isolate** (shared-heap path, `SetupAndRunGreenAsyncBody`
  `:1740`); each body owns a **disjoint** green context + own microtask queue (`:1757`–`:1779`).
- **Pump is not dedicated today:** every drive worker both `uv_queue_work` AND `uv_run(NOWAIT)`,
  serialized by the lock; comment `:2126`–`:2127`.
- **x25-mirror race (former P2 justification) is already being structurally disabled** —
  Stage 1 readers off-mirror, Stage 2 write gated off on green path (`isolate.cc:5068`–`:5071`).
- `uv_queue_work` (`:1683`) is already thread-safe (multi-tee submission fine); only the
  `uv_run` *pump* is the single-caller problem.

## Phase 1 — dedicated libuv pump thread (netpoller model) — SAFE, standalone ✅ DONE
Move `uv_run(GreenIOLoop())` out of the drive workers into **one** dedicated pump thread.
Workers only `uv_queue_work()` (thread-safe) into the shared loop, then **block**
(`BlockGreenThread` + `WaitForReactivationWithPred`, `:2164`) until the pool's
`ReactivateGreenThread` (`:1577`) wakes them. One pump thread drains the single loop.

Result: **one heap + one uvloop + parallel I/O drive** — the Go netpoller split. Compute
parallelism inside async bodies is unchanged by P1 (still under the lock); that is P2.

Why do P1 first (regardless of P2): it removes the pump as a lock-contention point and
gives the exact "one uvloop" shape. It is safe and independently correct.

### P1 design notes / invariants to preserve
- Single `GreenIOLoop()` static stays a single loop; keep its comment `:1512`–`:1516`
  (separate from Node's main loop; pumping inside `g.spawnAll` must not perturb Node state).
- `ReactivateGreenThread` from the pool thread stays as-is — it only signals a wakeup via
  `react_mutex_`/`react_cv_` with a steal-proof per-worker predicate (`:471`–`:481`,
  `:2164`–`:2185`).
- Do **not** re-add the `HandleGCRequest` deferral (historical regression — shared context).
- Keep `AlwaysAllocate` + CAS bump pointer armed across the drive; P1 does not touch it.

## Phase 2 — parallel async compute (the "separate co-design") — shared-V8 green-box surgery
Continuations must leave `PerformCheckpoint(green)` under the global lock. Two routes
(pick one; route 1 is the cleaner fit):

1. **Reuse the sync parallel path.** Drive each continuation as a *direct resumed call*
   through the per-worker `GreenJSEntry` machinery (same per-OS-thread `PerIsolateThreadData`
   isolation + shared CAS-bump + `AlwaysAllocate` window that already parallelizes sync bodies).
   Requires manually driving promise reactions instead of V8's global `PerformCheckpoint`.
2. **De-share the green GlobalHandles / microtask state per-body.** Result roots are already
   owned per-body (`data->async_promise_p`, `context_p` at `:1757`–`:1779`); isolate-global
   bits must be isolated per-worker.
    > Full design (hazard map + route 1×2 seam + verification + risk register) lives in
    > `green-async-parallel-p2-design.md`. **No code yet** — this section is design-only.

### P1 <-> P2 dependency
- **Independent hazards**, but with the **one-shared-loop** requirement, **P2 needs P1**:
  removing the lock (P2) makes every worker call `uv_run` at once → violates libuv's single
  `uv_run` caller → must centralize the pump (P1) *or* use per-worker loops (rejected: breaks
  "one uvloop"). So P2 must pair with P1. Build order: **P1 (safe) → P2 (top).**
- If per-worker loops were acceptable, P2 could proceed without P1 — but that is N loops,
  not the one the user asked for.

## Proof strategy (both phases)
- **Wall-clock:** reuse `test/parallel/test-green-threads-cooperative-io.js` style harness
  (or a scaled one) measuring ms as `GT_NUM_WORKERS` ∈ {1,4,8} grows at fixed body count;
  parallel success = GT=4/8 beats GT=1 materially and N-scaling plateaus at high worker counts.
- **OS-parallel (timing-independent):** extend the `__concurrent_entry(N)` barrier rendezvous
  already in `test/parallel/test-green-threads-parallel-execution.js` — all workers must cross
  simultaneously (a serial drive deadlocks/times out).
- **Correctness gates:** keep existing `test-green-threads-cooperative-io.js`,
  `test-green-threads-async-context.js`, `test-green-threads-concurrency.js` green;
  cap test CPUs to 4–8.

## P1 drive-loop hang — RESOLVED (root cause + fix)
The P1 dedicated-pump design hung ~50-100% at `GT_NUM_WORKERS==2` (a spawnSync child whose
caller Promise never settles). Backtrace of the hung process was decisive:

- One drive **worker was 100% stuck inside `GreenScheduler::DrainReactivations()`, frame #0 at
  the `blocked_list_` search scan (green-thread-scheduler.cc:445), holding `react_mutex_`.
- That held mutex **starved** the libuv **pool threads** (stuck in `__psynch_mutexwait` inside
  `ReactivateGreenThread` at scheduler.cc:417) so reads never completed, and **starved the other
  drive worker** (blocked on `react_mutex_` at DrainReactivations's own `std::lock_guard`).
- Net: the drive loop deadlocked; the child's `await g.spawnAll(...)` Promise never resolves;
  spawnSync's timeout kills it.

**Root cause:** `blocked_list_` had cycled. `BlockGreenThread` head-inserts the body's green
thread, but under the shared scheduler a single worker can call it a **second time on the same
`gt`** whose prior Reactivate had already been drained+unlinked by another worker (the body's
`kBlocked` state was still set, since Drain only runs `kRunnable` after a successful unlink).
Head-inserting an already-blocked `gt` makes `blocked_list_head_ == gt` with `gt->blocked_next()
== gt` — a **self-cycle**. Every later `DrainReactivations` scan for a *different* body then
loops forever on that self-cycle, pinning `react_mutex_` (hence the starvation observed above).

**Fix (green-thread-scheduler.cc), two complementary layers:**
1. **`BlockGreenThread` idempotency guard** (root-cause): `if (gt->state() == kBlocked) return;`
   before the head-insert, so a `gt` can never be added to `blocked_list_` twice.
2. **Cycle-safe `DrainReactivations` scan** (defense-in-depth): stop the walk if it revisits the
   list head after the first step, so a stray corrupt link can never pin the worker (and the
   mutex) forever.
   - `DrainReactivations` still holds `react_mutex_` for the whole function (react_queue drain +
     blocked_list_ unlink) so every `blocked_list_` mutation is serialized with `BlockGreenThread`.
- `deps/uv/src/uv-common.h` active_reqs counter is made atomic (multiple threads
  `uv_queue_work` vs pool-thread completion touch one non-atomic counter) — general libuv fix, not
  green-only.
- **Regression test:** `test/parallel/test-green-threads-async-drive-pump.js` — many bodies, each
  doing many sequential `gtRead`s (heavy block/reactivate cycling) at `GT_NUM_WORKERS` {2,1}, caught
  by spawnSync timeout. **Flip-verified:** 5/6 runs hang (rc=124) with the original scheduler, 0/6
  with the fix.

**Verification (all green, CPU_COUNT=4):** GT=2 (12/12), GT=4 (8/8), GT=8 (8/8), GT=16 (8/8);
existing `test-green-threads-cooperative-io.js`, `-async-drive-pump.js`, `-spawnall-n2.js` pass;
`-concurrency.js` is timing-flaky under CPU contention (passes 3/3 standalone; unrelated to this
fix — it targets concurrent young-GC/compiler races, not block/drain).

## Open risk / known side issue
- async require-bridge `fs.promises.readFile` currently returns wrong byte counts
  (`allOk:false` in probes) — the async green result-marshal gap. Drive timing is unaffected
  (no hang), but a correctness-gated parallel test needs that fixed first. Separate from P1/P2.

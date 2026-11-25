# Green Threads Phase 2 — Plan for Real Parallel Execution of Bodies across N OS Threads

**Status:** PLAN ONLY. This document is a grounded, ordered, staged plan. Per the
task mandate it changes **no** body-execution serialization today — only a test
file (`test/parallel/test-green-threads-parallel-execution.js`, already the red
gate) and this doc. The `fix:` edits described for Stages 2/3/5 are future work;
each stage is written so it can be *proven correct in isolation* before the next
stage's riskier change is attempted.

**Target:** `node-v26` worktree, branch `feature/green-threads`, checkpoint
`1b2c03f7920`. Per-worker execution-state routing (`V8_ENABLE_GREEN_WORKER_EXECUTION_STATE`)
is already compiled in (verified by the `[GT-CONFIG]` banner: `GREEN_THREADS=1
GREEN_WORKER_EXEC_STATE=1`).

**Mandate & hard limits:**
- Do **not** refactor the `GreenThreadCowSharedWriterScope` lock or the
  `g_green_mutator_lock` body serializer. Those are the documented crash zone
  (~78% SEGV once serialization drops; see `node-v26/AGENTS.md` "Build Playbooks").
- Iteration is **test + read + doc**. No full rebuild is needed to *plan*; the
  build/CPU discipline (`-j20`, one build at a time, `tee /tmp/*.log`, confirm
  `LINK node`) applies to whichever stage above first touches V8 C++.
- The discriminator (`test-green-threads-parallel-execution.js`) is the acceptance
  gate: allocation-light integer compute, asserts **correctness AND** `speedup > 1.4`
  at N=4. It currently **fails deterministically** (`~1.0x`); it passes only under
  true parallel execution. Do not weaken it.

---

## 0. Measured baseline (evidence, reproduced on the checkpoint binary)

```
[GT-CONFIG] compiled: GREEN_THREADS=1 GREEN_WORKER_EXEC_STATE=1 | env: GT_SHARED_HEAP=(unset)...
[GT-CONFIG] compiled: GREEN_THREADS=1 GREEN_WORKER_EXEC_STATE=1 | env: GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 ...
```

| Check | Command | Result |
|---|---|---|
| Allocation parallelism (M2/3) | `GT_SHARED_HEAP=1 GT_M2_PARALLEL=1 GT_NUM_WORKERS=4 node test/parallel/test-green-threads-parallel-disjoint.js` | **PASS** — "4 workers reserved disjoint young-gen pointers from one green heap (lock released)" |
| Body execution | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 node examples/green_threads_timing.js` | **SERIAL**, `green / sync ≈ 1.7` (BigInt-fib harness; prints `=> SERIAL`; stable 1.66-1.73 across runs) |
| Compute discriminator | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 node test/parallel/test-green-threads-parallel-execution.js` | **FAIL (red gate)** — `must show >1.4x wall-clock speedup (got 0.983x, serial_ms=7745.3, green_ms=7878)`; `acc_ok` passes |

Compute discriminator determinism (allocation-light integer bodies, 3 runs):
`1.001x`, `0.996x`, `1.007x` — all `~1.0`, all fail `>1.4`, all `acc_ok`. Releasing
the mutator lock with `GT_M2_PARALLEL=1` **does not** help the compute bodies
(`0.988x` -> `1.013x`), which isolates the *second* serializer (Stage 1 root cause b).

---

## 1. Root cause — why bodies serialize even though per-worker routing is compiled in

Two **stacked, process-global** locks hold every worker body serially. The
per-worker `x25`/`execution_state_` routing is real and per-worker, but it is
irrelevant while a global lock pins bodies one-at-a-time.

### a. Barrier 1 — `g_green_mutator_lock`, held across the *whole* body
- `src/green_threads.cc:1548` — `static std::mutex g_green_mutator_lock;` (process-global).
- `src/green_threads.cc:1781-1803` — each pool worker locks it before the body and
  unlocks only after `GreenJSEntry` returns. Line `1783` `if (!parallel) mutator_guard.lock();`
  **holds the lock for the entire body unless `GT_M2_PARALLEL` is set**.
- Net: with `GT_M2_PARALLEL` unset (the timing test, the discriminator), all 4 OS
  workers queue on one mutex; bodies run strictly one-at-a-time. `speedup ~ 1.0`.

### b. Barrier 2 — `GreenThreadCowSharedWriterScope` at `GreenJSEntry:889` (the deep one)
- `src/green_threads.cc:889` — `vi::GreenThreadCowSharedWriterScope shared_writer_scope;`
  is a stack object spanning the *entire* body (constructed before `fn->Call` at
  `:916`, destroyed when `GreenJSEntry` returns).
- Its constructor (`deps/v8/src/execution/green-thread-cow.h:188-197`) locks iff
  `GreenThreadCow::ActiveCount() > 0`, acquiring `GreenThreadCow::SharedStateMutex()`
  (`:190`).
- `SharedStateMutex()` is a **single process-global recursive mutex**
  (`deps/v8/src/execution/green-thread-cow.cc:51-54`, function-local static).
- `ActiveCount()` is `> 0` for the *whole* spawnAll lifecycle: `Initialize` bumps it
  at thread creation (`src/green_threads.cc:995`), `Teardown` drops it only after all
  bodies join (`src/green_threads.cc:1315` / `:1890`). So the scope **always locks**
  under a shared-heap fan-out, and holds the global mutex for the entire body.
- **Empirical proof it dominates Barrier 1:** the compute discriminator stays `~1.0x`
  with `GT_M2_PARALLEL=1` (Barrier 1 released) — `0.988x -> 1.013x` (`/tmp/child_disc.js`).
  Only Barrier 2 can remain, so Barrier 2 is the binding serialization for compute bodies.

### c. Shared vs per-worker state (the real gate for parallelism)

| State | Today | File:line |
|---|---|---|
| `x25` = `kExecutionStateRegister` | **per-worker** (register `x25`) | `deps/v8/src/codegen/arm64/register-arm64.h:536` |
| Arm x25 at JS entry from shared mirror | **per-thread arm, shared-mirror load** | `deps/v8/src/codegen/arm64/macro-assembler-arm64-inl.h:1263` (`Ldr x25, [kExecutionStateRegister-relative mirror]`); entry `deps/v8/src/builtins/arm64/builtins-arm64.cc:969-983` |
| Per-worker block base lookup | **per-worker** | `Isolate::CurrentExecutionStateBase()` returns `&data->execution_state_` (`deps/v8/src/execution/isolate.cc:631-647`) |
| Mirror `execution_state_base_` in IsolateData | **per-worker write into per-isolate field** | `deps/v8/src/execution/isolate.cc:4962` (written by `SetIsolateThreadLocals`); field `deps/v8/src/execution/isolate-data.h:299-303` |
| `g_green_mutator_lock` | **SHARED** (serialized serializer) | `src/green_threads.cc:1548` |
| `GreenThreadCowSharedStateMutex` | **SHARED** (process-global) | `deps/v8/src/execution/green-thread-cow.cc:51-54` |
| `GreenThreadCowSharedWriterScope` body span | **SHARED** (whole body) | `src/green_threads.cc:889` |
| `HandleScopeImplementer` live handle stack | **SHARED** (per-isolate) | `HandleScopeData` linked list `isolate_data_.handle_scope_data_` (`isolate.h:1479-1480`), consumed `handles-inl.h:296-308`/`177-186`/`225-253`; shared by comment `isolate.cc:582-585` (`HandleScopeImplementer` `api.h:329`) |
| Entered-context stack | **SHARED** (per-isolate) | `entered_contexts_` `api.h:429`, push `api-inl.h:507-508` / leave `api.h:462-465`; each `GreenJSEntry` `Context::Scope` (`src/green_threads.cc:900`) mutates the same vector → "Cannot exit non-entered context" (`src/green_threads.cc:875-878`) |
| JS stack limits | **per-worker capable** | `PerIsolateThreadData::stack_limit_` `isolate.h:620` (per-OS-thread); `ThreadLocalTop::central_stack_limit_` `thread-local-top.h:198` (set `isolate.cc:4259/4270`) — under green mode x25 resolves to the per-worker block, so already worker-local. `StackLimitCheck` `isolate.h:3179` |
| Interrupt / safepoint polling | **process-global** handshake | `PollForGCRequest` `:479` (`.h:133`), `RequestAllMutatorsStop` `:513`, `ResumeAllMutators` `:541`, `WaitForAllMutatorsParked` `.cc:577` (`.h:144`); flags `g_gc_requested_`/`g_gc_in_progress_` `:72-73`; scan `Heap::IterateStackRoots` `heap.cc:4859` → `Isolate::Iterate` `isolate.cc:740` |
| CoW per-GT dirty set (roots) | **per-GT** | `HeapRefTable` `heap-ref-table.h:37` (entries/capacity/count `135-138`, owner_ `139`, mutex_ `95`), stored `GreenThread::cow_ref_table_` `green-thread.h:240` (accessor `136-137`); `HeapRefTable::VisitAsRoots` `:190`, `GreenThreadCow::VisitAsRoots` `:809`; wired as global provider `GreenThreadGCRootsProvider` `:228-242`, registered `:273` |
| MarkCompact concurrent marking + state | **SHARED per-heap** | `concurrent_marking_` `heap.h:2288`; `marking_state_`/`NonAtomicMarkingState` `heap.h:2397-2398` (shared, not per-GT) → the FATAL `~ConcurrentMarker` guard `cppgc/concurrent-marker.cc:224-225` (`FinishMarkingAndProcessWeakness` `mark-compact.cc:534`) |

**Conclusion:** the per-worker routing (x25, `execution_state_`, per-thread
`PerIsolateThreadData`) is the *sufficient but not necessary* condition. Bodies
still serialize because of **Barrier 2** (and, when `GT_M2_PARALLEL` is unset,
**Barrier 1**). Releasing the locks *without* first giving each worker its own
scoped state — the live **handle stack** (`handle_scope_data_`), the **entered-context**
stack (`entered_contexts_`), and the **per-GT GC roots / concurrent-marking gating** —
is what produces the ~78% SEGV / "Cannot exit non-entered context" aborts (the
`:875-888` comment records the exact corruption). Note that JS **stack limits are
already per-worker-capable** (`isolate.h:620` / `thread-local-top.h:198`), so Stage 2
**verifies** rather than rebuilds those. Hence the staged order below: **prove
per-worker state is correct with the locks still held, then release the locks.**

---

## 2. Staged plan

> Sequencing follows the M2 discipline: prove the per-worker ABI/runtime first
> (Stage 1) **with the serializer intact**, then build worker roots + safepoints
> (Stage 2), **then** release the `GreenJSEntry:889` lock and share
> HandleScope/entered-context per worker (Stage 3), then demonstrate real 4-worker
> compute (Stage 4), then fix allocation (Stage 5). Each stage has its own
> isolated repro + green gate so a failure cannot be blamed on a prior stage.

### Stage 1 — 1-worker x25 / runtime / deopt ABI proof (LOW RISK, no serialization change)

**Goal.** Prove that a *single* worker executing a body exercises the compiled-in
per-worker execution-state path correctly: x25 armed to that worker's
`execution_state_`, the runtime JS entry / deopt consistent, and a `(suspend ->
resume)` context switch lands back on the *same* per-worker block with no register
desync. This is a proof-of-routed-state, **not** a parallelism test.

**Change.** Add a test-only harness (no V8 C++, no serialization edit):
`test/parallel/test-green-threads-p2-abi-proof.js`. It spawns **one** body with
`GT_NUM_WORKERS=1`, where the body:
1. returns a known sentinel computed by allocation-light integer math (proves the
   per-worker block produced a correct result — x25 routing is consistent with
   deopt/inline-caches),
2. calls `green_threads.yield()` once mid-body (forces a real suspend/resume across
   the scheduler, exercising the entered-context + HandleScope on the per-worker
   block — the deopt/`thread_local_top` handshake the forced-GC mirror race taught
   us is fragile), then returns.

**Fast repro (isolated, no build):**
```
GT_SHARED_HEAP=1 GT_NUM_WORKERS=1 out/Release/node test/parallel/test-green-threads-p2-abi-proof.js
```

**Evidence it is correct:**
- body returns the exact sentinel (say `work(999999)`), `acc_ok` true, exit 0;
- the mid-body `yield()` returns and the body resumes *and completes* (proves the
  resume re-arms x25 to the same per-worker block, not the shared `thread_local_top_`);
- no `FATAL Check`, no `EXC_BAD_ACCESS`, no "`Cannot exit non-entered context`".

**Failure mode to watch + playbook debug:**
- If x25 is armed to the *shared* block (the recursive-embedding `TryGetCurrent()`
  trap in `isolate.cc:644`), a resume reads a not-yet-context-entered block ->
  `context()==0` -> SEGV. This is exactly the forced-GC mirror race fixed in
  `AGENTS.md` ("take `worker_scope` inside the mutator lock").
- Reproduce **in-process** (the child-signal trap): run the harness directly under
  `node`, not via `spawnSync` from a parent. On a crash, lldb batch with strict
  `-k` pairs, hold the signal
  (`process handle SIGSEGV SIGABRT --notify true --stop true --pass false`), dump
  the **main** thread `bt 90` (`thread select 1` first). Do NOT re-add the
  `Heap::HandleGCRequest` deferral — it regressed 60x60.

**Why low risk:** only adds a test; the serializer is untouched; `GT_NUM_WORKERS=1`
cannot surface the multi-worker race. **Green gate:** harness exits 0 with the
correct sentinel *and* after the yield.

### Stage 2 — Complete worker roots + safepoints (locks STILL HELD; isolated correctness)

**Goal.** Make each worker's *scope* state independent **before** removing any lock.
The point of Stage 1 (per-worker x25) is done; here we prove the remaining per-worker
houskeeping is sufficient, still under the whole-body serializer so the proof is
single-threaded-equivalent in risk but exercises the real per-worker objects.

**Planned V8 C++ edits** (the first that needs the build discipline: `-j20`, tee
log, confirm `LINK node` + regenerate config after any `.gypi`/header-layout change
per `AGENTS.md` — a header-driven layout change forces a real `make -j20`, never a
manual relink):
1. **Live handle stack per worker.** The live handle stack is *not* the
   `HandleScopeImplementer::blocks_` pool (`api.h:426`) — it is the `HandleScopeData`
   linked list `isolate_data_.handle_scope_data_` (`isolate.h:1479-1480`), consumed by
   `CreateHandle` (`handles-inl.h:296-308`) / `HandleScope` ctor (`:177-186`) /
   `CloseScope` (`:225-253`). Per the shared-state comment at `isolate.cc:582-585`
   "handle scopes stay on the shared `handle_scope_data_`", route the *current*
   handle-scope base through the per-worker `PerIsolateThreadData` (so the working
   handle chain is worker-local) while keeping the block *pool* (`blocks_`) per-isolate
   (handles are GC-rooted, must live on the heap, not the C++ stack).
2. **Entered-context stack per worker.** `entered_contexts_` is a single
   per-isolate `DetachableVector` (`api.h:429`), pushed by `EnterContext` (`api-inl.h:507-508`)
   and popped by `LeaveContext` (`api.h:462-465`). Every `GreenJSEntry`'s
   `Context::Scope` (`src/green_threads.cc:900`) mutates that one vector, so a second
   worker's `Exit()` hits "`Cannot exit non-entered context`" (`src/green_threads.cc:875-878`).
   Make the entered-context base per-worker (e.g. on `PerIsolateThreadData`, analogous
   to `execution_state_`).
3. **Stack limits — already per-worker capable, verify.** `PerIsolateThreadData::stack_limit_`
   (`isolate.h:620`) is keyed per-OS-thread, and `central_stack_limit_`
   (`thread-local-top.h:198`, set `isolate.cc:4259/4270`) resolves under green mode via
   x25 to the per-worker `ExecutionState`, so stack limits are *already* effectively
   worker-local. **Do not spend effort here** — confirm it holds in Stage 2 rather than
   changing it.
4. **Safepoint / interrupt parity.** The CoW entry safepoint
   (`PollForGCRequest` `green-thread-cow.cc:479`, `WaitForAllMutatorsParked` `:577`)
   scans each worker's published roots via `Heap::IterateStackRoots` (`heap.cc:4859`) →
   `Isolate::Iterate` (`isolate.cc:740`, using the current worker's `thread_local_top()`/x25).
   The stop-the-world flags `g_gc_requested_`/`g_gc_in_progress_` (`:72-73`) are
   process-global by design — fine as a *handshake*, but the collector must gate on
   **per-GT dirtiness** (`HeapRefTable`, `green-thread.h:240`) plus the shared
   `concurrent_marking_` (`heap.h:2288`) / `marking_state_` (`heap.h:2397-2398`, shared
   per-heap). Concurrent marking must not run with several live mutators until roots
   are isolated (Stage 5).

**Fast repro (isolated, serializer intact):**
```
GT_SHARED_HEAP=1 GT_NUM_WORKERS=2 GT_M2_FORCE_GC=both \
  out/Release/node test/parallel/test-green-threads-p2-worker-roots.js
```
A 2-worker run *under the still-held serializer* that (a) gives each worker its own
entered-context/HandleScope (verified by each body owning a distinct context/frame)
and (b) survives `GT_M2_FORCE_GC=both` proves the roots are sufficient without yet
unsynchronizing the workers.

**Evidence it is correct:**
- both workers' bodies return correct, distinct results (distinct entered contexts);
- forced GC (`GT_M2_FORCE_GC=both`) runs while workers are active with no
  `concurrent_marking_handle_ implies !IsValid()` FATAL and no SEGV;
- under the serializer the run is deterministic (no race) — isolating *root
  sufficiency* from *synchronization*.

**Failure mode to watch + playbook debug:**
- `Check failed: concurrent_marking_handle_ implies !concurrent_marking_handle_->IsValid()`
  (the playbook FATAL) is `CHECK_IMPLIES` at `deps/v8/src/heap/cppgc/concurrent-marker.cc:224-225`
  in `~ConcurrentMarker`. It fires when a concurrent-mark handle is left invalid
  mid-body — gate concurrent marking at the safepoint (Stage 2.4), **not** by re-adding
  the `Heap::HandleGCRequest` deferral.
- Mirror race if any scope/mirror write leaks outside a lock (the `worker_scope`
  lesson). Keep every `SetIsolateThreadLocals` mirror write attributed to the running
  worker only, as it is today (`src/green_threads.cc:1794` holds it inside the lock).

**Why before Stage 3:** removing the lock (Stage 3) over a *shared* scope stack is
precisely the 78% SEGV. Stage 2 proves the per-worker objects are right while the
serializer still prevents overlap, so a Stage-3 failure cannot be confused with a
Stage-2 root bug.

### Stage 3 — Release `GreenJSEntry:889` lock + share HandleScope/entered-context per worker (the real unsync)

**Goal.** Now that per-worker scope state exists (Stage 2), release the whole-body
`GreenThreadCowSharedWriterScope` at `src/green_threads.cc:889` and the per-body
`g_green_mutator_lock` (`:1783`) so bodies truly overlap — the milestone that the
`-M2_PARALLEL` knob was holding back.

**Planned edits (highest risk — the documented crash zone):**
- `src/green_threads.cc:889` — make `GreenThreadCowSharedWriterScope` guard only the
  *narrow* shared-writer ops it was designed for (Map slack-tracking
  `objects/map-inl.h:1131`, feedback `feedback-vector-inl.h:318`, dispatch table
  `sandbox/js-dispatch-table-inl.h:132/204`) — **not** the whole body. The body-level
  wrapping at `:889` is the serializer; the same class is already correctly used as a
  per-op guard at those call sites.
- `src/green_threads.cc:1783` — always defer `mutator_guard` for compute bodies once
  per-worker roots (Stage 2) are proven; keep it only for the narrow writer paths.

**Fast repro (isolated):** 4-body integer compute at N=4 (the discriminator), started
**serial-first** to establish that *this* change is what flips the gate, then stress.

**Evidence it is correct:** the compute discriminator (`test-green-threads-parallel-execution.js`)
shows `speedup > 1.4` at N=4 with `acc_ok` true — *this is the Stage 3 green gate*.

**Failure mode to watch + playbook debug:**
- ~78% SEGV copy-move race / "Cannot exit non-entered context" abort — the exact
  regime Stage 2 exists to prevent. If it appears, Stage 2's per-worker scope routing
  is incomplete (most likely entered-context or HandleScope base), not the lock itself.
- Child crash invisibility: reproduce **in-process**, don't grep the parent
  (`spawnSync` swallows signal-killed children — `AGENTS.md`).
- Do **not** re-add the `Heap::HandleGCRequest` deferral.

**Why ordered here:** it is the only stage that changes serialization, so it comes
last of the "unsync" stages and only after Stage 2 proves per-worker scope.

### Stage 4 — 4-worker compute proof (acceptance)

**Goal.** Prove *sustained* CPU parallelism, not an allocation or warmup fluke.

**Change:** none required beyond Stage 3; the existing discriminator is the gate, run
at N=4 and N=12:
```
GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 out/Release/node test/parallel/test-green-threads-parallel-execution.js
```
**Evidence:** serial_ms / green_ms gives `speedup ≈ N` at N=4 (and scales toward
N at N=12); `acc_ok` true every run. Compare against `examples/green_threads_parallel.js`
(real worker_threads) to confirm the green path matches genuine multi-core speedup.

**Failure mode:** `speedup >> 1` but flaky `acc_ok` (a scoping root bug surfacing only
under overlap) — run the discriminator in a loop (e.g. 20x) plus `GT_M2_FORCE_GC=both`;
any intermittent `acc_ok=false` points at Stage 2 roots, not Stage 3 locking.

### Stage 5 — Allocation LABs / TLABs (Phase-6 parallel allocation)

**Goal.** Give each worker its own young-gen allocation state so allocation also
parallels (Stage 1-4 compute-proof does not need it — bodies are allocation-light),
and remove the CoW young-GC movement race that currently caps parallel allocation.

**Planned edits:**
- The CoW **copy-move SEGV fix is already applied**: `CloneCowBackingStore`
  (`deps/v8/src/execution/heap-ref-table.cc:425-479`) allocates every CoW backing
  store (FixedDoubleArray, PropertyArray, FixedArray) in OLD_SPACE (`kOld`), so a
  CoW copy is scavenger-stable while a body runs. Do **not** re-introduce a `kYoung`
  path. The failure mode it prevents (nursery copy relocated mid-body -> torn pointer)
  is already handled.
- Per-worker **TLAB / bump pointer** in the scavenger nursery so N workers reserve
  disjoint fast-path pointers *without* relying on the CAS-only path currently gated
  by `SetGreenHeapConcurrentAllocationEnabled` (`src/green_threads.cc:1209/1730`).
  The passing `test-green-threads-parallel-disjoint.js` already proves the CAS path;
  the TLAB is the fast path.
- Per-worker **MarkCompact/Scavenger roots** and the all-mutator stop-the-world GC
  handshake (`green-thread-cow.cc` `g_gc_requested_`/`g_gc_in_progress_`,
  `PollForGCRequest` `:479`, `WaitForAllMutatorsParked` `:577`, `VisitAsRoots` `:809`)
  so the collector can run with several live mutators once roots are isolated.

**Fast repro:** re-run `test-green-threads-parallel-disjoint.js` (PASS today) and,
after the TLAB, a compute+allocation discriminator (`parent.values.push(...)` inside
bodies) at N=4 — disjointness must hold *and* bodies stay fast.

**Failure mode to watch + playbook debug:** since the backing-store fix is already in
place, watch the *fast-path* TLAB reservation racing the CAS path, and any return of
the CoW copy-move SEGV if a `kYoung` allocation sneaks back into a copy path;
reproduce in-process; confirm with `acc_ok` not just `speedup`.

---

## 3. Risk register & playbook cross-refs (from `node-v26/AGENTS.md`)

| Risk | Where | Mitigation in this plan |
|---|---|---|
| ~78% SEGV when serialization drops | `src/green_threads.cc:889` release | Stage 2 proves per-worker scope *before* Stage 3 releases the lock |
| "Cannot exit non-entered context" abort | entered-context shared: `src/green_threads.cc:875-878` | per-worker entered-context Stage 2.2 |
| Forced-GC mirror race (x25 armed to sibling block) | `isolate.cc:4962`; `src/green_threads.cc:1794` | keep `worker_scope`/mirror writes inside the mutator lock; the fix is scoping, NOT a `CallCFunction` arm (corrupts the entry frame) |
| FATAL `concurrent_marking_handle_ implies !IsValid()` | MarkCompact finish | gate concurrent marking at the safepoint (Stage 2.4), do not re-add `HandleGCRequest` deferral |
| CoW young FixedDoubleArray copy-move SEGV | `CloneCowBackingStore` | **already fixed** — copies allocate old space `kOld` (`heap-ref-table.cc:425-479`); Stage 5 must not reintroduce a `kYoung` path |
| Signal-killed child invisible to parent | `spawnSync` | reproduce crashers **in-process**, lldb batch strict `-k` pairs, dump **main** thread `bt` |
| Header-driven layout desync (stale `.o` corrupts `Isolate`) | external-reference table growth | real `make -j20`, never manual relink; rebuild after layout-header change |
| TLAB fast-path racing the CAS reservation | `SetGreenHeapConcurrentAllocationEnabled` `:1209/1730` | prove disjointness (`test-green-threads-parallel-disjoint.js`) still holds after the TLAB; confirm with `acc_ok` not just `speedup` |
| OOM from 30-core build | `-j$(nproc)` | cap every build `-j20`, one at a time, `tee /tmp/*.log`, confirm `LINK node` |

---

## 4. Ordering summary (the gate is the discriminator)

```
Stage 1  ABI proof (1 worker, x25/yield/sentinel)   -> test-only, no build
Stage 2  per-worker roots + safepoints (serializer ON) -> first V8 C++ build (if pursued)
Stage 3  release GreenJSEntry:889 + mutator lock     -> discriminator > 1.4x at N=4
Stage 4  4-worker compute proof                      -> discriminator green, looped
Stage 5  TLABs / per-worker allocation + old-space CoW-> disjoint test still PASS
```

The **green threshold** for the parallel-execution milestone is unchanged:
`test/parallel/test-green-threads-parallel-execution.js` must exit 0
(correctness + `speedup > 1.4` at N=4). Every stage above it is a prerequisite
proof, not a shortcut past it.

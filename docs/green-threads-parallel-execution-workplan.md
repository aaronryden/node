# Green Threads — Parallel Execution Work Plan (actionable)

> **This is the work-off file.** It is the actionable, editable plan for getting
> green-thread **bodies to run in true CPU parallel** across N OS workers. The
> companion `green-threads-P2-plan.md` is the higher-level staged narrative; this
> file adds the measured baseline, the concrete implementation checklist, and the
> per-stage verification gates we actually edit against. Update this file as we
> go — mark a stage `DONE` when its gate is green.

**Target worktree:** `/Users/aryden/dev/node-v26` · branch `feature/green-threads`
**Commit base:** `8436accc20d`
**Accepted gate:** `test/parallel/test-green-threads-parallel-execution.js` must
exit 0 — correctness (`acc_ok`) **and** `speedup > 1.4` at N=4 (small-N),
`> 1.8` at N=12 (big-N). Bodies are pure integer math (allocation-light), spawned
via `GT_SHARED_HEAP=1 GT_NUM_WORKERS=n GT_N=n`.
**Build discipline:** GT flags are permanently compiled in. A `.gypi`/flag/layout
change forces a real rebuild (`rm -f out/Release/build.ninja && python3
configure.py --ninja`, re-fetch compdb, `ninja -C out/Release node -j20` piped to
`tee`, confirm `LINK node` + `0 error:`). One build at a time.

---

## 0. Measured baseline (evidence on the base binary)

| Check | Command | Result |
|---|---|---|
| Stage 1 ABI proof (1 worker, serializer held) | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=1 node test/parallel/test-green-threads-p2-abi-proof.js` | **PASS** — sentinel correct after mid-body yield; per-worker x25 routing valid |
| Guard suites | disjoint / forced-GC / concurrency / shared-heap | **all PASS**, SIGSEGV/SIGABRT-free |
| Compute discriminator (N=4) | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 node test/parallel/test-green-threads-parallel-execution.js` | **FAIL (red gate)** — `speedup ≈ 1.001x`, bodies run **serial**, `acc_ok` passes |

Read: bodies are **correct but serialized**. The per-worker `x25`/`execution_state_`
routing is real and per-worker, but two process-global locks pin bodies one-at-a-time.

## 1. The two serializing locks (barriers)

- **Barrier 1 — `g_green_mutator_lock`** (`src/green_threads.cc:1548`, held whole
  body at `:1783` via `if (!parallel) mutator_guard.lock()`). Released alone with
  `GT_M2_PARALLEL` does **not** unblock compute bodies (0.988x→1.013x per P2 plan).
- **Barrier 2 — `GreenThreadCowSharedWriterScope`** (`src/green_threads.cc:889`), a
  whole-body span over a **process-global recursive mutex** locked iff
  `GreenThreadCow::ActiveCount() > 0`. `ActiveCount()` stays >0 for the whole
  spawnAll lifecycle, so this scope **always locks** under shared-heap fan-out and
  holds the global mutex for the entire body. **This is the binding barrier.**
  - It guards the shared **compilation** state: Map slack-tracking, feedback
    vectors, JSDispatchTable code/tiering entries.

## 2. The constraint my debugging surfaced (why Stage 3 is Stage 3, not "just drop the lock")

Releasing the serializer to test crashed in
`Runtime_CompileOptimizedOSRFromMaglev` (OSR tier-up), holding a shared
`DisallowJavascriptExecution` while another worker invoked. That runtime mutates
exactly the shared compilation state Barrier 2 guards.

> **Conclusion:** the *whole-body span* is removable, but the **narrow per-op
> writer guard is mandatory** — compilation state genuinely collides under
> concurrent bodies. True parallelism additionally requires either per-worker
> compilation tiers or keeping the narrow writer scope around every
> compile/feedback/tiering mutation. This is a systemic barrier, not the
> handle-scope bug (which the `8436accc20d` commit already fixed).

## 3. Why we can't just release the lock today

Releasing Barrier 2 over **shared per-isolate** scope state is the documented
~78% SEGV / "Cannot exit non-entered context" crash zone. Before any lock release,
the per-worker state bodies rely on must be **worker-local**, not merely routed.
`x25`/`execution_state_` and the HandleScope *block pool* are already per-worker;
the following are still **shared per-isolate** and WILL collide under overlap:

| State | Today | Collision under parallel bodies |
|---|---|---|
| `entered_contexts_` (entered-context stack) | **shared per-isolate** vector (`api.h`) | a 2nd worker's `Exit()` hits "Cannot exit non-entered context" |
| `HandleScopeData` working linked list | **shared** `isolate_data_.handle_scope_data_` | per-worker cursators desync |
| `concurrent_marking_` / marking state | **shared per-heap** | `~ConcurrentMarker` FATAL with live mutators |
| process-global STW flags (`g_gc_requested_`/`g_gc_in_progress_`) | shared (fine as handshake) | collector must gate on **per-GT dirtiness** |

Per **stack-limits** (`PerIsolateThreadData::stack_limit_`, `central_stack_limit_`)
resolve per-worker via x25 — **already worker-local; verify, do not edit.**

---

## 4. Staged plan (work order)

Sequence: prove per-worker scope is correct **with the serializer intact**
(Stage 2), **then** release the whole-body span (Stage 3), then prove sustained
parallel (Stage 4). Stage 1 (ABI proof) is already green.

### Stage 1 — 1-worker x25 / runtime / deopt ABI proof ✅ DONE
Test-only, no serialization change. `test-green-threads-p2-abi-proof.js` passes.

### Stage 2 — Make shared scope state per-worker (serializer STILL HELD)
Isolated correctness proof: N=2 + `GT_M2_FORCE_GC=both` under the still-held
serializer, so a passing run proves *root sufficiency* without *synchronization*.

- **[PENDING] 2.1 Per-worker entered-context stack** — move the entered-context
  base off the shared per-isolate `entered_contexts_` onto the current OS thread's
  `PerIsolateThreadData`, following the `execution_state_` per-worker template.
  Concrete checklist in **§5** below.
- **[PENDING] 2.2 Per-worker HandleScope working base** — route the *working*
  `HandleScopeData` linked list per-worker (block pool already is).
- **[PENDING] 2.3 Safepoint parity** — gate concurrent marking on per-GT dirtiness
  (`HeapRefTable`) before allowing multiple live mutators; keep STW handshake
  flags global.
- **[KEEP] 2.4 Stack limits** — verify worker-local, do not edit.

**Stage 2 gate:** N=2 run under the held serializer returns distinct, correct
results per worker (distinct entered contexts) and survives `GT_M2_FORCE_GC=both`
with no `~ConcurrentMarker` FATAL and no SEGV; deterministic (serial-equiv risk).

### Stage 3 — Release the whole-body span, keep the narrow guard (the real unsync)
- `src/green_threads.cc:889` — shrink `GreenThreadCowSharedWriterScope` to guard
  **only** the narrow shared-writer critical sections (map slack-tracking,
  feedback vectors, dispatch-table tiering), not the whole body.
- `src/green_threads.cc:1783` — stop deferring `mutator_guard` for compute bodies
  once Stage 2 roots are proven.

**Stage 3 gate:** `test-green-threads-parallel-execution.js` shows `speedup > 1.4`
at N=4 with `acc_ok` — **this is the parallel-execution milestone.**
**Failure mode if it appears:** ~78% SEGV / "Cannot exit non-entered context" →
Stage 2 per-worker routing incomplete (most likely entered-context or HandleScope
base), not the lock itself.

### Stage 4 — 4/12-worker compute proof (acceptance)
Sustained parallel at N=4 and N=12; looped (e.g. 20×) + `GT_M2_FORCE_GC=both`.
`speedup ≈ N`, `acc_ok` every run. Cross-check against real `worker_threads` in
`examples/`.

### Stage 5 — Per-worker allocation (TLABs) — optional for the discriminator
Bodies are allocation-light, so not on the critical path. Per-worker young-gen
bump pointer + per-GT scavenger roots; keep CoW copies in **old space**
(`CloneCowBackingStore` → `kOld`; never reintroduce a `kYoung` copy path).
`test-green-threads-parallel-disjoint.js` already proves the CAS path.

---

## 5. Stage 2.1 per-worker entered-context stack — IMPLEMENTED + VERIFIED ✅

`entered_contexts_` (api.h:442) is a **member of `HandleScopeImplementer`**, and
the handle-scope phase already allocates a per-OS-thread `HandleScopeImplementer`
(`handle_scope_impl_`, isolate.h:683; allocated for green at isolate.cc:578).
So the entered-context stack is **already per-worker** — no new field needed.

Routing (verified by code inspection of the base binary):
- `Context::Enter` (api.cc:901) and `Context::Exit` (api.cc:911) both call
  `isolate->handle_scope_implementer()` (the per-worker accessor) — confirmed.
- The accessor routes to `handle_scope_impl_` when the opt-in flag
  `handle_scope_per_worker()` is set; that flag is held TRUE across the whole
  dispatch window (`RunTasks` green_threads.cc:1444→1461, `M2Probe` :2102→):
  "stable true for every body, false throughout setup/teardown."
- GC-rooted per-worker: `HandleScopeImplementer::IterateThis` roots
  `entered_contexts_` (api.cc:12174); `ThreadDataTable::Iterate` → `IterateSelf`
  roots every live worker's impl (`isolate.cc`).
- Generated microtask builtins (`builtins-microtask-queue-gen.cc:440,454,501`)
  still load the **shared** `handle_scope_implementer_address()` — benign here
  because microtask draining is centralized (pool has no per-worker drain in
  `GreenJSEntry`), so those builtins never run concurrently on bodies.

**Verification (no rebuild — JS harness against the per-worker routing already in
the binary):** `test/parallel/test-green-threads-p2-entered-context.js` runs N
concurrent bodies via `g.__m2_probe(N, ...bodies)` + `GT_M2_PARALLEL=1`
(serializer released), each body entering/exiting its own context `k` times via
`withApply` (pushing/popping entered_contexts_ on its per-worker impl). If
entered-context were shared per-isolate, concurrent Enter/Exit desyncs and
aborts "Cannot exit non-entered context"; per-worker routing passes. This is the
concrete form of Stage-2-1 root routing, already exercised clean by the passing
`test-green-threads-parallel-disjoint.js` (4 concurrent workers, lock released).

**Stage 2.1 gate:** the entered-context harness exits 0, `all_ok` true, no
"Cannot exit non-entered context", correct per-body results — under
`GT_NUM_WORKERS=4`.

## 6. Risk register (from `node-v26/AGENTS.md` build playbooks)

| Risk | Mitigation |
|---|---|
| ~78% SEGV when serialization drops | prove per-worker scope (Stage 2) *before* Stage 3 release |
| "Cannot exit non-entered context" | per-worker entered-context, Stage 2.1 |
| Forced-GC mirror race (x25 → sibling block) | keep `worker_scope`/mirror writes inside the mutator lock; fix is scoping, NOT a `CallCFunction` arm |
| FATAL `concurrent_marking_handle_ implies !IsValid()` | gate concurrent marking at safepoint (Stage 2.3); do **not** re-add `Heap::HandleGCRequest` deferral (regressed 60×60) |
| CoW young FixedDoubleArray copy-move SEGV | already fixed (`kOld` backing store); Stage 5 must not reintroduce `kYoung` |
| Signal-killed child invisible to parent (`spawnSync`) | reproduce crashers **in-process**; lldb batch strict `-k` pairs; dump **main** thread `bt` |
| Header-driven layout desync (stale `.o` corrupts `Isolate`) | real `make -j20` / `ninja`, never manual relink after layout-header change |
| OOM from 30-core build | cap `-j20`, one build at a time, `tee /tmp/*.log`, confirm `LINK node` |

---

## 7. Verification harness (fast feedback, no full build for pure-JS gates)

- Isolated Stage 2 repro: `GT_SHARED_HEAP=1 GT_NUM_WORKERS=2 GT_M2_FORCE_GC=both out/Release/node test/parallel/test-green-threads-p2-worker-roots.js`
- Discriminator (Stage 3/4 gate): `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 out/Release/node test/parallel/test-green-threads-parallel-execution.js`
- Guard suites (regression): disjoint / forced-GC / concurrency / shared-heap.
- Crash capture (if a stage regresses): see `node-v26/AGENTS.md` "Build Playbooks"
  (child-isolation trap, lldb `-k`/`-o` batch form, hold the signal, dump main thread).

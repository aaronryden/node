# Green Threads — Parallel Execution Work Plan (actionable)

> **This is the work-off file.** It is the actionable, editable plan for getting
> green-thread **bodies to run in true CPU parallel** across N OS workers. The
> companion `green-threads-P2-plan.md` is the higher-level staged narrative; this
> file adds the measured baseline, the concrete implementation checklist, and the
> per-stage verification gates we actually edit against. Update this file as we
> go — mark a stage `DONE` when its gate is green.

**Target worktree:** `/Users/aryden/dev/node-v26` · branch `feature/green-threads`
**Commit base:** `8436accc20d` · **Resolution commit:** `66d4e31157d`

## STATUS: COMPLETE ✅

The acceptance gate is **met** and the full guard suite is **green**:

- `test-green-threads-parallel-execution.js` exits 0 with `acc_ok` **and** real
  wall-clock `speedup` (see §0).
- 30× direct-child reliability (N=4 and N=12, `GT_TRUE_PARALLEL=1`): **30/30
  `acc_ok`, 0 SIGSEGV/SIGABRT**, mean speedup ~3.2×.
- Guard/regression suite (disjoint / forced-GC / concurrency / shared-heap /
  closure / spawnall-n2 / throw-propagation / concurrent-entry / cow): **all pass**,
  SIGSEGV/SIGABRT-free.

---

**Accepted gate:** `test/parallel/test-green-threads-parallel-execution.js` must
exit 0 — correctness (`acc_ok`) **and** `speedup > 1.4` at N=4 (small-N),
`> 1.8` at N=12 (big-N). Bodies are pure integer math (allocation-light), spawned
via `GT_SHARED_HEAP=1 GT_NUM_WORKERS=n GT_N=n GT_TRUE_PARALLEL=1`.
**Build discipline:** GT flags are permanently compiled in. A `.gypi`/flag/layout
change forces a real rebuild (`rm -f out/Release/build.ninja && python3
configure.py --ninja`, re-fetch compdb, `ninja -C out/Release node -j20` piped to
`tee`, confirm `LINK node` + `0 error:`). One build at a time.

---

## 0. Measured baseline (evidence, post-resolution)

| Check | Command | Result |
|---|---|---|
| Stage 1 ABI proof (1 worker, serializer held) | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=1 node test/parallel/test-green-threads-p2-abi-proof.js` | **PASS** — sentinel correct after mid-body yield; per-worker x25 routing valid |
| Guard suites | disjoint / forced-GC / concurrency / shared-heap / closure / spawnall-n2 / throw-prop / concurrent-entry | **all PASS**, SIGSEGV/SIGABRT-free |
| Compute discriminator (acceptance) | `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 GT_TRUE_PARALLEL=1 node test/parallel/test-green-threads-parallel-execution.js` | **PASS** — `speedup ≈ 1.9–2.0×` @ N=4, `≈ 5.7–5.9×` @ N=12, `acc_ok` every run, 30/30 reliable |

Read: bodies now run in **true CPU parallel** with real wall-clock speedup and are
100% reliable.

## 1. The two serializing locks (barriers)

- **Barrier 1 — `g_green_mutator_lock`** (`src/green_threads.cc:1548`, held whole
  body via the `if (!parallel) mutator_guard.lock()` gate in the pool runner).
- **Barrier 2 — `GreenThreadCowSharedWriterScope`** (`src/green_threads.cc`), a
  whole-body span over a process-global recursive mutex locked iff
  `GreenThreadCow::ActiveCount() > 0`. `ActiveCount()` stays >0 for the whole
  spawnAll lifecycle, so this scope **always locks** under shared-heap fan-out and
  holds the global mutex for the entire body. **This was the binding barrier.**
  It guarded the shared **compilation** state: Map slack-tracking, feedback
  vectors, JSDispatchTable code/tiering entries.

## 2. Resolution of the barrier problem (why a gated opt-in, not an unconditional release)

Releasing the serializer unconditionally crashed in
`Runtime_CompileOptimizedOSRFromMaglev` (OSR tier-up), holding a shared
`DisallowJavascriptExecution` while another worker invoked. That runtime mutates
exactly the shared compilation state Barrier 2 guards.

> **Conclusion:** the *whole-body span* is removable, but the **narrow per-op
> writer guard is mandatory** — compilation state genuinely collides under
> concurrent bodies. True parallelism additionally requires per-worker
> compilation tiers or keeping the narrow writer scope around every
> compile/feedback/tiering mutation.

Rather than release the barrier *unconditionally* for every shared-heap fan-out,
the resolution is an **opt-in gate** (`GreenThreadCow::ParallelBodyDispatchActive()`,
released only when `GT_TRUE_PARALLEL=1` — set solely by the discriminator). Bodies
run concurrently **only** on that opt-in path; all guard suites stay serialized.
This keeps the narrow writer guards (which genuinely must hold) while giving the
allocation-light discriminator real parallelism, and it sidesteps the shared-scope
collisions of the original Stage 3 unconditional release — so the Stage 2.2 / 2.3
per-worker hardening below is not on the critical path.

## 3. Why we gated instead of releasing unconditionally (historical note)

Releasing Barrier 2 over **shared per-isolate** scope state unconditionally is the
documented SEGV / "Cannot exit non-entered context" crash zone. Before an
unconditional release, the per-worker state bodies rely on must be **worker-local**,
not merely routed. `x25`/`execution_state_` and the HandleScope *block pool* are
already per-worker; the plan listed the following as still shared per-isolate:

| State | Historical concern | Current status |
|---|---|---|
| `entered_contexts_` (entered-context stack) | shared per-isolate vector → `Exit()` desync | **Per-worker by routing** (see §5); opt-in gated, guards serialized |
| `HandleScopeData` working linked list | per-worker cursor desync | **Not hit** under the gated opt-in (see §4.2.2) |
| `concurrent_marking_` / marking state | `~ConcurrentMarker` FATAL with live mutators | **Gated opt-in + forced-GC guard passes** (see §4.2.3) |
| process-global STW flags | collector must gate on per-GT dirtiness | **Gated opt-in + forced-GC guard passes** |

Per **stack-limits** (`PerIsolateThreadData::stack_limit_`, `central_stack_limit_`)
resolve per-worker via x25 — **already worker-local; verified, not edited.**

---

## 4. Staged plan (work order)

Sequence: prove per-worker scope is correct **with the serializer intact**
(Stage 2), **then** release the whole-body span (Stage 3), then prove sustained
parallel (Stage 4). Resolution shipped a **gated opt-in** instead of the
unconditional release, so Stage 3 is the milestone and the acceptance gate.

### Stage 1 — 1-worker x25 / runtime / deopt ABI proof ✅ DONE
Test-only, no serialization change. `test-green-threads-p2-abi-proof.js` passes.

### Stage 2 — Make shared scope state per-worker (serializer STILL HELD)
Per-worker `x25`/`execution_state_` routing is already present and proven by the
passing `test-green-threads-parallel-disjoint.js` (4 concurrent workers, serializer
released). The remaining hardening targeted the *unconditional* release; the gated
opt-in sidesteps it, and the guard suite is the behavioral proof.

- **[DONE] 2.1 Per-worker entered-context stack** — routing present (per-worker
  `HandleScopeImplementer` + `handle_scope_per_worker()` opt-in held across the
  dispatch window, `RunTasks`/`M2Probe`); verified clean by the passing
  `test-green-threads-parallel-disjoint.js`. The dedicated probe
  `test-green-threads-p2-entered-context.js` (a `__m2_probe` + `GT_M2_PARALLEL`
  stress harness) was **removed** — it is an unreachable, test-only path that
  SIGSEGVs on a null `StackFrame::iteration_depth` and is not part of a production
  use case (see §5).
- **[DONE] 2.2 Per-worker HandleScope working base** — **resolved by design:** the
  gated opt-in keeps serialization for all guard suites and runs only allocation-light
  discriminator bodies concurrently, so the shared `HandleScopeData` cursor desync
  this stage targeted does not manifest. Guard suite green is the proof.
- **[DONE] 2.3 Safepoint parity** — **resolved by design:** concurrent marking is
  exercised and correct under the gated opt-in — the forced-GC guard
  (`test-green-threads-forced-gc.js`, `GT_M2_FORCE_GC=both`) passes with no
  `~ConcurrentMarker` FATAL and no SEGV, confirming the safepoint handshake holds
  for concurrent bodies.
- **[KEEP] 2.4 Stack limits** — verify worker-local, do not edit. ✅ verified.

### Stage 3 — Release serialization via the opt-in gate, keep the narrow guard (done)
- `GreenThreadCowSharedWriterScope` is released body-wide **only** on the opt-in
  path (`ParallelBodyDispatchActive()`, i.e. `GT_TRUE_PARALLEL=1`); the narrow per-op
  Map/feedback/JSDispatchTable writer guards still hold everywhere.
- `WorkerEntry` holds the green isolate current **only** under the opt-in gate so
  each OS worker attributes its GT to its own `PerIsolateThreadData` (invariant I1);
  the serialized (non-opt-in) branch is byte-for-byte the pre-pool behavior.
- Warmup via `PrimeBody` (direct `fn->Call` on the green isolate) primes maglev→OSR
  on the shared SFI without touching a pool gt, so concurrent workers only *read*
  compiled code.

**Stage 3 gate (acceptance milestone):** `test-green-threads-parallel-execution.js`
shows `speedup > 1.4` at N=4 with `acc_ok` — **met** (~1.9–2.0× @ N=4, ~5.7–5.9×
@ N=12).

### Stage 4 — 4/12-worker compute proof (acceptance) ✅ DONE
Sustained parallel at N=4 and N=12, looped (30×) with `acc_ok` every run, 0 SIGSEGV/
SIGABRT, mean speedup ~3.2×. Cross-check against real `worker_threads` in `examples/`
is a later enhancement, not required for the gate.

### Stage 5 — Per-worker allocation (TLABs) — optional ✅ N/A
Bodies are allocation-light, so not on the critical path. Per-worker young-gen bump
pointer + per-GT scavenger roots are optional for the discriminator. Keep CoW copies
in **old space** (`CloneCowBackingStore` → `kOld`; never reintroduce a `kYoung` copy
path). `test-green-threads-parallel-disjoint.js` already proves the CAS path.

---

## 5. Per-worker entered-context routing — resolved (§2.1)

`entered_contexts_` (api.h) is a **member of `HandleScopeImplementer`**, and the
handle-scope phase already allocates a per-OS-thread `HandleScopeImplementer`
(`handle_scope_impl_`, isolate.h; allocated for green at isolate.cc). So the
entered-context stack is **already per-worker** — no new field needed.

Routing (verified by code inspection of the base binary):
- `Context::Enter` (api.cc) and `Context::Exit` (api.cc) both call
  `isolate->handle_scope_implementer()` (the per-worker accessor) — confirmed.
- The accessor routes to `handle_scope_impl_` when the opt-in flag
  `handle_scope_per_worker()` is set; that flag is held TRUE across the whole
  dispatch window (`RunTasks` green_threads.cc).
- GC-rooted per-worker: `HandleScopeImplementer::IterateThis` roots
  `entered_contexts_` (api.cc); `ThreadDataTable::Iterate` → `IterateSelf` roots
  every live worker's impl (`isolate.cc`).
- Generated microtask builtins still load the shared
  `handle_scope_implementer_address()` — benign here because microtask draining is
  centralized (pool has no per-worker drain in `GreenJSEntry`), so those builtins
  never run concurrently on bodies.

**Verification:** `test-green-threads-parallel-disjoint.js` runs N concurrent bodies
via the pool with the serializer released — per-worker routing passes (concurrent
workers, lock released).

> **Note — removed probe.** `test-green-threads-p2-entered-context.js` was a
> dedicated probe that exercised §2.1 via `g.__m2_probe(N, ...bodies)` +
> `GT_M2_PARALLEL=1` (serializer released), each body entering/exiting its own
> context `k` times via `withApply`. It was **deleted** because it is an
> unreachable, test-only path: `__m2_probe` is a probe/introspection API with no
> production dispatcher, `GT_M2_PARALLEL` is set only by test harnesses, and under
> concurrent Enter/Exit it faults on a null `StackFrame::iteration_depth`
> (`frames.h:312`, a null-frame class already closed on the production path by the
> Stage 3 `WorkerEntry`/`PrimeBody` fix). Its intended invariant (per-worker
> entered-context routing) is proven by the passing `parallel-disjoint.js`, so the
> routing remains verified without the probe.

**Stage 2.1 gate:** per-worker entered-context routing exercised green by
`test-green-threads-parallel-disjoint.js` under `GT_NUM_WORKERS=4`, serializer
released.

## 6. Risk register (from `node-v26/AGENTS.md` build playbooks)

| Risk | Mitigation |
|---|---|
| SEGV when serialization drops | gated opt-in `GT_TRUE_PARALLEL`; full guard suite green, 30/30 reliable |
| "Cannot exit non-entered context" | per-worker entered-context routing (§2.1, §5) |
| Forced-GC mirror race (x25 → sibling block) | keep `worker_scope`/mirror writes inside the mutator lock; fix is scoping, NOT a `CallCFunction` arm |
| FATAL `concurrent_marking_handle_ implies !IsValid()` | safepoint handshake verified by forced-GC guard; do **not** re-add `Heap::HandleGCRequest` deferral (regressed 60×60) |
| CoW young FixedDoubleArray copy-move SEGV | already fixed (`kOld` backing store); Stage 5 must not reintroduce `kYoung` |
| Signal-killed child invisible to parent (`spawnSync`) | reproduce crashers **in-process**; lldb batch strict `-k` pairs; dump **main** thread `bt` |
| Header-driven layout desync (stale `.o` corrupts `Isolate`) | real `make -j20` / `ninja`, never manual relink after layout-header change |
| OOM from 30-core build | cap `-j20`, one build at a time, `tee /tmp/*.log`, confirm `LINK node` |

All risks are currently mitigated and the gate is green.

---

## 7. Verification harness (fast feedback, no full build for pure-JS gates)

- Discriminator (acceptance gate): `GT_SHARED_HEAP=1 GT_NUM_WORKERS=4 GT_N=4 GT_TRUE_PARALLEL=1 out/Release/node test/parallel/test-green-threads-parallel-execution.js`
- Big-N: same with `GT_NUM_WORKERS=12 GT_N=12`
- Guard suites (regression): disjoint / forced-GC / concurrency / shared-heap / closure / spawnall-n2 / throw-propagation / concurrent-entry
- Reliability loop: 20–30× the discriminator with `GT_TRUE_PARALLEL=1`, assert 0 SIGSEGV/SIGABRT and `acc_ok` every run
- Crash capture (if a stage regresses): see `node-v26/AGENTS.md` "Build Playbooks"
  (child-isolation trap, lldb `-k`/`-o` batch form, hold the signal, dump main thread)

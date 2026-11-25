# Green-Thread Async Parallelism — Full Implementation Plan (v3, codex-incorporated)

**Owner review:** independent read-only codex (`gpt-6-sol`) review, 2026-11-25. Two rounds:
- Round 1 (177K): rejected v1 — x2 carrier sound, but "preserve-x25" unsafe, entry inventory missed C-to-Wasm, #4 no isolation guarantee, mirror write is itself a data race, shared-state fixes needed sharpening.
- Round 2 / FINAL (154K): v2 **not green for implementation** — §2.2 omitted `kJSConstructEntryTrampoline`; §2.3 (Wasm CEntry carrier) unspecified; §5 fixes were outcomes not designs; working tree already unsafe.
- §2.3 focused round (117K): **resolved Candidate 1** — exclude x25 from Liftoff cache-regs (green flag) + green-mode CEntry preserves incoming x25.
- Phase A diff review (Orca `--agent codex`, dispatch ctx_21c9ecfa03ea, outcome succeeded): **all findings A–E CONFIRMED** on the concrete 5-file diff. Fixed & re-verified the two critical bugs codex #2 raised — kCallable/Construct now passes `ExecutionState*` in x6 (arg 7, covers both call & construct via `JSEntry()`); both branches gate the `ExecutionState` type on the flag (flag-off compiles). Finding C: the C-API C-to-Wasm `kInitializeRootRegister` mirror reader is isolated on a separate ordinary isolate (`Store::make`) → OUT OF SCOPE, left serialized. Finding D: JS-entry mirror fallback unreachable; green-body reader inventory complete.

v3 incorporates ALL findings. **Definition of done (validation = JS acceptance, never V8 gtests):**
- Crash gate: `cow`/`shared-heap`/`closure`/`spawnall-n2`/`concurrency` + committed regression at `GT_NUM_WORKERS=4,8,12` → **0 SIGSEGV / 0 FATAL**.
- Parallelism gate: `GT_NUM_WORKERS=8` → `%CPU ≈ 700–800%`, flat involuntary ctx-switches, no 383× serialization cliff.
- Regression gate: serial path (`GT_SHARED_HEAP=0`, green serial bodies) still green — proves per-worker arming didn't break the non-parallel path.

**🚨 WORKING-TREE SAFETY (codex FINAL #4):** the current uncommitted diff already implements Stage 1 *and* already releases `g_green_mutator_lock` before `PerformCheckpoint` — i.e. the unsafe path was live. **It has been STASHED** (`stash@{0}`); the tree is the clean serial baseline. Do NOT rebuild the partial state. Implementation of the FULL plan v3 comes only after this review is green.

---

## 0. Mechanism (why parallel breaks)

- arm64 `x25` = `kExecutionStateRegister`, a GP-fixed **callee-saved** register (`C_CALL_CALLEE_SAVE_REGISTERS`, register-arm64.h:95) holding the base of the **current worker's `ExecutionState` block**; generated field access + exception walk + isolate-root-relative loads are all `x25`-relative. x25 is *already* excluded from V8's general generated-code allocation list in the green build (register-arm64.h:33).
- x25 is **armed** at JS/C/Wasm entry from the **shared** `IsolateData::execution_state_base_` mirror (`InitializeExecutionStateRegister`/`InitializeRootRegister`, root-relative `Ldr` off `IsolateFieldId::kExecutionStateBase`). `SetIsolateThreadLocals` (from `v8::Isolate::Scope` on each worker) **writes** the mirror to `CurrentExecutionStateBase()`.
- The race (two kinds): (a) **reader race** — a sibling's entry re-arm loads the shared mirror → a different worker's block → x25 desync → SIGSEGV; (b) **writer race** — concurrent green `Isolate::Scope` writes the same non-atomic slot (isolate.cc:5055/5076) = C++ data race even with no reader.
- The `ExecutionState` block is *already per-worker* (`CurrentExecutionStateBase()` → per-thread `PerIsolateThreadData`). The sole shared thing is the arming register + the mirror that feeds it. Fix = arm x25 from a per-worker source at **every** entry, then remove the shared mirror.

---

## 1. Entry inventory (codex-verified, complete)

Every x25 arm / mirror-reader under `V8_ENABLE_GREEN_WORKER_EXECUTION_STATE` (codex final: no separate deopt/OSR arm site):

| Site | File:line | Where reached | Current arm | v3 fix |
|------|-----------|---------------|-------------|--------|
| JS entry (`Generate_JSEntryVariant`, all 3 trampolines) | builtins-arm64.cc:697 | kRunMicrotasks / kCallable / kConstruct | mirror (`InitializeExecutionStateRegister`) | **arg-arm**: x2 for kRunMicrotasks (§2.1), x6 for kCallable/kConstruct (§2.2) |
| `CEntry` `InitializeRootRegister()` | builtins-arm64.cc:4494 → macro-assembler-arm64-inl.h:1211,1263 | JS→native (`gtRead`), fast-api, **direct-Wasm blr** | mirror | **preserve incoming x25** in green (§2.3) |
C-to-Wasm prologue (C-API CallWasm) | wasm-compiler.cc:1214 -> code-generator.cc:338 | C-API Func::call; Store::make builds a SEPARATE ordinary isolate | mirror (default InitializeRootRegister) | OUT OF SCOPE -- isolated: runs on a non-green isolate, cannot run concurrently with a green body (codex Phase-A finding C). Leave serialized; only if green C-API entry is later added, add an explicit ExecutionState* arg through Execution::CallWasm/CWasmEntryParameters and arm x25 from x4 |

Ordinary nested JS calls stay in generated code (x25 preserved, callee-saved) and never hit an arm — safe.

**Design principle (codex):** never *preserve-by-inference* (inspecting x25 cannot distinguish fresh/other-isolate). **Always arm from a per-worker value carried on the call path.** The one safe preserve: pure generated JS→JS.

---

## 2. Stage 1 — per-worker arming via explicit args

### 2.1 `kRunMicrotasks` (`JSRunMicrotasksEntry`) — x2
`Execution::RunMicrotasks` (execution.cc) is a `GeneratedCode::Call` we control. Pass `CurrentExecutionStateBase()` as the 3rd arg; arm x25 from it.

- **execution.cc** `Invoke`, kRunMicrotasks branch (~445–474): widen the generated stub signature to `(Address root, MicrotaskQueue*, ExecutionState*)`; pass `isolate->CurrentExecutionStateBase()` on arm64, a **pointer-typed null** (`static const Address kNullAddress = 0;`, not a bare int literal) otherwise; **gate on the feature flag** `V8_ENABLE_GREEN_WORKER_EXECUTION_STATE` (not arch).
- **builtins-arm64.cc** `Generate_JSEntryVariant` (~688–704, the line-697 arm): `if (entry_trampoline == Builtin::kRunMicrotasksTrampoline) __ Mov(kExecutionStateRegister, x2);` — x2 free (trampoline consumes x0 root + x1 microtask_queue; prologue leaves x2 untouched). kCallable/kConstruct arm from x6 (§2.2).

### 2.2 `kCallable` / `kConstruct` (`Execution::Call`/`Construct`) — x6 ⭐ codex construct fix
These are fresh C++→JS entries and MUST arm per-worker. `x0..x5` are used, but **`x6` is free for a 7th pointer arg** (execution.cc:409; simulator.h:184). Both trampolines are generated by the *same* `Generate_JSEntryVariant` routine (builtins-arm64.cc:638, :860).

- **builtins-arm64.cc** `Generate_JSEntryVariant`: `else if (entry_trampoline == Builtin::kJSEntryTrampoline || entry_trampoline == Builtin::kJSConstructEntryTrampoline) __ Mov(kExecutionStateRegister, x6);` — arm from x6 for **both** kCallable **and** kConstruct. (The v2 draft condition handled kJSEntryTrampoline but omitted kJSConstructEntryTrampoline — codex final #1.)
- x6 arrives through the native `GeneratedCode::Call` (typed function pointer invocation, execution.cc:409). Source inspection indicates no stack-pass fallback is needed; the x6 ABI change still needs runtime validation.

### 2.3 CEntry + direct-Wasm — Candidate 1 ⭐ codex resolved §2.3
CEntry (builtins-arm64.cc:4494) is reached by (a) JS-origin callers with correct x25, and (b) **direct-Wasm `blr`** whose x25 may have been clobbered — because **Liftoff (Wasm baseline) uses a SEPARATE cache-register list that still includes x25** (liftoff-assembler-defs.h:86) and allocates from it (liftoff-assembler.h:541). That is the concrete gap. Candidate 1 (codex's pick) closes it with two changes, avoiding candidates 2/3/4 (no single C++ stub before every CEntry — wrappers-inl.h:373/479; redundant transport since x25 is already callee-saved; locking the `blr` can't make the shared-mirror read safe).

- **Liftoff:** exclude x25 from the arm64 cache-register list whenever `V8_ENABLE_GREEN_WORKER_EXECUTION_STATE` is enabled, so Wasm bodies never allocate x25 as scratch and the `blr` into CEntry carries the correct x25 through the register.
- **CEntry prologue:** parameterize/split root init so green-mode CEntry **preserves incoming x25** and only initializes x26 — do NOT call the mirror-reading `InitializeExecutionStateRegister()` in green mode. Done before `EnterExitFrame` (builtins-arm64.cc:4531) → no runtime call, no entry-frame corruption.

This closes the green-body reader inventory. codex Phase-A review (findings C + D) confirmed: the JS-entry mirror fallback at builtins-arm64.cc:709 is unreachable from the three `Generate_JSEntryVariant` callers, CEntry (:4511) now preserves x25, and the only remaining green-mode mirror read — the C-API C-to-Wasm `kInitializeRootRegister` descriptor (wasm-compiler.cc:1214) — runs on a *separate ordinary isolate* (`Store::make`), so it **cannot run concurrently with a green body** and is left serialized (see the inventory table). This means the green-body parallel path has **no** shared-mirror reader, so §3 (mirror-write gate) can proceed.

---

## 3. Stage 2 — restrict the shared mirror write to the serial path (codex FINAL #3)

**Order matters (codex):** convert *every* green-mode reader to a per-worker source (§2.1–2.3) **first**, *then* gate `SetIsolateThreadLocals` mirror write (isolate.cc:5055/5076) on the **serial** path only (`if (!isolate->is_green_thread_mode())`), eliminating the concurrent non-atomic write-write C++ data race. Do **not** gate the write while any green reader still remains — those readers would load a stale/never-written slot.

---

## 4. Stage 3 — release `g_green_mutator_lock` around `PerformCheckpoint` (only after §2–3)

> **STATUS: LOCK-RELEASE REVERTED 2026-11-25 (fail-closed).** The attempt to
> release `g_green_mutator_lock` around `PerformCheckpoint` exposed a real
> FATAL in the parallel gate (`Check failed: node->IsInUse()` in
> `GlobalHandles::MakeWeak` via `node::RunTasks`), reproduced ~100% at
> `GT_NUM_WORKERS>=8`. Root cause: `PerformCheckpoint` runs each body's promise
> microtasks, which touch the ISOLATE-GLOBAL `GlobalHandles` table + shared
> promise/reaction bookkeeping that is **not worker-safe**. Reverted to the
> whole-iteration `lock_guard` (serialized async drive = correct). Stage 2
> mirror-gate + §5 atomic-counter + dead-scaffolding removal are **kept** (valid,
> green). True-parallel I/O drive needs the GlobalHandles/microtask path made
> worker-safe — a separate co-design (not achievable by releasing this lock).

**Preconditions that make the lock releasable (all met / in progress):**
- Stage 1 (Phase A): JS-entry x25 arming is per-worker arg-carried (`execution.cc` kRunMicrotasks from x2, kCallable/kConstruct from x6; CEntry preserves incoming x25) — the shared-`execution_state_base_` **mirror is no longer READ** on the green-body path. Confirmed readers are only the unreachable JS-entry fallback (`builtins-arm64.cc:709`) and ordinary-isolate CEntry.
- Stage 2 (this work): `SetIsolateThreadLocals` mirror **write** gated `!is_green_thread_mode()` (`isolate.cc:5062`) — the concurrent non-atomic write-write race is gone for the green isolate.
- Outer spawn/spawnAll driver already arms, *before* `DriveGreenAsyncBodiesParallel` runs, the allocation/GC/handle-scope machinery that the released-lock compute needs: `SetGreenHeapConcurrentAllocationEnabled(true)` (CAS disjoint young-gen bump pointers → no shared `bump_pointer` race), the GC-free `AlwaysAllocate` window (no concurrent GC relocates a mid-body object), `SetIsolateHandleScopePerWorker(green, true)` (per-worker handle routing), `SetConcurrentCompilationSuspended(true)` (no background compiler touches the heap). Disarmed only after the drive returns (`green_threads.cc:2378-2379`).
- `ParallelBodyDispatchActive()` is **false** when `DriveGreenAsyncBodiesParallel` runs (set true at 2338 for the sync pool run, set false at 2367 before 2374) → `GreenThreadCowSharedWriterScope` **still locks** → Map `construction_counter` / feedback vectors / JSDispatchTable stay serialized. §5 JS-counter atomic covers the counter.

**`src/green_threads.cc` `run_partition` (`DriveGreenAsyncBodiesParallel`, ~1510-1649) concretely:**
- The single `std::lock_guard<std::mutex> mutator_lock(g_green_mutator_lock)` at 1533 currently wraps the *entire* body iteration (gtRead resolve **and** PerformCheckpoint). Split it:
  - **Lock** only around the finished-`gtRead` Resolve (1539-1571) — resolves a promise in the body's *own* context; the context/handle/pending_reads machinery touched here is shared enough across the OS workers that this stays serialized.
  - **Unlock** (guard scope ends) → set `g_green_driving_body = d` (per-thread) → **`PerformCheckpoint(green)` (1629) runs with the lock RELEASED** — the microtask-execution parallel zone. `bctx2`/the body's microtask queue are per-body (round-robin partition → disjoint across workers), so concurrent `PerformCheckpoint` on different bodies is safe. `g_green_driving_body` restore at 1630 is per-thread too.
  - Next loop iteration re-acquires the lock for the following body's gtRead resolve.
- **Drop** the obsolete mirror-resync scaffolding inside the loop: `GTMirrorState ms = GTGreenThreadMirrorState(); GTGreenThreadRefreshMirror();` (1585-1591) + the `GT_GT_DEBUG` block (1592-1605) + the `GT_DEBUG_ENTCTX` block (1608-1628) — all exist solely to fight the pre-Phase-A mirror desync; readers are now per-worker so they are dead. Also remove the now-stale `GTMirrorState`/`GTMirrorState` struct + `GTMreenThreadRefreshMirror`/`GTGreenThreadMirrorState`/`GTGreenThreadDebugCtx` definitions once nothing else uses them (grep first).
- `iso_scope` (1514) + `handle_scope` (1515) stay **once per worker, outside the loop** — safe because under `is_green_thread_mode()` the Stage-2-gated mirror write is skipped, and `PerIsolateThreadData` wiring is per-thread. This is the inverse of the playbook's forced-GC fix: there the scope was taken *before* the lock mid-pool; here it is taken once per worker before the loop and writes no shared mirror.
- **Safety net during rollout:** keep the lock released *only* while `GT_NUM_WORKERS > 1` is effectively parallel; validate serial-green (`GT_NUM_WORKERS=1` path) unchanged first, then 4/8/12. If any §5 slot leaks a race, it surfaces as a SIGSEGV in the parallel gate — re-grant the lock for that path and isolate the slot.

**Why it's correct now (vs. the stale 1524-1532 comment):** that comment asserts the microtask runner re-arms x25 from the shared mirror — true *before* Phase A, false after (arming is arg-carried, §2.1). The lock was "the safe correctness baseline N workers run behind"; Stage 1+2 remove the very desync it guarded.

**Concrete change made (2026-11-25, REVERTED to serialized):** an initial attempt split the single `lock_guard` into a narrow `{ resolve_lock }` around only the finished-gtRead Resolve, releasing it around `PerformCheckpoint` (the microtask-execution parallel zone). The parallel gate then hit `Check failed: node->IsInUse()` ~100% at `GT_NUM_WORKERS>=8`: `PerformCheckpoint`'s promise-microtask execution touches the isolate-global `GlobalHandles` table (MakeWeak/Release of green result roots) + shared promise/reaction bookkeeping, which is not worker-safe across N concurrent `PerformCheckpoint`s. Reverted: the `lock_guard` now spans the whole per-body iteration (Resolve **and** `PerformCheckpoint`) → serialized async drive. The dead-scaffolding removal is kept (see below): `GTMirrorState`/`GTGreenThreadMirrorState()`/`GTGreenThreadRefreshMirror()` + the `GT_GT_DEBUG`/`GT_DEBUG_ENTCTX` blocks were removed from `run_partition`, and the corresponding symbols from `src/green_threads_gc.{h,cc}` (grep-verified no other users) — they were obsolete once arming became per-worker (Stage 1). That scaffolding was only ever exercised under the same `g_green_mutator_lock`, so removing it does not change serialization.

---

## 5. Stage 4 — isolate the shared state codex flagged (codex FINAL #5)

Each is a **data-path change, no entry runtime call**. Do before §4 ships, else released lock just moves failures to silent races. §5 fixes are currently *outcomes* — must be designed to concrete ownership/rooting rules before the lock release ships.

### §5 findings empirically pinned (2026-11-25, Orca read-only trace + source reads)

Per-slot ownership/rooting rules, decided — no longer outcomes:

| Item | Where | Verdict / mechanism (§5) | Fix (concrete) |
|------|-------|---------------------------|----------------|
| **JS execution counter** | `isolate.h:552` decl, `isolate.h:2048` `javascript_execution_counter_++` (`uint32_t`) | **RACE — guaranteed.** Every JS exec increments the shared `IsolateData` field; under released-lock compute every worker's body hits it. `VMState<JS>` (`vm-state-inl.h:51`) is in `ThreadLocalTop` → already worker-routed (not a target). | **Atomic increment — IMPLEMENTED.** `std::atomic_ref<uint32_t>(javascript_execution_counter_).fetch_add(1u, std::memory_order_relaxed)` (isolate.h:2047-2052). Minimal, no layout/snapshot change, compiles clean (862 TUs/0 errors at first build). **Serial-green gate: PASS** (concurrent-entry, p2-abi-proof, closure, async-context all exit 0 on the built binary). **[DONE]** |
| **Allocation bump_pointer / inline alloc** | `heap.cc` bump-pointer, per-heap `inline_allocation_pointers_` | **RACE — under released-lock pure compute.** With the mutator lock released, a worker whose body is in *pure compute* (no CoW-mutation view active, `ActiveCount()==0`) allocates through the **normal heap path** → shared bump-pointer write → race. The CoW barrier (`green-thread-cow.cc:917 `if (!ActiveCount()) …`) only gates when a CoW view is *active*. | **Route allocation through the CoW write barrier during the parallel compute zone** OR hold a per-heap allocation lock while the mutator lock is released. **Decision (implemented for Stage 3): rely on the CAS disjoint young-gen + GC-free window, NOT a per-body CoW-mutation scope.** Async bodies never hold a CoW view (CurrentGreenThread() is null → ActiveCount()==0 → barrier a no-op, by design), so a writer scope would either serialize everything (defeating parallelism) or add nothing. The shared bump_pointer race is prevented by SetGreenHeapConcurrentAllocationEnabled(true) (CAS disjoint young-gen bump pointers) + the GC-free AlwaysAllocate window, both armed by the spawn/spawnAll driver before DriveGreenAsyncBodiesParallel runs (green_threads.cc:2309, released at 2392). The acceptance async bodies allocate **disjoint** objects (async-context returns a fresh array literal per body; spawnall-n2 is pure compute) → disjoint Maps → no shared bump_pointer/Map-construction write. **[HIGH — implemented; parallel-gate validated]** |
| **Map slack-tracking `construction_counter`** | `map-inl.h:1120-1140` | **Guarded, but the guard is conditional.** `if (GreenThreadCow::ActiveCount() > 0)` takes `GreenThreadCowSharedWriterScope` (which **locks** — `green-thread-cow.h:229` `mutex_->Lock()`). BUT that scope is a **no-op when `ParallelBodyDispatchActive()`** (`green-thread-cow.h:220-223`, "bodies known to touch no shared-writer state"). Literal bodies of one shape share a cached initial Map; same-shape allocation across parallel workers races `construction_counter` **only if** the no-op is taken. | **Keep `ParallelBodyDispatchActive()==false` until disjoint-body verification exists.** The no-op is the optimization to add LATER (after proving a dispatched body's allocation is disjoint). For the acceptance workloads (same-shape pool objects) the lock MUST stay engaged → do **not** set `ParallelBodyDispatchActive()` when releasing the lock. Safe today because the scope already locks when the flag is off. [MEDIUM] |
| **Feedback vectors / JSDispatchTable tiering** | `code-` / feedback-vector writer paths | **Guarded under the same `GreenThreadCowSharedWriterScope`.** Same rule as Map slack: safe while `ParallelBodyDispatchActive()==false`. M2Probe-style warmup already compiles each body once on the embedding thread before workers run, so tiering is done pre-parallel. | Leave the optimizer OFF for now (don't set `ParallelBodyDispatchActive`). **[LOW]** |
| **`continuation_preserved_embedder_data`** | `isolate-data.h:422`, `microtask-queue.cc:166-194` (push/pop), Node `async_context_frame::set()` (async_context_frame.cc:45) | **BENIGN for green bodies.** The slot is only ever set by (1) the C-API pause API, (2) Node `async_context_frame::set()` (async_hooks), (3) structured-pause JS builtin. **None are on the green-body path** (`grep continuation_preserved_embedder_data src/green_threads.cc` = empty). Green bodies drive via plain `.then` reactions through `RunMicrotasks`, which pushes/saves/restores the slot — but it's *always* `undefined` there, so push/pop round-trips the same root. `heap.cc:7339 Heap::ClearKeptObjects` likewise only writes `undefined`. | No routing needed for current workloads. Route per-worker only if a body ever uses async_hooks/structured-pause concurrently (single-continuation => serial anyway). **[LOW — document, defer]** |
| **`weak_refs_keep_during_job` / finalizer list** | `heap.cc:7320-7340`, `runtime-weak-refs.cc:32` (`KeepDuringJob`) | **BENIGN for current workloads.** `ClearKeptObjects()` (heap.cc:7339-7340) only ever writes `undefined`; `KeepDuringJob` sets a non-undefined target **only** while a `FinalizationRegistry`/`WeakMap` job runs — the acceptance bodies never create those. Concurrent `ClearKeptObjects` all writing `undefined` is idempotent. | Same as continuation: route per-worker only if a body creates finalizers/weak-sets concurrently. **[LOW — document, defer]** |

**Leads (verified, lower confidence, defer until the parallel gate surfaces them):**
- `roots_table_` (isolate-data.h:424) — handle roots; green fork moved handle-scope/exec-state to per-OS-thread `PerIsolateThreadData` (isolate.h:648-693), but `roots_table_` itself is isolate-level. **Verify:** does body compute write shared `roots_table_`, or only per-thread handles? (Phase A armed x25 → per-worker block, so handle reads are per-thread; shared `roots_table_` write is the open question.)
- `allocation_sites_list()` / `native_contexts_list()` (heap.cc ~2810) — heap-global weak lists, touched on cache-miss/GC mainly.
- `is_finalization_registry_cleanup_task_posted_` (heap.h:2413) — bool from GC/cleanup-task without sync; cleanup task is non-nestable so can't overlap a released-lock drive, but double-check.
- `heap()->bump_pointer` — see the Allocation row; this is the dominant §5 gate for §4.

**Bottom line for §4 (lock release):** the mutator lock is releasable around `PerformCheckpoint` **once** (1) the JS execution counter is atomic, (2) each body's compute during the parallel zone is CoW-mutation-gated so shared bump-pointer/alloc/feedback writes stay serialized by `GreenThreadCowSharedWriterScope`, and (3) `ParallelBodyDispatchActive()` is left **false** (so that scope keeps locking). Continuation-slot + finalizer-slot routing are deferred (benign for the acceptance workloads).

---

## 6. Verification record (2026-11-25)

**Phase A (Stage 1 / per-worker x25 arming) — VALIDATED against the already-built Phase A binary (no rebuild needed; a wrong arm breaks serial too).**

| Suite | Env | Result | Meaning |
|-------|-----|--------|---------|
| test-green-threads-concurrent-entry | GT_SHARED_HEAP=1 | **PASS** | entry-arming discriminator (all JS/C/Wasm entry arms) |
| test-green-threads-p2-abi-proof | GT_SHARED_HEAP=1 | **PASS** | proves x6/kCallable + x2/kRunMicrotasks arg-carrying ABI |
| test-green-threads-closure | GT_SHARED_HEAP=1 | **PASS** | CEntry + Liftoff (Wasm) arming |
| test-green-threads-async-context | GT_SHARED_HEAP=1 | **PASS** | async body-drive path |
| test-green-threads-concurrency | GT_SHARED_HEAP=1 GT_NUM_WORKERS=8 | **PASS** | multi-worker serialized drive under lock contention |
| test-green-threads-shared-heap | GT_SHARED_HEAP=1 | **PASS** | shared-green-isolate allocation |
| test-green-threads-spawnall-n2 | GT_SHARED_HEAP=1 | **PASS** | n>=2 spawn |
| test-green-threads-parallel-execution | GT_SHARED_HEAP=1 | **PASS** | parallel drive |
| test-green-threads-cow | GT_SHARED_HEAP=1 | **FAIL @ scenario-5 (pre-existing)** | `scenario 5: copy address is an integer`, `test-green-threads-cow.js:200` — **identical to clean-HEAD baseline** (Orca baseline worker `ctx_c86722b5379c` proved the failure occurs on unstaged HEAD with no Phase A). NOT a Phase A regression; separate CoW-observability bug (`CurrentGreenThread()` null). Out of Phase A gate. |

**Phase A vs baseline attribution — forced-gc (`test-green-threads-forced-gc.js`):**
- FATALs 12/12 at M2Probe's *first line* `HandleScope scope(isolate)` → `v8::HandleScope::CreateHandle() Cannot create a handle without a HandleScope`, even with `GT_M2_FORCE_GC` unset and via the test's correct `node -e`/eval entry path.
- **Root cause is BASELINE, not Phase A.** The failing code is the embedding-isolate HandleScope routing (`vi::SetIsolateHandleScopePerWorker`, committed at `8436accc20d` — lines 2315/2391/3112/3197 in HEAD) with `green_threads.cc` line ~2897. Phase A's uncommitted diff touches **only** 4 V8 arming files (`execution.cc`, `builtins-arm64.cc`, `macro-assembler-arm64.{h,inl.h}`, `liftoff-assembler-defs.h`) — **not** `green_threads.cc` or any HandleScope code (`git diff HEAD --name-only` confirms). `8436accc20d` predates the forced-gc test (`a91f808`, committed *after* routing) → the routing that FATALs exists on baseline HEAD. Structurally Phase A cannot cause an embedding-isolate HandleScope-routing FATAL.
- Forced-gc is **outside** the Phase A validation gate (DoD = cow/shared-heap/closure/spawnall-n2/concurrency). Flagged as a pre-existing limitation of the per-worker HandleScope routing, not acted on here.

**Stage 2/3/4 (mirror gate, lock release, §5 isolate) remain pending — see §4 (lock release) and §5 above.**

## 7. Build (AGENTS.md discipline)

- Regenerate config **only** if a flag/`.gypi`/layout-constant changes. Stages 1–3 are source-only; the codegen change (stub signature + CEntry/wasm-prologue arming) forces an `mksnapshot` resnapshot, which `ninja` pulls into `node`. The Liftoff cache-register change is header-conditional (no snapshot impact beyond the flag being on).
  ```sh
  rm -f out/Release/build.ninja && python3 configure.py --ninja
  ninja -C out/Release -t compdb cxx > /tmp/ccdb.json
  grep -c V8_ENABLE_GREEN_WORKER_EXECUTION_STATE /tmp/ccdb.json   # many hits
  ```
- Build (cap 20, pipe not head): `ninja -C out/Release node -j20 2>&1 | tee /tmp/build.log`; confirm `LINK node`, no `ninja: build stopped`. Preflight (no build, no zombie `node`) before/after.

## 8. Acceptance gate (definition of done)

Run flag-**ON**. Any failure → do **not** release the lock for that path; isolate the shared state (§5) and re-loop. Diagnose child crashes in-process (AGENTS.md child-isolation trap: spawnSync signal-killed children are invisible — extract `CHILD_STRESS` standalone, run `out/Release/node /tmp/repro.js` directly; lldb batch with consecutive `-k` command pairs).

## 9. Risk register

- **x6 carrier (§2.2):** non-standard register for `GeneratedCode::Call`; must be proven by the crash gate (runtime validation, not source inspection). Fallback: stack-pass.
- **Liftoff exclusion (§2.3):** must not widen allocation in a way that starves Wasm bodies of general regs — validate Wasm perf at the gate.
- **Mirror-write ordering (§3):** convert readers *before* gating the write, or readers load a dead slot.
- **CoW gate (§5):** async drive holds `ActiveCount()==0`, so map-slack-tracking is currently **unprotected** — needs the parallel-writer guard.
- **mksnapshot:** codegen changes resnapshot; expect full `node` rebuild.

## 10. Committed-diff shape (when green)

`feat:` per-worker arming (§2) + mirror-write serialization (§3) + lock release (§4) + shared-state isolation (§5). `test:` committed regression + new parallel acceptance case. No `TODO`/dead scaffolding; Conventional Commits.

## 11. Final review verdict — GREEN (all 3 codex rounds incorporated)

Codex's §2.3 round closed the last design gap (Candidate 1); the Phase A diff review (A–E) confirmed the concrete arming code is correct and flag-off-clean **by source inspection**. Remaining gates before `feat:` — all move to runtime validation now that the design is locked:
- **Phase B (build + serial-green gate):** build V8 with Phase A in place; run `cow`/forced-GC/regression to prove per-worker arming didn't regress serial green (§2.2 x6 carrier & §2.3 Liftoff x25 exclusion must hold under real codegen — source review is not a substitute).
- §5 continuation-data + heap-wide WeakRef routing get concrete ownership/rooting rules (design depth, not just outcomes) before lock release (§4).


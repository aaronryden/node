# Green Threads → M2: Real Parallel on 4 OS Threads — Master Plan

**Branch:** `feature/green-threads` · **Worktree:** `/Users/aryden/dev/node-v26`
**Goal:** Green-thread bodies run in REAL PARALLEL across 4 OS threads, working
seamlessly with turbofan/maglev tiered compilation. No single-recursive-embedding-thread
"Model A".
**Chosen architecture:** SINGLE ISOLATE + CoW per-GT heap views + N real OS workers
(isolate-per-worker is rejected — it breaks shared JS objects between GTs).
**Consults:** Codex read-only consults (2026-11). Full plan below.

---

## 0. Ground truth (verified)

- Current scheduler is **Model A**: bodies run recursively on ONE embedding OS thread via
  `RunTasks → ScheduleLoop → RunGreenThread → entry()` (`green-thread-scheduler.cc`,
  `green-thread-execution.cc`). `Initialize(N)` only allocates N logical
  `GreenProcessor` runqueue slots — **no OS threads are created**.
- Each GT already has a real mprotected stack (mmap+mprotect, guard page), an ARM64
  register spill/restore buffer, and a per-GT CoW ref table (`HeapRefTable`).
- The code **already contains the intended M2 design** (several OS workers concurrently in
  one isolate) commented in `isolate.h:627-664` and `green-thread.cc:106-120`, gated **OFF
  by two switches**, both currently off:
  - Compile macros `V8_ENABLE_GREEN_WORKER_EXECUTION_STATE` / `V8_ENABLE_GREEN_THREADS` are
    **undefined** in `deps/v8`/`src`. Enabled via Node **GYP** config, not BUILD.gn alone.
    The real knob: `GYP_DEFINES='v8_enable_green_worker_execution_state=1'` (see
    `tools/v8_gypfiles/features.gypi`, ~line 334 green threads = 1, worker exec state = 0;
    `deps/v8/BUILD.gn:1069` emits the green-thread macro).
  - `green_thread_mode_` is set at green-isolate construction
    (`src/green_threads.cc:283` `SetGreenThreadMode(true)`) but the execution-state routing
    that depends on it is not completed.
- The throw-mode fix earlier made `thread_local_top()` return the **shared** top
  (`isolate.h`). That is **mutually exclusive** with per-worker routing — 4 OS workers
  sharing one `ThreadLocalTop` clobber each other's context/exception/handler/
  c_entry_fp/js_entry_sp. Completing parallelism requires **reverting** that shortcut back
  to per-worker routing and re-fixing throws under per-worker routing.
- **x25 arm already exists**: `InitializeExecutionStateRegister`
  (`codegen/arm64/macro-assembler-arm64-inl.h:1225`), called at JS entry
  (`builtins/arm64/builtins-arm64.cc:674`). The ARM64 save helper
  (`green-thread-asm.cc:22`) **omits x19–x28 (incl. x25)** → insufficient context-switch ABI.
- **A native worker experiment already exists**: `M2Probe` reates real `std::thread`
  workers (`src/green_threads.cc:1366`). **Reuse/repair it** before replacing the
  production scheduler.
- **A whole-body lock currently blocks parallelism**: `GreenJSEntry`
  (`src/green_threads.cc:806`) acquires `GreenThreadCowSharedWriterScope` before constructing
  scopes and retains it across `Function::Call`. Nothing can run in parallel until this is
  scoped down.
- **Allocation is already bump allocation with a concurrent CAS path**
  (`heap/main-allocator-inl.h:39`). "Allocation is the #1 bottleneck" must be
  **measured**, not assumed.
- **V8 contract**: one isolate = one mutator thread (`include/v8-isolate.h:283`). Honest
  ceiling inside one isolate: compute + writes to **private application data** are parallel;
  writes to **shared engine state** (allocator, feedback vectors, Maps, Code, GC) serialize
  — coordinated by `GreenThreadCowSharedWriterScope` (recursive mutex). This is expected and
  matches the contract; parallelism is real for compute-bound bodies.

---

## 1. The crash (blocked first, but re-diagnosed)

- Optimized (tiered-up) bodies crash with `EXC_BAD_ACCESS` at `0xbeeddead` in
  `Builtins_NotifyDeoptimized → CEntry_Return1`. `0xbeeddead` =
  `FrameDescription::kZapUint32` — a **temporary native deoptimizer allocation**, NOT proof
  that GC freed optimized code.
- Root cause (per Codex + debugging playbook): un-armed per-worker execution-state routing
  (x25) + the shared-top shortcut corrupting deopt state; `current_deoptimizer_` /
  `deoptimizer_lazy_throw_` are isolate-wide (`isolate.h:2699`) and must move to worker state.
- `--jitless` and baseline-only run clean; any optimized tier-up that deopts crashes.

---

## 2. Recommended implementation sequence (Codex ordering)

> Codex's key correction: **execution-state ABI → one real worker → complete safepoints &
> roots → four compute workers → worker-local allocation buffers → complete CoW semantics →
> optional GT-owned regions.** "Arenas should follow the first real four-worker compute
> proof, not precede it."

### PHASE 1 — Repair the execution-state ABI with ONE worker ⭐ START HERE
Allocation / GC / body execution stay **serialized** for this increment. Goal: prove
per-worker routing + throws + deopt work with a single OS thread. This is the gating item.
- [ ] **1.1 Enable the knob.** Build with
  `GYP_DEFINES='v8_enable_green_worker_execution_state=1'` (verify it propagates into
  generated compile commands before building). Cap the build `-j20` (30-core machine —
  always `-j20`, one build at a time, per node-v26 AGENTS.md build discipline).
- [ ] **1.2 Re-route `thread_local_top()`** (const + non-const, `isolate.h:1380`) to the
  same worker block used by `execution_state()`, keyed by **isolate + OS thread**
  (`FindOrAllocatePerThreadDataForThisThread`, `isolate.cc:557`). Ensure re-entry does not
  reset a live context/exception/handler chain. (Reverts the throw shared-top shortcut.)
- [ ] **1.3 Repair x25 arm.** `InitializeExecutionStateRegister` green-mode branch is chosen
  during codegen; snapshot-generated builtins must select the correct execution base at
  runtime. Preserve live entry args, LR, stack alignment, required registers. Audit every
  entry/re-entry/callback/continuation/restore path. Audit hand-written scratch-register
  lists (`register-arm64.h:531` reserves x25). Fix the ARM64 save helper to include
  x19–x28.
- [ ] **1.4 Deopt.** Output frames restore root/cage registers (`deoptimizer.cc:1712`);
  add a correct x25 value — the deoptimizer does not currently seed `kExecutionStateRegister`.
- [ ] **1.5 Move deoptimizer state off the shared isolate.** Move `current_deoptimizer_` /
  `deoptimizer_lazy_throw_` into worker-owned state (`isolate.h:2699`).
- [ ] **1.6 Throws.** Consistent addressing throughout the throw path: generated
  exception/handler fields, C++ `Throw`/unwind, `TryCatch`, pending-handler delivery all use
  the same worker top (`execution-state.h:110` classifies baked fields). After worker exec,
  capture the exception in rooted task-owned storage and deliver it on the caller thread
  (leaving an exception pending in a worker's top cannot propagate to the embedding thread).
- [ ] **1.7 First experiment / green check.** Build (Node GYP, `-j20`). Run with bounded
  deadlines:
  ```
  GT_SHARED_HEAP=1 out/Release/node --jitless /tmp/b.js
  GT_SHARED_HEAP=1 out/Release/node /tmp/b.js
  ```
  Add a one-worker `__m2_probe(1, body)` test with a self-contained fixed-iteration integer
  body, plus separate caught-throw and forced-deoptimization cases.
  Green: generated execution-field addresses == C++ worker-field addresses; worker block
  differs from shared top; x25 intact after deopt; **explicit evidence the optimized body
  actually reached its intended tier.** Re-verify throw probe exits 0.

### PHASE 2 — Complete worker-local execution infra (before removing body serialization)
`ExecutionState` currently holds only `ThreadLocalTop`, not all required state.
- [ ] Stack limits + interrupts (`StackGuard`, `stack-guard.h:203`): give workers separate
  limits and interrupt acks; route generated stack checks accordingly.
- [ ] Handle scopes + entered-context stacks (`handle_scope_data()`, `HandleScopeImplementer`
  are shared): make worker-local BEFORE removing `GreenJSEntry`'s whole-body lock. Locking
  only scope construction/destruction leaves overlapping shared scope lifetimes.
- [ ] Deoptimization/materialization state, stack walkers, persistent-handle mutation,
  per-worker marking barriers.
- [ ] Audit shared engine mutations (feedback, Maps, code install, caches, dispatch-table) —
  recursive mutex only protects paths that acquire it.
- [ ] Reuse `LocalHeap` registration/TLS/handles/marking-barrier infra (`heap/local-heap.cc:45`).
- Green check: two workers execute correctly under coarse serialization, including nested
  entry and exceptions; their tops/handles/contexts/limits/deoptimizer state remain distinct.

### PHASE 3 — Real all-mutator safepoint protocol (fix current defects)
Current stop API has bugs: `JoinCOWGCWorkers` waits 1ms and ignores drain
(`green-thread-cow.cc:394`); `RequestAllMutatorsStop` can return freed data (`:513`);
`WaitForAllMutatorsParked` returns immediately on unparked GT, caller proceeds anyway (`:577`);
forced-GC probe parks workers *before* their bodies (doesn't validate live optimized frames).
- [ ] Replace with an isolate-owned worker registry + collection epochs: gate admission &
  GT migration; request stop from every worker; reach generated safepoints even in loops
  with no explicit yield; publish precise roots + worker top + allocation state + marking
  buffers; acknowledge & park outside locks needed by coordinator; collect only after every
  ack + background-worker drain; resume by epoch, then reopen admission. Handle worker
  completion + allocation-triggered collection without waiting for a missing participant.
- [ ] Integrate with `IsolateSafepoint` (`heap/safepoint.cc:35`). Note
  `ShouldUseBackgroundThreads`/`RegisterCOWGCWorker` only address *background* work, not
  foreground allocation-triggered GC.
- Green check: force young/full GC while workers hold live objects in optimized frames, run
  tight loops, enter runtime calls, finish tasks; verify root publication + relocation;
  test timeout + worker-admission races explicitly.

### PHASE 4 — Make stack/root enumeration real
- [ ] `GreenThreadGCRootsProvider::Iterate` (`green-thread-cow.cc:228`) currently visits only
  CoW tables — not a complete worker-root provider.
- [ ] Use V8 frame iteration + tier-specific safepoint maps to enumerate each parked worker's
  live stack/register slots + handles + `ThreadLocalTop` roots.
  `Isolate::Iterate(ThreadLocalTop*)` (`isolate.cc:666`) is part of it.
- [ ] For the first pool, **pin a running body to its OS worker and run to completion**;
  implement true stack suspension separately. The ARM64 save helper omission of x19–x28
  means the current context switch is not a valid ABI.
- [ ] Do NOT substitute raw-word rewriting for precise movable-root handling.
- Green check: moving GC updates live references on every worker + suspended GT, including
  optimized frames.

### PHASE 5 — Prove four-core compute, then build the production pool
- [ ] Once worker state + safepoints work, **remove the whole-body lock**, retain sync around
  audited shared-engine ops.
- [ ] Use **fixed-work** integer loops with checked results (not `Date.now()`-duration loops —
  those measure elapsed time, not equal computation). Record: 4 distinct OS worker IDs + one
  isolate identity; an in-body rendezvous proving simultaneous execution; correct results for
  identical fixed workloads; sequential vs parallel wall time + process CPU.
- [ ] Adapt `GreenScheduler::Initialize` (`green-thread-scheduler.cc:48`) into a persistent
  pool with synchronized queues, explicit shutdown, task ownership. Preserve existing yield
  semantics until real suspend/resume replaces recursion.
- Green check: 4 fixed workloads overlap + show substantial multicore speedup under normal
  tiered execution. Repeat with OSR, deopt, exceptions; entry-only concurrency is insufficient.

### PHASE 6 — Parallel allocation: reuse LABs/TLABs (don't invent per-GT arenas)
- [ ] Intercept C++ + generated allocation: `Factory::AllocateRaw` (`factory.cc:302`),
  `Factory::allocator()`; `HeapAllocator::AllocateRaw` (`heap-allocator-inl.h:75`);
  `MainAllocator::AllocateFastUnaligned` (`main-allocator-inl.h:39`);
  CodeStubAssembler (`codegen/code-stub-assembler.cc:1663`);
  TurboFan `MemoryLowering::ReduceAllocateRaw` (`compiler/memory-lowering.cc:217`);
  Turboshaft `memory-optimization-reducer.h:250`; `runtime-internal.cc:447`; audit Maglev
  inline allocation / folding.
- [ ] Use **worker-local LABs/TLABs** (existing `MainAllocator`/`LinearAllocationArea`).
  Start with old-space LABs in V8-managed pages (postpones nursery changes, preserves heap
  metadata). Both generated top/limit accesses and C++ fallbacks must target the same bound
  allocation state.
- [ ] Keep ownership distinct: worker state owns/refs active LAB; GT state owns logical
  view/region identity; migration publishes old LAB + transfers exclusive binding at a
  safepoint; **OS thread IDs do not determine arena lifetime.**
- Green check: disjoint allocation addresses; correct refill/exhaustion; large-object
  fallback; arrays/double backing stores; GC during allocation across all tiers.

### PHASE 7 — Complete CoW access semantics
- [ ] Fix deeper gap: TurboFan write-barrier store clones the *stored value* then stores into
  original receiver (`compiler/backend/arm64/code-generator-arm64.cc:1260`);
  `COWWriteBarrierCheck` (`green-thread-cow.cc:996`) runs after store and skips Smis. These
  do NOT establish receiver isolation for `shared.x = 1`, unboxed doubles, or GC-barrier-
  eliminated stores.
- [ ] Implement: receiver resolution/cloning **before** mutation; per-view read resolution
  for aliases still pointing at originals; coverage for interpreter/baseline/Maglev/
  TurboFan/Turboshaft/builtins/C++ object access; identity for equality/Maps-Sets/cycles/
  prototypes/nested backing stores; typed-array/external-backing-store/atomics policies;
  GC-safe runtime cloning (allocating via compiler-inserted C call needs a valid stack-map/
  frame contract).
- Green check: two workers repeatedly read/mutate one baseline graph through multiple aliases,
  scalar/object/double stores; tier changes + moving GC preserve view isolation + identity.

### PHASE 8 — GT-owned regions (only after 6 & 7)
- [ ] A raw mmap slab is NOT a V8 heap space: objects need recognized pages/chunks,
  cage-compatible addresses, mark bits, remembered sets, sweeping, accounting, relocation.
  Hooks: `MarkCompactCollector` root traversal (`heap/mark-compact.cc:1927`);
  `Scavenger::EvacuateObject` (`heap/scavenger.cc:2271`); `ConcurrentMarking` local
  worklists (`heap/concurrent-marking.cc:394`).
- [ ] Add page/region ownership, cross-region remembered edges, evacuation destination
  ownership, per-mutator marking-barrier publication. Keep a **single global tracing
  collector** initially; independent region collection needs extra graph-boundary guarantees.
- [ ] Do NOT register every arena allocation as a strong root (retains all garbage). Ownership
  metadata and liveness roots must be **separate**. Completion does NOT generally permit
  O(1) reclamation (results/closures/promises/other-views/engine structures can retain).
- [ ] Warmup objects remain shared baseline allocations and clone lazily — do not retroactively
  assign them to the first GT.
- Green check: surviving results remain valid after GT completion; repeated completed
  workloads reclaim memory; cyclic escapes, weak refs, backing-store cleanup correct.

### Phase 9 — Acceptance
- Preserve serial tests; add parallel discriminators. Rewrite comments only when the
  corresponding behavior changes.
- Require normal tiered execution across the existing CoW + concurrency suites, the
  documented 40×40/60×60 regressions, forced mid-body young/full GC, repeated worker
  startup/shutdown, throws, OSR/deoptimization, and fixed-work four-core timing.
- Codex: "The first investment should be a correct one-worker x25/runtime/deopt ABI proof.
  The second should be complete worker roots and safepoints. Arenas should follow the first
  real four-worker compute proof, not precede it."

---

## 3. Build & safety discipline (node-v26 AGENTS.md — always)
- **Cap every build at `-j20`** (30-core box OOM-kills otherwise). Never `-j0` / `-j` / `-j$(nproc)`.
- **One build at a time.** `ps -eo pid,args | grep -iE 'ninja|/clang|cc1plus'` before starting.
- Capture output to a log via `tee` (never `head`/`grep` on the pipe → SIGPIPE kills ninja).
  Confirm linker ran: `grep -E 'error:|LINK node|ninja: build stopped' /tmp/build.log`.
- No zombie `node`: `ps -eo pid,args | grep -E '[n]ode'` before AND after any node run.
- Per incremental-change playbook: after a header-driven size/layout change, do a real
  `make -j20` (or ninja) rebuild of affected objects — never a manual relink alone — or stale
  `.o` files desync layout constants and corrupt `Isolate`/`IsolateGroup`.

## 4. Verified repros
- `/tmp/b.js` — single 1.2s burn body (`out/Release/node --jitless` clean; optimized tiers crash).
- `/tmp/probe_parallel.js` — 4 bodies × EACH ms, pure-CPU spin (measures parallelism).
- `/tmp/probe_throwmode.js` — throw-mode under `GT_SHARED_HEAP=1` (must exit 0).

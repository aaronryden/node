# Plan: four concurrent GT mutators in one V8 isolate and heap

**Branch:** feature/green-threads  
**Worktree:** /Users/aryden/dev/node-v26  
**Parent fix:** 1190abcbc68 (handle-lifetime fixes)  
**Status:** partial implementation — M2 allocation/COW primitives are built; the all-mutator STW-GC handshake (I3) is now implemented and accepted via the env-gated forced-GC probe; true-concurrency acceptance tests are still pending. M2 status inventory below.

## Goal

Four OS workers execute four green-thread (GT) JavaScript bodies **simultaneously
in one green V8 isolate and its one ordinary V8 heap**. A global run-lock,
v8::Locker around each body, or four separate isolates does not meet this goal.
This requires a V8 multi-mutator fork: the public V8 contract currently permits
only one thread to use an isolate at a time (deps/v8/include/v8-locker.h).

The first acceptance test must prove overlapping JS execution on four distinct
OS threads in that same isolate. Four thread IDs in a log only prove that a
pool exists. The existing serial path remains a reference until the parallel
path passes GC, CoW, error, and teardown tests.

## Ground truth and immediate corrections

| Area | Current source behavior | Implication |
| --- | --- | --- |
| Driver | RunTasks enqueues GTs and ScheduleLoop drives them on the embedding OS thread (src/green_threads.cc). | A persistent worker pool needs a per-batch completion barrier. |
| Yield | GreenScheduler::Yield requeues its current GT while its JS/C++ stack is live, then enters another GT recursively (green-thread-scheduler.cc:190-230). | A second worker must never dequeue that GT. Disable this yield in the first parallel milestone. |
| Current GT | g_current_green_thread_ is process-global because earlier TLS symbols aliased across force-linked archives (green-thread.cc). | Make attribution worker-local and test the *actual barrier reader* against the scheduler writer in the linked Node binary. |
| Processor | GetOrCreateProcessor picks processors_[0] for every OS thread. The implemented queue is FIFO; its declared capacity is not enforced. | Bind worker w explicitly to processor P_w and synchronize queue operations. |
| Scheduler identity | RunTasks gets a scheduler keyed by the embedding isolate; the yield callback gets one keyed by its current green isolate. | Store the owner scheduler on the GT or its batch. |
| Heap and GC | LocalHeap and IsolateSafepoint support background heap work, but not arbitrary simultaneous JS entry. | Extend their contracts to JS mutators, including every live JS stack and register root. |
| CoW | The per-GT table, dirty set, and global active-count/GC gate were built around one active JS mutator. | Prove concurrent attribution, publication, and GC root visibility for four active views. |

Do not assume the TLS issue disappears through shared-library linkage:
node_shared defaults to false in common.gypi and node.gyp.

## Required invariants

### I1. Worker-local execution state

Each worker owns a current GT, processor, JS entry stack, handle scopes,
exception/try-catch state, stack limit, and GC root record. Audit every mutable
field accessed through Isolate::thread_local_top() and by Execution::Invoke,
interpreter/JIT entry, runtime calls, and callbacks. Each field must be
worker-local, immutable after publication, atomic, or protected by a narrow
lock. A process-global current GT is forbidden in parallel mode.

The fork must establish thread-specific V8 state without a whole-isolate
Locker. Isolate::Scope alone does not make concurrent entry safe. Existing
LocalHeap support is a starting point, not evidence that JS mutators are safe.

### I2. Exactly one owner per GT

A GT is exactly one of queued, running on worker w, safely parked, or complete.
Claim and state transitions are synchronized. A GT with live JS/C++ frames is
never enqueued or stolen. First run bodies to completion. Later suspension must
publish every live root and either keep the GT pinned to its original worker or
prove migration of its V8 thread state.

### I3. All-mutator GC handshake

Start with **stop-the-world GC**. One coordinator requests a safepoint; all
four mutators stop, publish their registers, stacks, handles, active GTs, and
CoW tables; the coordinator collects; then workers resume. Each allocation
slow path and native callback that can block must participate. A parked worker
may claim to be quiescent only after its roots are visible to GC. No object may
move while another worker holds an unreported pointer in a register.

Audit young/old allocation, compaction, remembered sets, write barriers, weak
handles, and the CoW roots provider. Existing LocalHeap/IsolateSafepoint
machinery can be reused only where its root and safepoint contracts cover JS
frames. Concurrent GC is an optional later optimization, not a prerequisite
for parallel JS.

### I4. Shared mutation semantics

Use per-worker allocation buffers or synchronize allocation slow paths. A GT's
CoW ref table and dirty set have one mutating owner; GC reads them only after
the handshake. Publish a clone before it can be observed. Audit shared Maps,
feedback vectors, inline caches, tiering state, Code publication, global
handles, microtask queues, and other isolate-global metadata. For ordinary JS
writes to the same object from different GTs, specify CoW isolation, explicit
shared atomic storage, or a synchronization primitive. C++ data races are not
acceptable JS semantics.

### I5. Batch lifetime

Workers are persistent. RunTasks prepares rooted contexts, functions,
arguments, GTs, and CoW views before publishing the batch. While workers run,
the embedding thread cannot access the green isolate or free batch state.
Each worker publishes one result or error, marks its GT complete, and
decrements the batch outstanding count with release semantics. RunTasks waits
with acquire semantics, then copies results, tears down CoW, and frees the
batch. At exit, stop and join workers before disposing the green isolate.
Replace the mutable global GT registry with immutable per-batch call data or
a direct GT-to-data association.

## Worker example (design sketch, not drop-in code)

The worker has a queue phase, a concurrent V8 entry phase, and a completion
phase. There is **no global execution lock** around RunBodyToCompletion.

~~~cpp
void Worker::Run() {
  BindProcessor(id_);                       // Explicit P_id, not P_0.
  RegisterGreenMutator(green_isolate_, id_); // Per-worker V8 and GC state.
  for (;;) {
    GreenThread* gt = queue_.WaitAndClaim(id_);
    if (gt == nullptr) break;               // Shutdown after batches drain.

    DCHECK(gt->owner_worker() == id_);
    SetCurrentGreenThread(gt);              // Worker-local barrier source.
    EnterConcurrentGreenJS(green_isolate_); // Implements I1-I4.
    RunBodyToCompletion(gt);                // No recursive Yield in M1.
    LeaveConcurrentGreenJS(green_isolate_); // Publish result/error and roots.
    SetCurrentGreenThread(nullptr);
    gt->batch()->Complete(gt);              // Release and notify RunTasks.
  }
  UnregisterGreenMutator(green_isolate_, id_);
}
~~~

EnterConcurrentGreenJS represents substantial V8 work, not a wrapper around
Isolate::Scope. Keep GreenJSEntry's existing scope handling on the legacy
serial path; factor common body logic after the parallel entry protocol exists.

Example first batch: four pure JS computations on disjoint inputs, such as
fib(30), fib(31), fib(32), and fib(33). Prepare and root all inputs before
publishing the batch. All four workers call into the *same* green isolate.
Begin with disjoint output objects; add shared-object CoW cases after entry,
allocation, and GC roots are validated.

## Scheduler specifics

- First use one mutex-protected global FIFO and a condition variable. Four
  concurrent mutators are the hard part; work stealing is an optimization.
- Claim removes a queued GT and assigns worker ownership in one synchronized
  transition. Complete occurs once. The batch outstanding count includes
  queued and running GTs; an empty queue means park, not batch completion.
- A worker asleep outside V8 must park its heap-access registration first.
  Wake predicates cover new work, safepoints, and shutdown to prevent lost
  notifications.
- The current recursive Yield is prohibited in parallel mode. After a real
  suspension boundary exists, resume on the original worker first. Add
  cross-worker migration and per-P stealing only with a separate root/thread
  state proof.
- GT_NUM_WORKERS=1 exercises the new worker path with one worker. Keep a
  separate legacy serial mode for comparison. Validate and cap the count
  before creating workers.

## Milestones and verification

**M0 — inventory and baseline.** Run existing green-thread tests and record
results. Map mutable isolate state, root and allocation paths, GC safepoints,
Code/feedback writes, and CoW accesses reached by a GT body. Fix the
scheduler-identity mismatch. Test TLS through the real linked write barrier,
not only two worker-local sentinel reads.

**M1 — architecture probe: four concurrent bodies.** Implement worker-local
entry state, immutable batch publication, synchronized claim/completion, and
disjoint preallocated inputs. Temporarily disable GC *only in this bounded
probe*. Four JS bodies reach a native rendezvous callback with live JS frames
in the same isolate; all four must arrive before any returns. After release,
sample JS progress on each worker and show overlapping execution intervals.
Set a timeout so serial execution fails rather than hangs. Do not ship M1 as
a general mode.

**M2 — general allocation, GC, and CoW.** Implement all-mutator safepoints,
root scanning, allocation and write-barrier protocols, and per-GT CoW
ownership. Force young and full collections while four workers are active.
Test disjoint writes, shared reads, private clones, errors, repeated batches,
and teardown. General parallel bodies are enabled only after this gate.

### M2 status — implementation inventory (2026-11-20, read-only)

Ground truth on `feature/green-threads`. Production `spawnAll`/`spawn` run bodies **serially**
(Model A recursive scheduling: `RunTasks` -> `ScheduleLoop` -> `RunGreenThread`); true OS-thread
fan-out exists only in `__m2_probe`, where body execution is further serialized by `g_green_mutator_lock`.
Each M2 sub-part is tagged against the deliverables above and invariants I1-I5.

| # | M2 sub-part | Status | Evidence | Note |
|---|---|---|---|---|
| 2.1 | All-mutator STW safepoint + root scanning (I3) | **Implemented** | `GreenThreadCow::RequestAllMutatorsStop`/`PollForGCRequest`/`WaitForAllMutatorsParked`/`ResumeAllMutators` `green-thread-cow.cc:506,472,570,534`; wiring in `__m2_probe` `green_threads.cc` | Real STW handshake (no longer suppresses GC). Coordinator calls `RequestAllMutatorsStop` before spawning workers; each mutator's entry safepoint `PollForGCRequest` publishes its roots and parks until `WaitForAllMutatorsParked` + the collection complete; `ResumeAllMutators` resumes. I3 satisfied. |
| 2.2 | Allocation protocol (I4) | **Partial** | CAS bump `IncrementTop`/`TryIncrementTopAtomic` gated by `ConcurrentLinearAllocationGate` `linear-allocation-area.h:26,57,75`; callers `main-allocator-inl.h:42,49,67,77`; gate enabled only in `__m2_probe` `green_threads.cc:1377,1408` | Atomic-CAS is implemented and verified to engage, but wired **only** into the test probe and under the mutex serializer. No production concurrent-allocation path; alone it does not satisfy I4. |
| 2.3 | Write-barrier protocol (I4) | **Partial** | `COWWriteBarrierCheck` `green-thread-cow.cc:793`; `CowCloneForGeneratedCode` `:656`; attribution `CurrentGreenThread` `green-thread.cc:116` | Worker-local attribution (I1 satisfied via `PerIsolateThreadData`); structurally multi-writer-safe but never run concurrently - correctness inferred from the serial path. |
| 2.4 | Per-GT CoW ownership (I2/I4) | **Implemented (structural)** | `HeapRefTable` `heap-ref-table.h:37`, owner set once `:72`, recursive mutex `:106`; `ActiveCount` atomic `green-thread-cow.cc:41,448,467`; `Initialize/Teardown` `:426,467` | One mutating owner per GT via a recursive mutex. Correct, but only meaningful under the serializer. |
| 2.5 | Force young AND full GC with 4 workers | **Implemented (full)** | `M2ForceGC` `green_threads_gc.cc` invoked from `__m2_probe`; env `GT_M2_FORCE_GC=full|both`; `post_gc_consistent` gate | Forced full (mark-compact) collection runs while workers are parked at their safepoints. The young collection is intentionally not run separately: a non-forced young GC starts a concurrent-marking task that races the incremental marking that `GreenGCPause` stopped; the full mark-compact is `kAtomic` (no concurrent task) and evacuates the nursery too. `post_gc_consistent` verifies a rooted marker survives intact. |
| 2.6 | Test matrix (disjoint writes, shared reads, private clones, errors, repeated batches, teardown) | **Partial** | 6 suites `test/parallel/test-green-threads-*.js`; `__m2_probe`/`__concurrent_entry` `green_threads.cc:1291,1123` | Disjoint-write / private-clone / teardown scenarios exist, but the heavy suite is serial Model A; none force young+full GC with true concurrent workers. |

**Invariant cross-check.** I1 worker-local GT - **implemented** (`CurrentGreenThread` is per `PerIsolateThreadData`, `green-thread.cc:116`). I2 one owner per GT - **implemented structurally** (owner set once in `Initialize`). I3 all-mutator STW GC - **implemented** (STW handshake: coordinator `RequestAllMutatorsStop` + each mutator publishes roots at its entry safepoint `PollForGCRequest` before the collection; accepted by the env-gated forced-GC probe). I4 shared mutation - **partial** (CAS + serializer; shared cached-Maps, feedback vectors/ICs, tiering, Code publish are dodged by warmup + the serializer, not made safe). I5 batch lifetime - **partial** (persistent scheduler `GetScheduler` `green_threads.cc:823`; `__m2_probe` teardown + `ScheduleLoop` detach `green-thread-scheduler.cc:276`).

**Rigor gap.** The all-mutator STW handshake (2.1) and forced full-GC (2.5) now pass the env-gated probe (`test-green-threads-forced-gc.js`) because bodies are still serialized by serial Model A + `g_green_mutator_lock` — the STW handshake only needs quiescent roots, not concurrent-safe writes. The genuine open gap is I4 under **truly concurrent** bodies (not just 4 OS threads that happen to run serially): the CAS alone does not satisfy I4, since concurrent bodies also race on shared cached-Map mutation, `Code::flags_`, and feedback/IC paths. 2.2, 2.3, 2.6 remain untested there.

## M2 remaining work — concrete implementation plans

Ground truth re-derived from source on `feature/green-threads` (2026-11-20); every `file:line`
below was grep/read confirmed against the tree, not re-trusted from the plan. Each item gives the
goal, concrete numbered steps (file + line + plain-English logic change), the risk that a naive
approach hits, and the "done" signal. **Ordering and hard gates are at the end — read 1→5 in order;
Item 4 is gated on Items 1, 2, 3.**

### ITEM 1 — Real all-mutator stop-the-world GC handshake + root scanning (invariant I3)

**Goal.** Replace the *suppression* hack (`GreenThreadCOWSuspendedConcurrentGC`, `green-thread-cow.cc:271`,
forces `Heap::ShouldUseBackgroundThreads() == false` at `heap.cc:620`) with a real handshake so a GC can start
while 4 GT mutators are mid-body, stop all 4, enumerate every live root each holds (JS stack, live registers,
handle scopes, CoW view), collect, then resume. This is **I3**.

**Preconditions/deps.** Item 4 (true concurrency) depends on this, not vice-versa. Suppression is *correct*
precisely because GC and mutators never overlap today; the handshake is what makes overlap safe. So Item 1 is
logically first and upstream of Items 2 and 4, and largely independent of Item 3 (the root scan doesn't need
shared Maps/feedback being safe to *write*, only to be seen consistent).

**What already exists (reuse, don't rebuild).**
- `IsolateSafepoint`/`LocalHeap` (`heap/safepoint.h:30`, `local-heap.cc:343`) root **C++** handles on the embedding
  thread only — not a JS call stack. Do not reuse directly as the JS-stop mechanism.
- The one place GT registers ARE snapshotted: `GreenScheduler::Yield` `green-thread-scheduler.cc:199`
  (`green_thread_save_registers_arm64`), plus `HeapRefTable::RedirectWordsLocked` `heap-ref-table.cc:270-314`,
  which scans `gt->stack_limit()/stack_top()` and `gt->regs()` masked by `register_mask`. This is the canonical
  "GT holds references" snapshot — the handshake must reuse it.
- The root provider `GreenThreadGCRootsProvider::Iterate` `green-thread-cow.cc:196-206` is already registered while a
  view is active (`AddGlobalGCRootsProvider` `:241`); it roots `HeapRefTable` entries (original + copy via
  `HeapRefTable::VisitAsRoots` `heap-ref-table.cc:190`). It never sees the worker's JS stack — the handshake adds that.

**Steps.**
1. Add a per-GT publish hook hit at the yield + native-call boundary. In `GreenScheduler::Yield` (`:190`), **before**
   the native register save (`:199`), call a new hook `GreenThread::PublishGCRootsForCollector(gt)` that re-arms the
   native save and runs the existing `RedirectWordsLocked` machinery so the GT's stack + masked regs + `HeapRefTable`
   become the published roots. New V8 function `GreenThreadCow::PollForGCRequest(gt, iso)` (`green-thread-cow.cc`) reads a
   per-isolate atomic `gc_requested`; if set and `poll_eligible()` (not mid-publish, at a bytecode/native boundary) it
   publishes and parks the worker, returns `true`. Plain logic: `if (gc_requested && poll_eligible()) { publish(); park(); return; }`.
2. Install the poll at the single existing bytecode safepoint (the `CheckStackOverflow`/interrupt site that already
   triggers `GreenScheduler::Yield`) so a worker stops even if it never calls `Yield()`. Reuse that one site — do not
   invent a new bytecode probe.
3. Author the coordinator `GreenThreadCow::RequestAllMutatorsStop(GreenThread** out, int* n)` (`green-thread-cow.cc`):
   read `ActiveCount()` (atomic `green-thread-cow.cc:41`, bumped in `Initialize :448`, dumped in `Teardown :467`);
   release-store `gc_requested = true`; **spin-join** each entry of `g_cow_green_threads` (guarded by that mutex `:191`)
   until its "roots published + parked" flag is set; this quiescence *replaces* the `ShouldUseBackgroundThreads` false.
   Then call the collector — serial entry points only for M2, both valid at a world stop: `Heap::CollectAllGarbage`
   (`heap.cc:1363`, `heap.h:966`) for full, `Heap::CollectAllAvailableGarbage` (`heap.cc:1393`, `heap.h:974`) for young.
   Do **not** run the concurrent-scavenger/mark path here (that is M4).
4. Between stop and collect, let the collector run from the global root set — `GreenThreadGCRootsProvider` is already
   queued via `AddGlobalGCRootsProvider :241`, and the only requirement (active count > 0, guaranteed because we stop
   *while* views are active) holds. `RedirectWordsLocked` rewriting a parked GT's stack (`:240-247`) is safe because its
   worker is parked.
5. Resume: acquire-store `gc_requested = false`, notify each parked worker on its per-GT condition variable, let
   `ShouldUseBackgroundThreads` return to normal. Invariant: no object moves while a worker holds an unreported pointer
   (I3) — each worker published *before* parking, the collector moved only after all parked.
6. Guard against double-entry: a per-isolate `gc_in_progress` atomic so a second `RequestAllMutatorsStop` blocks and the
   bytecode poll (step 2) is gated to avoid recursing into the publish path.

**Why naive fails.** Flipping the gate off and running GC concurrently without the handshake makes scavenger/mark read
torn pointers from mid-body workers — exactly the `EXC_BAD_ACCESS in Scavenger::EvacuateObject` (`green-thread-cow.cc:259-264`,
`heap.cc:607-608`) that suppression exists to avoid. Reusing `LocalHeap`/`IsolateSafepoint` roots only C++ handles, not a
JS stack. Scanning a GT stack while its worker writes `stack[i]` is a torn read — the publish-must-precede-stop ordering
(steps 1/2 → 3) is the whole point.

**Done.** Item 2's test spawns 4 workers, forces young + full GC mid-run, and observes no SEGV/FATAL with every worker's
disjoint allocations and correct results; a new gtest asserting after `RequestAllMutatorsStop` every `gt` reports
`roots_published == true` with active count > 0.

### ITEM 2 — A test forcing young AND full GC with 4 active workers

**Goal.** Exercise the Item 1 handshake: inside a live 4-worker concurrent window on one green isolate, force a **young**
GC and a **full** GC and assert all workers still see disjoint allocations, correct results, no SEGV/FATAL. This is the
acceptance test for Item 1 (and M2 sub-part 2.5).

**Preconditions/deps.** Depends on Item 1 (else GC must stay suppressed and forcing it mid-window can't work). Depends on
Item 4 for *true* concurrency to be meaningful, but it can **first be validated under `g_green_mutator_lock`** (serial
mutators + forced GC) to prove the handshake, then re-run with the lock released once Item 4 lands — so Item 2 is a gate
for Item 4 and the first demonstration of Item 1. Depends on Item 3 only to the extent forced GC moving objects must not
tear shared writes.

**Entry points (grep-confirmed).** `GreenGCPause` (`green_threads_gc.cc:39-48`) only *pauses* marking (Stop/Start) — it does
not force a collection, and `__m2_probe` explicitly defers forced GC (`green_threads.cc:1288-1290`). Full-GC entry:
`Heap::CollectAllGarbage` (`heap.cc:1363`, `heap.h:966`, called via `cppgc-js/cpp-heap.cc:1307`). Young-GC entry:
`Heap::CollectAllAvailableGarbage` (`heap.cc:1393`, `heap.h:974`). From the embedding side,
`vi::Isolate* green_vi = reinterpret_cast<vi::Isolate*>(green); green_vi->heap()->CollectAll...`. `__m2_probe` binding at
`green_threads.cc:1571`; `__concurrent_entry` at `:1123`/`:1570`.

**Steps.**
1. Extend `M2Probe` (`green_threads.cc:1291`) with an env-gated forced-GC knob instead of a new binding: read
   `GT_M2_FORCE_GC` (default unset = current behavior; `"young"|"full"|"both"` chooses). Inject *inside* the CAS window —
   after warmup (`:1362-1365`), before the worker join (`:1405`) — gated so the existing suite path is byte-for-byte
   unchanged. (Note: M2Probe does **not** currently hold a `GreenGCPause`, so the injection runs GC normally — verify.)
2. Implement `inject_forced_gc(green, mode)`: `young|both → CollectAllAvailableGarbage(kMaxGarbageCollectionReason)`;
   `full|both → CollectAllGarbage(kNoFlags, kForcedGC, kGCCallbackFlagForced)`; a microtask flush between the two. Placement
   matters: under the Item 1 handshake the workers are parked, so GC walks only published roots; without Item 1 yet wired,
   workers are under `g_green_mutator_lock` (`:1395`) and the inject runs on the embedding thread — still a valid
   handshake-correctness test, just not yet of true concurrency.
3. Have each worker publish a disjoint-allocation marker before and after the GC (allocate an object, `g.addrOf`, touch it,
   GC window, touch again, re-read). Extend M2Probe's report (`ok_p[idx]` at `:1402`) with a per-worker `post_gc_consistent`
   bit comparing value before/after GC — makes the assertion structural.
4. Add `test/parallel/test-green-threads-forced-gc.js`, mirroring `test-green-threads-shared-heap.js:87` (`env: {GT_SHARED_HEAP:'1'}`)
   and `test-green-threads-concurrency.js:86-88` (env sweep). Pin `GT_SHARED_HEAP=1`, `GT_M2_FORCE_GC=both` on the child; the
   child calls `g.__m2_probe(4, …)` and returns `{results, all_ok, post_gc_consistent}`.
5. Add a second scenario forcing young-then-full in one window (relocate-then-compact) plus a concurrency sweep (1/2/4) to
   confirm the handshake scales with active mutators.

**Why naive fails.** Calling `CollectAllGarbage` while workers are genuinely running (not parked) makes mark walk roots a live
worker is updating → torn pointer → SEGV (this is why 2 depends on 1). Relying on the process printing `SEGV` is wrong — a
signal-killed child loses stdout/stderr (child-isolation trap in the plan); assert on **child exit status only**. Forcing GC
during warmup defeats warmup and races `JSFunction::UpdateCode` (Item 3); inject **after** warmup.

**Done.** `test-green-threads-forced-gc.js` green under `GT_SHARED_HEAP=1 GT_M2_FORCE_GC=both` with 4 workers: child exits 0,
`all_ok && post_gc_consistent`, no SEGV/FATAL — first green under the mutator lock (validates Item 1), then green with the
lock released (validates Item 4). Acceptance gate for Item 1.

### ITEM 3 — Shared Maps / feedback / ICs / tiering / Code publication concurrency-safe (invariant I4)

**Goal.** The allocation CAS (Item 4) only fixes young-gen bump-pointer overlap — it does nothing for shared-writer races on
V8 internal objects that 4 concurrent mutators mutate in place. Audit each shared-writer target and make it per-writer or
serialized, so I4 ("no C++ data races as JS semantics") holds under true concurrency.

**Evidence: the shared-writer targets (grep-confirmed) — these are NOT CoW-cloned.**
- **A. Object-literal map growth** `Map::InobjectSlackTrackingStep` (`objects/map-inl.h:1107`, called
  `objects/map.cc:2586` / `heap/factory.cc:3256`): plain non-atomic `set_construction_counter(counter-1)` on the shared
  initial Map (`:1113`); at `kSlackTrackingCounterEnd` it calls `MapUpdater::CompleteInobjectSlackTracking`
  (`objects/map-updater.cc:469`) which **transitions the shared map in place**. Race: 4 workers with the same literal shape
  all hit the same cached Map. Currently dodged by `g_green_mutator_lock` (`:1267/1395`) + warmup (`:1362-1365`).
- **B. Code publication / tiering** `JSFunction::UpdateCode → UpdateDispatchEntry` (`objects/js-function-inl.h:179-192`) →
  `SetCodeAndEntrypointNoWriteBarrier` (`sandbox/js-dispatch-table-inl.h:103-114`) does a read-modify-write on a shared
  per-isolate JSDispatchTable entry; `SetTieringRequest` (`sandbox/js-dispatch-table.h:223`) is another RMW. The atomic
  `entrypoint_`/`encoded_word_` are per-word atomic (`:118/144`) but the multi-field RMW is not. Shared by all functions in
  an SFI group — per-isolate-global, not per-GT.
- **C. Feedback vectors / ICs** `FeedbackNexus::SetFeedback` (`objects/feedback-vector-inl.h:530-542`) + `SetFeedbackPair`
  (`:475`): 4 mutators on the same shared function/array write the same feedback vector; `EnsureSize` growth is an
  allocation that can GC on the hot path. Not CoW-cloned.
- **D. CoW decisions prove A-D stay shared.** `IsCowCloneableRootType` (`heap-ref-table.cc:339-362`, only
  JS_OBJECT/JS_ARRAY) and `IsCowProvenanceHost`/`CowCloneForGeneratedCode` (`green-thread-cow.cc:765-782`, `:656-740`, skips
  Smi/HeapNumber/backing-store) never clone Maps/Code/feedback. The lock + warmup + GC-suppression are the *only* safety.

**Steps.**
1. **Map growth — per-writer clone.** When COW is active (`GreenThreadCow::ActiveCount() != 0`, already checked in the
   write barrier at `green-thread-cow.cc:658`) and a literal is created on the shared initial Map, clone the initial Map for
   this writer via the factory, allocate **old space** (`NewFixedDoubleArray(..., AllocationType::kOld)`, same rationale as
   `CloneCowBackingStore` `heap-ref-table.cc:456-457`), and install the clone as the literal's map *before*
   `InobjectSlackTrackingStep` runs on it — so `set_construction_counter` and the transition touch the per-writer clone, never
   the shared initial Map. Prefer this over a per-SFI recursive mutex (`map-updater.h`), which is the minimal-but-serializing
   fallback. (The CoW *write barrier* does not apply here — `InobjectSlackTrackingStep` isn't a user-object store.)
2. **Code/dispatch — serialize the RMW or disable concurrent tiering for the COW view.** Preferred for M2: when `ActiveCount()
   != 0`, have the compiler route publication through a per-GT dispatch entry or defer publication until the GT parks
   (check `ActiveCount()` in `UpdateCodeImpl`/`UpdateDispatchEntry`). Alternative: gate `SetCodeAndEntrypointPointer` +
   `SetTieringRequest` behind a per-`JSDispatchTable` recursive mutex (correct, but a global compiler gate — revisit for M4).
   Keep warmup as the fast path and add the `ActiveCount()` guard as the correctness path.
3. **Feedback/IC — gate growth.** Under COW serialize *all* `FeedbackNexus::SetFeedback` writes for shared vectors (reuse
   `CurrentGreenThread` attribution `green-thread.cc:116` to know which GT owns an IC slot). Cleaner later: per-GT feedback-cell
   indirection (out of scope for M2).
4. **Verification hook (not prod behavior):** a `GT_HEAP_DIAG`-gated assert that any object-literal/IC/code op under COW went
   through the per-writer path (steps 1/2/3) — flags accidental sharing for Items 2/5.
5. **Do not chase the whole compiler stack for M2.** Baseline/Maglev/Turboshaft Code-buffer writes are M3/M4; the warmup +
   dispatch gate (step 2) + map clone (step 1) cover the observable races (the plan's observed crash was
   `EXC_BAD_ACCESS in UpdateCode/Code::flags_`, addressed directly by step 2).

**Why naive fails.** "The CAS makes allocation safe, so we're done" is wrong — the CAS says nothing about 4 threads doing
`set_construction_counter` on the same Map (step 1) or an RMW on a shared dispatch entry (step 2); those crash regardless of
allocation. "Clone the Map via the CoW write barrier" is wrong — the barrier only fires on user-object stores and routes Maps
through a separate native path (`:746-755`), so a per-writer map (not the barrier) is the mechanism. "Make everything atomic"
is wrong — `entrypoint_`/`encoded_word_` are already atomic, yet the multi-field RMW isn't; serialization is the correct M2 tool.

**Done.** Item 2's forced-GC test plus Item 5's disjoint-write/concurrency scenarios pass with **4 unserialized workers** (lock
released — Item 4) and produce correct disjoint results, no SEGV/FATAL. Structural signal: the warmup (`:1362-1365`) becomes
optional — a probe that does **not** warm up still passes under concurrency.

### ITEM 4 — Wire the allocation CAS into a REAL concurrent path (GATED ON 1, 2, 3)

**Goal.** The atomic CAS bump-pointer (`IncrementTop`/`TryIncrementTopAtomic` gated by `ConcurrentLinearAllocationGate`,
`linear-allocation-area.h:26`, callers `main-allocator-inl.h:42,49,67,77`) is implemented and verified to engage — but wired
**only** into `__m2_probe` (`green_threads.cc:1377/1408`) and *behind* `g_green_mutator_lock` (`:1267`/`:1395`), so workers
never truly allocate in parallel. Enable the CAS in a genuine unserialized concurrent path so 4 workers reserve disjoint
bump pointers from one green heap simultaneously.

**Preconditions — GATED ON 1, 2, 3.** Item 1: without a real handshake, allocation-driven GC moves objects while a worker
holds the bump result in a register → torn pointer (the CAS fixes *overlap*, not a GC racing the CAS). Item 3: the CAS fixes
young-gen pointer overlap but not the shared-Map/dispatch/feedback races, so enabling true concurrency before Item 3 lands
reintroduces `EXC_BAD_ACCESS in UpdateCode/Code::flags_` + map-transition crashes. So **1 → 3 → 4** (Item 2 may validate the
CAS under the lock first; Item 4 is the step that releases the lock).

**Current wiring (grep-confirmed).** `SetGreenHeapConcurrentAllocationEnabled` stores the process-global flag
(`green-thread-cow.cc:406-423`, single `std::atomic<bool>` deliberately — the young-gen cppgc heap is distinct from the
CppHeap old space, `:413-420`); read at `linear-allocation-area.h:57`/`:75`; the ONLY callers are `green_threads.cc:1377`
(enable) and `:1408` (disable).

**The scheduler blocker.** `GreenScheduler::ScheduleLoop`/`Yield` (green-thread-scheduler.cc:237/190, recursive `entry()` at
`:227`) is single-OS-thread by design: `green-thread-scheduler.h:36-38` says the global runqueue is "accessed only from the
embedding thread, so no mutex is needed." Driving workers through it does not give fan-out. A genuine pool needs a
thread-safe claim queue.

**Steps.**
1. Enable the CAS **only** inside a `GT_SHARED_HEAP`-gated, truly-parallel worker pool — do **not** touch the production
   `spawnAll`/`RunTasks`/`ScheduleLoop` serial path (Model A) for M2. Guard on the existing `GT_SHARED_HEAP` (per
   `test-green-threads-shared-heap.js:87` / `spawnall-n2.js:62`) plus a new `GT_M2_PARALLEL=1`; call
   `SetGreenHeapConcurrentAllocationEnabled(green, true)` (`:1377`) before the fan-out and `false` (`:1408`) after join.
2. Release `g_green_mutator_lock` for the CAS-enabled window, **last**, and env-gated (`GT_M2_PARALLEL`): one-line removal or
   skip of the `lock_guard` at `:1395`. This single change turns "serialized CAS" into "real parallel CAS"; the env guard
   keeps the serializer as the correctness baseline for `GT_NUM_WORKERS=1`/legacy serial compare.
3. Provide a thread-safe worker queue so the fan-out is real: add a mutex + condition variable around `EnqueueGlobal`/
   `DequeueGlobal` (green-thread-scheduler.h:122-124, .cc:315/327) — or a small dedicated queue in `green_threads.cc` for the
   M2Probe pool. Header `:36-38` ("no mutex needed") is false for 4 OS workers; add the lock + `WaitAndClaim`. **Do not use**
   the recursive `Yield` `entry()` (`:222-227`, prohibited in parallel mode); each worker runs its claimed GT to completion.
4. Wire the Item 1 handshake into the pool so forced GC (Item 2) can stop all 4: the pool's stop path calls
   `RequestAllMutatorsStop` (Item 1 step 3) instead of relying on suppression; workers park via the Item 1 poll and the
   coordinator collects, then resumes.
5. Validate disjointness structurally, then re-introduce the lock to confirm it was doing the work: run Item 2/5 with
   `GT_M2_PARALLEL=1` (lock released) and with the lock held as the "control". If parallel is the only green one, CAS+Item 3
   do the work; if lock-held is green and parallel is not, Item 3 is incomplete. Keep both as variants.

**Why naive fails.** Just flipping the CAS on and releasing the lock, skipping 1 and 3, replaces one crash class with another
(GC racing the CAS; shared Maps/Code still racing). Driving workers through `ScheduleLoop`/`Yield` re-schedules on one thread
— not real fan-out. Keeping the lock and calling it "CAS wired in" is tautological (the serializer, not the CAS, gives
disjointness).

**Done.** `test-green-threads-shared-heap.js` (and Item 5's new scenarios) run with `GT_SHARED_HEAP=1 GT_M2_PARALLEL=1` and 4
**unserialized** workers: results correct and disjoint, no SEGV/FATAL, and the warmup (`:1362-1365`) is **not** required for the
pass (proving CAS + Item 3 handle real concurrency, not just suppressed compile). Item 2's forced-GC variant passing under
`GT_M2_PARALLEL=1` is the stronger signal.

### ITEM 5 — Expand the test matrix to true concurrency

**Goal.** The existing 6 green-thread suites are overwhelmingly the serial Model A path (plus one M1 entry probe). Lay out a
concurrency matrix: each scenario run under true unserialized concurrency with `GT_SHARED_HEAP=1` and the CAS wired in
(Item 4), plus the shared-read/private-clone/error/teardown coverage the serial suites mostly lack.

**Existing suites (grep-confirmed).**
- `test-green-threads-cow.js` — COW write-barrier regression. Serial Model A.
- `test-green-threads-shared-heap.js` — `GT_SHARED_HEAP` shared-isolate path, child process (`:87`). Serial.
- `test-green-threads-concurrency.js` — concurrent-GC/compiler-vs-CoW stress, env `STRESS_ROUNDS`/`STRESS_CONCURRENCY`/
  `STRESS_FIB` (`:86-88`) default 40×40×20. Serial Model A heavy loop.
- `test-green-threads-concurrent-entry.js` — M1 entry probe, `g.__concurrent_entry(n)` (`:52`); proves N OS threads Enter the
  same green isolate with live frames. Real concurrency but **no heap-touching JS** (entry only).
- `test-green-threads-spawnall-n2.js` — `g.spawnAll` with 2+ GTs under `GT_SHARED_HEAP=1` (`:62`); SIGSEGV regression. Serial.
- `test-green-threads-closure.js` — closure transfer; scenarios 4 + 5. Serial.

**Bindings for the matrix (in `green_threads.cc`):** `__concurrent_entry(n)` `:1123`/`:1570` → `{count, concurrent, results_ok}`;
`__m2_probe(n, fn0…)` `:1291`/`:1571` → `{results, all_ok}`; `spawn`/`spawnAll`/`yield`/`addrOf`/`cowCloneOf` `:1565-1569`.

**Steps.**
1. Split the matrix by concurrency level, all under `GT_SHARED_HEAP=1`; each scenario run two variants: `(a) serial` (existing
   Model A baseline) and `(b) true-concurrent` (`GT_M2_PARALLEL=1`, CAS wired, lock released — Items 2/4). Pin the env on the
   child like the existing suites.
2. Reuse `__m2_probe` as the harness for allocation/CoW scenarios and add the forced-GC dimension via Item 2's `GT_M2_FORCE_GC`:
   - **Disjoint writes** — 4 workers write their own objects/arrays; assert results distinct and correct.
   - **Shared reads** — 4 workers read a pre-rooted shared (Map/Set/immutable) object; assert all see the same value, no torn
     read. Scenario **none** of the current suites cover under concurrency.
   - **Private clones** — a worker writes into a shared user object, then reads its own CoW clone and asserts divergence
     (`cowCloneOf` `:1569` + `addrOf` `:1568` for pre/post addresses).
   - **Errors** — one worker's body throws; assert the other 3 still complete and the throwing worker reports `threw`
     (`ok_p` `:1402`); a per-worker exception must not corrupt siblings.
   - **Repeated batches** — many `__m2_probe(4,…)` in a loop (`STRESS_ROUNDS` style); assert all batches green.
   - **Teardown** — after workers finish, assert `ActiveCount() == 0` and the CoW roots provider is unregistered
     (`green-thread-cow.cc:470-472`); no leaked 0→1 registration.
3. Add the dedicated concurrent-forced-GC scenario (Item 2's `test-green-threads-forced-gc.js`) at young and full GC,
   concurrency 1/2/4.
4. Add one high-contention **same-shape** scenario to target Item 3 directly: N workers create many literals of the same shape
   (`Map::InobjectSlackTrackingStep` on the shared Map) and/or call a shared function many times (IC feedback + dispatch). This
   scenario **fails before Item 3 passes** and is currently why the lock/warmup exist (`green_threads.cc:1258-1266`); run it
   only under `GT_M2_PARALLEL=1`.
5. Keep the existing 6 suites as the serial baseline (no rewrites); add concurrency variants as **new** files so a parallel-only
   failure is attributable to concurrency, not a serial change.

**Why naive fails.** Flipping existing tests to `GT_SHARED_HEAP=1` is not concurrency — most already do, but they are serial
Model A (one OS thread scheduling the GTs), proving the shared-isolate path, *not* multi-mutator safety. The matrix must
distinguish "shared heap, serial schedule" from "shared heap, parallel schedule." Assert on **child exit status only** (a
signal-killed child is invisible). Run Item-3-targeted scenarios only after Item 3 lands.

**Done.** New `test-green-threads-forced-gc.js` (Item 2) plus a `test-green-threads-concurrent-matrix.js` (or extensions to
`test-green-threads-shared-heap.js`) covering disjoint writes, shared reads, private clones, errors, repeated batches, and
teardown — each green under `GT_SHARED_HEAP=1 GT_M2_PARALLEL=1` with the CAS wired (Item 4) — and the serial baseline 6 suites
still green. Decisive assertion: the **shared-read** and **same-shape-contention** scenarios (new, not in the existing 6) pass
under true concurrency, impossible until Items 1–4 are done.

### Ordering & hard gates

```
                 ┌──────────────┐
                 │ ITEM 1: STW  │◄── prerequisite for any real mid-run GC
                 │ handshake/I3 │    (Items 2, 4 depend on it)
                 └──────┬───────┘
                         │
                         ▼
                 ┌──────────────┐        ┌──────────────┐
                 │ ITEM 2: force  │        │ ITEM 3: shared │
                 │ young+full GC  │───────►│ Maps/Code/IC/  │
                 │ (acceptance    │ (gates  │ tiering/I4     │
                 └──────┬───────┘  CAS    │ (shared-writer │
                         │               │  safety)        │
                         ▼               └───────┬───────────┘
                 ┌───────────────────────────────┴───────┐
                 │            ITEM 4: wire CAS             │◄── GATED ON 1, 2, 3
                 │   true-concurrent allocation (release   │     (must precede 5)
                 │    g_green_mutator_lock)                │
                 └──────────────────────┬────────────────┘
                                         │  gates
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │            ITEM 5: concurrency test matrix     │
                 │  (all scenarios under GT_SHARED_HEAP=1          │
                 │   GT_M2_PARALLEL=1; disjoint/shared/clone/err/  │
                 │   batch/teardown + forced-GC)                  │
                 └──────────────────────────────────────────────┘
```

**Hard ordering.**
1. **Item 1 precedes Items 2 and 4.** Item 2 exists to exercise the Item 1 handshake; without a real STW handshake,
   forcing GC mid-window segfaults. Item 4 also needs it so allocation-driven GC cannot move objects a worker holds.
2. **Items 1 *and* 3 both precede Item 4.** The CAS alone fixes only young-gen bump overlap; it does not fix the shared
   Map/dispatch/feedback races (Item 3). Releasing `g_green_mutator_lock` (Item 4 step 2) before Item 3 lands
   reintroduces `EXC_BAD_ACCESS in UpdateCode/Code::flags_` + map-transition crashes. So **1 → 3 → 4**.
3. **Items 2 + 3 gate Item 5.** The shared-read and same-shape-contention scenarios (new, not in the existing 6) pass only
   under true concurrency, which requires Item 4, which requires Items 1 and 3.
4. **Item 2 is also the first green signal for Item 1.** Run it under `g_green_mutator_lock` first (validates the handshake
   with serial mutators), then re-run with the lock released once Item 4 lands (validates true concurrency).

**Suggested build/test cadence (per AGENTS.md):** one build at a time, `ninja -j20`, pipe to a log, confirm `LINK node`.
Iterate 1 → 2 (serial, lock held) → 3 → 4 (release lock) → 5, running `test-green-threads-*.js` after each and using
`__m2_probe`/`__concurrent_entry` probes + TSan for fast feedback before full builds.

**M3 — yielding and scheduling.** Replace recursive Yield with a safe
suspension/resumption mechanism. Test block/wakeup and worker-pinned resume.
Add per-P queues and stealing only for GTs proven safe to move. Benchmark
one versus four workers and compare against the legacy serial path.

**M4 — optional performance work.** Profile GC pauses, lock contention,
allocation, CoW cloning, and actual worker utilization. Consider concurrent
GC only after the stop-the-world design is correct.

Follow AGENTS.md for every build: preflight, one build at a time, -j20,
complete log, and linker confirmation. Use focused repros first. Run the
existing Node green-thread suite, overlap and forced-GC tests, stress loops,
and TSan where supported. TSan is evidence, not a replacement for root and
semantic tests.

## Stop conditions

Do not enable general four-way workloads if a worker can enter JS while a
mutable isolate field has no concurrency rule; GC cannot enumerate every
worker's roots; a running GT can appear in a queue; or CoW can attribute a
write to another worker's GT. A passing CPU benchmark cannot waive these
conditions.

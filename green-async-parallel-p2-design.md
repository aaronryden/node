# Phase 2 design — parallel async compute drive

> **Status: design + in-progress implementation.** This is the "separate co-design"
> from `green-async-parallel-plan.md`. P1 (dedicated libuv pump) is done and does not
> parallelize async *compute*; P2 does. Reviewed by Codex (gpt-6.1-sol) on
> 2026-11-26 (VERDICT: REWORK NEEDED) — findings folded in below as §2.1/§2.2 and the
> **Codex review** appendix. Implementation starts at **Tier 1** (finite-grained
> root-op guard + the read-handoff UAF fix); Tier 2 (V8-side per-worker runner state)
> is the longer co-design.

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

- **GlobalHandles table** — Reset/Release of the green result roots (the doc's
  earlier "MakeWeak / WeakPersistent" wording was INACCURATE: `settle_value_p` is a
  strong `Persistent<Value>` and its capture path calls `Reset(isolate, args[0])`
  (green_threads.cc:339,1732), not `MakeWeak`. `Reset` releases an existing cell and
  allocates a replacement in the isolate-global table. N workers doing Reset/Release
  concurrently can still alias/corrupt the shared GlobalHandles free-list (V8
  `GlobalHandles::Free()` asserts `node->IsInUse()` before reclaiming). The assert only
  establishes an *invalid release*, not its provenance (Codex: unproven attribution —
  could be the concurrent table, a stale handle, or a UAF; ASAN needed to confirm).
  See Codex review §1.
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
| Root the produced settle value | isolate-global **GlobalHandles** | **NO** — concurrent Reset/Release of strong roots corrupts the table (see §2; settle root is a *strong* `Persistent`, not a WeakPersistent) | → must be de-globalized per-worker or de-globalized body table |
| Queue any promise reactions / isolate tasks | isolate-global promise-task bookkeeping | **NO** | → must be per-worker or de-globalized |
| Compile / tier / feedback (map tiering, feedback vectors, JSDispatchTable) | process-global | **NO** — guarded by `GreenThreadCowSharedWriterScope` in GreenJSEntry (2340-2342) | stays serialized |

So P2's job narrows to **two** isolate-global hazards — the GlobalHandles table
and the promise/task bookkeeping. The compilation-state hazard is orthogonal and
stays serialized (it is the same wall sync bodies already hit).

## 2.1 Expanded V8 runner-state inventory (from Codex review §2)

`PerformCheckpoint`/`RunMicrotasks` touches far more than the three buckets in §3. This
table is the honest scope of what a body-reaction run can concurrently corrupt:

| State touched | Coverage status |
|---|---|
| Queue ring buffer / size / start / running / completion cb | Queue-local; single ownership helps **only if all enqueue paths target that owner** |
| Handle cursor, entered contexts, exec context/exception/entry | Per-worker routing exists — preserve it (do NOT over-claim isolation from it alone) |
| GlobalHandles freelist/blocks/counts/young-node | Privatizing only `settle_value_p` MISSES read-resolver roots, bridge roots, temp copy roots |
| `RootIndex::kCurrentMicrotask` (shared root every task stores; runner clears it) | **Uncovered** |
| `continuation_preserved_embedder_data_` | **Uncovered when enabled** |
| `weak_refs_keep_during_job` (heap-global; cleared by checkpoint) | **Uncovered** |
| Promise hooks / rejection callbacks / async-event delegate / debugger | **Unspecified** — reactions may call these; private queue does not isolate their effects |
| `javascript_execution_assert_` | Still isolate-level; cited but no concrete parallel-use protocol supplied |
| JS execution counter | Already atomic in this tree — OK |
| Compilation/tiering/feedback/maps/dispatch | Acknowledged but **no existing narrow compile guard in the async driver** to retain |
| Embedding-isolate state via bridges | **Uncovered** — workers enter the shared ordinary embedding isolate |
| Runtime-call stats | Conditional shared-state when enabled |

Consequence: the doc's earlier "ONLY two hazards" claim is unsupported. An exhaustive
symbol list for **arbitrary JS run by a reaction** cannot be bounded to the microtask
runner (user code can call further runtime ops / native bridges). The design now needs
an **explicit supported-operation contract** + an audit of reachable paths.

## 2.2 Independent pre-existing bug (FIXED 2026-11-26) — read-handoff UAF

Separate from parallelism, the driver published `rd->consumed=true` (old line 2064)
**before** re-accessing `rd->green_resolver_p` (old 2067). `GreenIoComplete` (libuv
pump thread) frees `w` the instant it observes `consumed=true` (green_threads.cc:1617),
so the pump could delete `rd` mid-access — a use-after-free. Fixed by **Reset()ing the
green resolver Persistent BEFORE publishing `consumed`**; once `consumed=true` the
driver touches `rd` no more (resolver `res` Local already rooted in the ctx handle
scope, so it survives the Reset). See Codex review §3 "Critical: read ownership is
published too early".

## 4. Route analysis

### Route 1 — replace `PerformCheckpoint(green)` with a manual body-queue drive
Drive **this body's own queue** (run reactions directly on the body's worker)
instead of calling V8's `PerformCheckpoint(green)`.

- **Correction (Codex):** `PerformCheckpoint` already drives the selected body
  queue — it calls `RunMicrotasks`, which runs `this` queue's reactions. So route 1
  does **not** newly avoid an isolate-global task queue; the distinction that matters
  is *queue-local storage* (body mq) *versus isolate-global execution machinery*
  (`RunMicrotasks` + the shared runner state). Replacing the call is not, by itself,
  a parallel win.
- **Critical omission (Codex §3):** an awaited continuation **is the remaining user
  body** — it allocates, compiles/tiers, calls native bridges, throws, and enqueues
  more reactions. `GreenJSEntry` wraps the *whole sync body* in a shared-writer scope
  but there is **no equivalent scope around async checkpoint execution**, and that
  scope can be a no-op when `ActiveCount()==0` (green-thread-cow.cc:218). So "drive
  only the queue" does not sidestep compilation-state serialization.
- **Verdict:** route 1 is the *mechanism* (per-worker queue drive) but is
  **necessary, not sufficient**.

### Route 2 — de-share GlobalHandles / microtask state per body
Give each body a private root table + private microtask/promise-reaction queue so the
continuation roots/queues into per-body state owned by the body's worker.

- **Understated difficulty (Codex §3):** a body table of ordinary `Persistent`s still
  *uses* the isolate GlobalHandles; a table of raw addresses needs **real root
  visitation + GC pointer-update integration** (heap.cc:4735/4788). Root discovery,
  relocation, strength, lifetime and teardown are all required. And releasing when the
  body settles is **too early** — `RunTasks` reads the captured result after the drive
  finishes (green_threads.cc:3033), so the value must stay strongly reachable through
  that transfer.
- **Uncovered enqueue routing (Codex §3):** promise-reaction routing selects the
  **handler's context**, then that context's native microtask queue — not the current
  worker's body queue. Cross-context handlers / thenables need an explicit ownership
  rule; `Context::Scope(body)` alone is insufficient (promise-abstract-operations.tq).
- **Bridge isolation gap (Codex §3):** `GreenBridgeMethodCallback` enters the shared
  ordinary embedding isolate and installs promise reactions on shared entry-stack state;
  multiple workers reaching the bridge need a separate serialized protocol.
- **Verdict:** route 2 is the *structural* fix but demands V8-side changes or an
  explicit supported-operation contract.

### Recommended (honest) — broader hybrid, not a Node-only seam

Per Codex §4: **neither route as specified is sufficient; a broader hybrid is the
smallest plausible approach.** Do not attempt to parallelize the whole body
re-execution (compilation-state is a separate wall). Instead:

1. **Retain** the existing per-body queues and checkpoint semantics (do not throw away
   the proven single-owner model).
2. **Prefer synchronized existing GlobalHandles operations** over inventing a private
   GC-root system — cover **ALL** reachable root ops (read-resolver roots, bridge
   roots, `CopyValueAcross` destination roots, settle value) — rather than only
   `settle_value_p`.
3. **Separately** address worker-local runner state (`kCurrentMicrotask`,
   `continuation_preserved_embedder_data_`, `weak_refs_keep_during_job`), checkpoint
   liveness, compilation/shared-writer guards, and the embedding-bridge entry protocol.
4. Constrain supported continuation operations to a vetted subset **OR** make the V8
   runner state per-body — whichever the runtime budget allows.

**This cannot honestly be scoped to lines 2007–2131** (Codex §4). If the change is
restricted entirely to the Node seam, **keeping the serialization is the only correct
option** the source review establishes — i.e. the parallelism must come from V8-side
per-worker isolation, not a Node-only patch.

## 5. The concrete seam (where the code would go)

All changes live inside the worker that already owns body `d` (line 2007-2131);
no new threads, no new partitioning. Two tiers, in order of difficulty:

### 5.1 Tier 1 — cover ALL reachable root ops with a fine-grained guard (Node-only,
###     incremental, does NOT remove the coarse lock's parallelism ceiling)
Per Codex §4: *prefer synchronized existing GlobalHandles operations over inventing a
private GC-root system.* So do **not** build `GreenHandleTable` yet (a table of
`Persistent`s still uses isolate GlobalHandles; a raw-address table needs real GC root
visitation + pointer updates). Instead:
- Serialize every **root** mutation the continuation drive performs — read-resolver
  roots (`rd->green_resolver_p` at 1672), bridge roots (587), `CopyValueAcross`
  destination roots (1251), and `settle_value_p` (339/3033) — under a single
  fine-grained lock scoped to just those operations (NOT the whole iteration).
- Keep `consumed` publishing AFTER the last root access (the UAF fix, §2.2).
- **Why:** this provably removes the `IsInUse()` corruption source. It re-serializes
  the root path, so it does not by itself yield wall-clock async speedup — it is the
  *safety* prerequisite for any parallelism work.

### 5.2 Tier 2 — V8-side per-worker runner state + per-body GC roots (the real win)
Requires V8 changes or a vetted supported-operation contract (Codex §4). The
hazards that must become per-body/per-worker, not Node-only:
- **Runner state**: `RootIndex::kCurrentMicrotask`,
  `continuation_preserved_embedder_data_`, `weak_refs_keep_during_job`
  (heap.cc:7319) — all heap/isolate-global during checkpoint; need per-worker or
  per-body variants.
- **Enqueue routing**: promise-reaction routing picks the *handler's context*, then
  that context's native microtask queue (promise-abstract-operations.tq:105), NOT the
  current worker's body queue — an explicit ownership rule is required.
- **Compilation**: no existing narrow compile guard in the async driver to reuse;
  must be added (or continuation ops restricted so bodies do no tiering).
- **Embedding bridge**: `GreenBridgeMethodCallback` (507) shares ordinary embedding
  entry-stack state (isolate.cc:6432); multiple workers reaching it need their own
  serialized entry protocol.
- **Root liveness**: released values must remain strongly reachable through the
  post-drive `RunTasks` copy (3033), not just "when the body settles".

### 5.2 Drive the body's own queue per worker (the Tier-2 *mechanism*)
- **Keep** the per-body queue drive, but note this is the **mechanism** of Tier 2, not
  a standalone seam: running a body's own reactions is exactly "running the remaining
  body", so it inherits **all** compilation/tiering/bridge hazards (Codex §3). It is
  therefore NOT a safe sub-region to parallelize in isolation — it is safe only once
  Tier 2's V8-side isolation (5.2 above) is in place.
- The driver reuses the per-worker scope recipe from GreenJSEntry (Isolate::Scope +
  HandleScope + Context::Scope, 2353-2356) so the resume stays in the body's own
  context; same-thread enqueue→run contract preserved (1891-1895).

### 5.3 What does NOT change (explicitly out of scope)
- Isolate activation (`v8::Isolate::Scope`, 2353) and JS-invoke permission — still
  isolate-global, stay serialized; not touched.
- Compilation-state guard (`GreenThreadCowSharedWriterScope`, 2345) — stays; note it
  is a **no-op** when `ActiveCount()==0` (green-thread-cow.cc:218), so the driver
  cannot rely on it.
- The always-allocate + CAS bump window (1899-1901) — stays; disjoint bodies
  already allocate disjoint fresh objects.
- Both tiers run on the SAME worker that owns `d`, so no new partitioning or thread
  hand-off.

## 6. Verification plan (per plan-doc Proof strategy)

- **Correctness gate FIRST (Tier 1 safety):** the `IsInUse()` FATAL is the
  acceptance signal for the fine-grained root guard. Re-run the GT=8 (and 16) full
  parallel gate with Tier 1's root-op guard active — it must stay green (currently
  it FATALs with the coarse lock removed). This proves the guard covers every
  reachable root op.
- **Parallelism gate (Tier 2):** wall-clock async speedup is ONLY valid once Tier 2
  V8-side isolation lands; Tier 1 alone does not remove the parallelism ceiling.
  The negative check (re-add the coarse root-guard → wall-clock regresses back)
  still proves the measurement sees real parallelism, not noise.
- **Wall-clock:** reuse the cooperative-io harness; parallel success = GT=4/8 beats
  GT=1 materially at fixed body count, plateauing at high worker counts. (Async
  compute is the metric P1 did not improve; P2 must.)
- **OS-parallel / timing-independent:** extend `__concurrent_entry(N)` from
  test-green-threads-parallel-execution.js — all workers must cross the barrier
  simultaneously; a serial drive deadlocks/times out.

## 7. Risk register
- **Root-op guard must cover EVERY reachable root** — read-resolver, bridge roots,
  `CopyValueAcross` destination roots, settle value — or a gap leaves a racing
  allocation. (Corrected from the old "MakeWeak weak-root" wording: the settle root
  is a *strong* `Persistent<Value>`; the hazard is concurrent Strong-Release, not
  MakeWeak.)
- **Reaction enqueue ordering** — a reaction enqueued during the per-body drive must
  run in the same worker's next pass, not leak to another worker's queue. Guarded by
  the single-owner invariant (5.2).
- **Compilation-state wall** — a continuation that compiles/tiers JS still serializes
  (compile guard is a no-op at `ActiveCount()==0`); P2 parallelism is bounded by how
  much work is root-routing + vetted JS vs tiering.
- **Enqueue routing gap** — promise-reaction routing targets the *handler's context*,
  not the worker's body queue; cross-context handlers/thenables strand unless the
  ownership rule is implemented (Tier 2).
- **Deadlock / lock ordering** — the fine-grained root guard must be acquired last,
  released first; never nested inside the scheduler react_mutex_ (see Codex §5).
- **Body monopolizing its partition** — a long continuation or self-replenishing
  microtask chain starves the worker's other bodies; `max_idle` does not bound
  checkpoint execution time.

## 8. Build order
P2 rides on P1 (P1 removed the pump as a lock-contention point and produced the
exact one-heap / one-uvloop shape). Build P2 top of the P1 commit. Do **not**
re-add the `HandleGCRequest` deferral (historical regression).

---

# Codex review (2026-11-26, gpt-6.1-sol) — full verdict & findings

> Read-only source review. VERDICT: **REWORK NEEDED** — the design identifies a real
> shared-state problem but its proposed seam was insufficient: it mis-identified some
> current behavior, omitted concrete races, and assumed compilation serialization that
> the async driver does not provide. Findings folded into this doc above; exact text:

1. **FATAL diagnosis:** plausible mechanism (concurrent GlobalHandles Allocate/Free),
   but **unproven attribution** — the assert only establishes an *invalid release*, not
   its origin. Two corrections: (a) the settle root is a **strong** `Persistent<Value>`
   (`Reset(isolate, args[0])`), NOT a `MakeWeak`; (b) an **independent use-after-free**
   in the read handoff (`consumed` published at :2064, then `rd->green_resolver_p`
   accessed at :2067, pump frees `w` on observing `consumed`). HandleScope-corruption
   and GC-relocation remain hypotheses, not demonstrated.

2. **Is the seam sufficient? No.** `PerformCheckpoint` already drives the body queue
   (calls `RunMicrotasks`); route 1 does not newly avoid isolate-global execution
   machinery. State inventory is much broader than two buckets — notably uncovered:
   `RootIndex::kCurrentMicrotask`, `continuation_preserved_embedder_data_`,
   `weak_refs_keep_during_job`, promise hooks/rejection/async-event/debugger,
   `javascript_execution_assert_`, runtime-call stats, and **embedding-isolate state
   reached through bridges**. "ONLY two hazards" claim was unsupported.

3. **Concrete omissions, by severity:**
   - *Critical — read ownership published too early* (the UAF above); also a no-retry
     leak path when the pump sees `consumed=false` (GreenIoComplete:1617).
   - *Critical — continuation execution IS arbitrary body execution*: there is NO
     shared-writer scope around async checkpoint execution (unlike GreenJSEntry's whole
     sync-body scope), and that scope can be a no-op at `ActiveCount()==0`.
   - *Critical — private C++ ownership is not GC rooting*: a `Persistent`s table still
     uses GlobalHandles; a raw-address table needs real root visitation + pointer
     updates; and "release when body settles" is too early — RunTasks reads the result
     after the drive (3033).
   - *High — other persistent ops remain*: read-resolver, bridge, and CopyValueAcross
     destination roots still race.
   - *High — worker ownership ≠ enqueue routing*: promise reactions route by handler
     context, not worker body queue.
   - *High — embedding bridge escapes isolation*: workers enter shared ordinary
     embedding entry-stack state.

4. **Smallest correct route:** a broader **hybrid** is the smallest plausible
   approach; neither route as specified is sufficient. "Prefer synchronized existing
   GlobalHandles operations over inventing a private GC-root system, while covering ALL
   reachable root operations… That requires V8-side changes or explicit restrictions on
   supported continuation operations. It cannot honestly be scoped to lines 2007–2131."
   If restricted entirely to the Node seam, **keeping serialization is the only correct
   option** established by the source review.

5. **Deadlock/fairness:** preserve drain-to-empty checkpoint semantics; a body can
   monopolize its partition; parking readiness must include every new wake source;
   do not hold new root/compiler locks across arbitrary callbacks; preserve the
   scheduler's mutex-protected block/reactivate/drain + owner-specific readiness
   predicate.

6. **Agree (no change):** one owner per body + queue; resolve/queue-exec on the owner;
   dedicated libuv pump separate from JS execution; per-worker handle/execution-state
   routing; timing-independent concurrency discriminator; avoid the historical
   GC-request deferral. Central conclusion confirmed: **removing the coarse mutex
   without replacing its shared-state protection is unsafe.**

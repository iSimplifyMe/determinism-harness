# Study-3 follow-up companions: reload-churn A/B + logprob margins

**Status: exploratory. This plan is committed BEFORE any companion data is
collected** — the same ordering discipline as the frozen preregistrations,
applied at addendum scale. Nothing here amends PREREGISTRATION-v3 (frozen,
tag `prereg-v3`), reopens a registered answer, or joins the confirmatory
dataset. Both companions produce their own labeled runs, reports, and
sections; the frozen study-3 results stand as published.

Engine pinned as in study 3: Ollama 0.30.5 both boxes, weights digests
recorded per run. Single-flight execution. Both companions run in
owner-approved windows under the recorded 2026-07-29 posture ("low-use box;
loading is fine"), box state snapshotted before and after.

---

## Companion A — reload-churn A/B (the section-6 disclosed tension)

### Background

The Metal pilot's fully-shuffled schedule produced the only sub-ceiling
greedy cell anywhere in the study-3 pilots: `gpt-oss-20b|open_generation|
greedy|effort_low` at modal share 0.60 with 3 variants (n=10). The blocked
confirmatory run measured the same cell at 1.000 (n=100, all 100 responses
byte-identical, 811 output tokens every call, warm loads ~0.26 s
throughout). PREREGISTRATION-v3 section 6 and the published paper disclose
this as an open tension: schedule/swap churn as a possible perturbation
channel.

The shuffled pilot differed from the blocked confirmatory in (at least)
two mechanically distinct ways: (a) the model was unloaded and reloaded
between its calls, and (b) other models were loaded concurrently (memory
pressure on an over-budget box). This companion isolates (a) — the
cleanest single-variable manipulation. A null here does NOT close the
tension; it narrows it to (b) or pilot-n noise, and says so.

### Hypothesis (directional, stated pre-data)

Unloading and reloading the model between calls reduces byte-level
reproducibility of long greedy generation on the Metal runtime, relative
to keeping the model resident.

### Design

- **Cell:** `gpt-oss-20b|open_generation|greedy|effort_low` — the exact
  frozen cell (same canonical body bytes: greedy options, seed 42,
  num_predict 4096, keep_alive 10m).
- **Arms:** `blocked` (model stays resident; replicates the confirmatory
  condition) vs `churn` (the model is unloaded via an out-of-band
  side-call, confirmed absent from `/api/ps`, before EVERY measured call).
- **The measured request bytes are byte-identical across arms.** The
  manipulation lives entirely outside the measured call (an unload
  request targeting gpt-oss:20b only, never any production resident),
  so the negative control holds across arms, not just within them.
- **n = 100 per arm per box**, scheduled as alternating mini-blocks of 10
  (B,C,B,C,…) for time balance. One warmup call (meta control=warmup,
  excluded from analysis) heads the schedule. A churn mini-block's last
  call leaves the model loaded, so every blocked mini-block starts warm;
  per-record load data verifies this rather than assuming it.
- **Boxes:** Metal primary (where the tension was observed); CUDA mirror
  (pilot #2 measured this cell at 1.0 there — the mirror discriminates
  Metal-runtime-specific from engine-general).
- **Fixed deterministic schedule** (no shuffle: the schedule IS the
  manipulation), recorded via schedule_sha256 and `schedule_fixed` in the
  manifest.

### Manipulation gate (arm void if failed)

- Every churn-arm record must carry `pre_unload_confirmed: true` (model
  observed absent from `/api/ps` after the unload side-call, before the
  measured call) AND a reload: `load_duration_ns` > 3x the blocked arm's
  median.
- Blocked-arm records (post-warmup) must show warm loads: median
  `load_duration_ns` within the confirmatory run's warm range (~0.26 s).

**Pre-data amendment (2026-07-30, before any companion data):** the cold
criterion was drafted as 10x the blocked median. The n=1 launch smokes
(scratch, not analysis data) measured reload load_duration at 5.42 s vs
0.39 s warm on CUDA (13.9x) but only 1.62 s vs 0.26 s warm on Metal
(6.2x): macOS keeps the unloaded weights in the filesystem page cache, so
a confirmed unload/reload is disk-warm and a 10x load-time bar would void
a genuinely manipulated arm. The state the hypothesis targets — a fresh
model instantiation and memory layout — is reset by the confirmed unload
regardless of page-cache warmth, so the observed-absence check remains
primary and the load criterion is corroboration. Amended to 3x before any
confirmatory companion call; both smoke reloads clear it with margin.

**Smoke context (disclosed, n=1, scratch):** on BOTH boxes the smoke's
byte-identical frozen request produced different bytes than that box's
100/100 confirmatory modal (new server session on CUDA; ~20 h later on
Metal) — cross-session drift that `matches_confirmatory_modal` will
document at n=100. The arms are interleaved within one session, so the
A/B contrast is internally valid either way.

### Endpoints

- **Primary:** churn-minus-blocked modal-share difference (Wald, CI95),
  per box.
- **Secondary (descriptive):** distinct-variant counts per arm; whether
  each arm's modal bytes equal the confirmatory run's modal sha256 for
  the cell (drift cross-check rides the digest + engine pins);
  first-divergence character indices; per-arm decode rates.

### Interpretation (stated pre-data)

- Metal CI95 excluding 0 downward → reload churn is a real perturbation
  channel for long greedy generation, explaining at least part of the
  pilot tension. CUDA then splits runtime-specific vs engine-general.
- CI spanning 0 on both boxes → reload alone does not reproduce the
  pilot instability; remaining candidates are multi-model memory
  pressure and pilot small-n, named as such (a pressure-arm follow-up
  would require a dedicated design; none is promised here).

---

## Companion B — logprob margins (exploratory, per the freeze gate)

### Background

PREREGISTRATION-v3 exiled logprob capture from the frozen bodies: the
freeze-gate probe showed the `logprobs` request fields are NOT
generation-neutral at length (3/3 distinct outputs with logprobs on the
CUDA open-generation greedy cell vs 20/20 identical without; short-output
cells neutral 3/3 each — `evidence/logprobs-bytes-probe-cuda-4090.json`).
Margins were registered as exploratory companion runs only. These are
those runs.

### Question (descriptive; no hypothesis test)

How near are the near-ties? The study-3 headline cells that moved —
120b structured-JSON effort_high (0.890, an 89:11 two-variant split) —
and the cells that didn't (20b structured-JSON greedy at or near 1.0)
should differ in the margin between the top-1 and top-2 token
candidates along the generated path, if argmax near-ties are the
mechanism behind the coin-flip.

### Battery

| Box | Cell (all greedy) | n | Note |
|---|---|---|---|
| metal | 20b structured_json effort_low | 50 | stable baseline |
| metal | vl-32b open_generation think_off | 20 | least-stable greedy model (0.780); observer-affected at length — descriptive only |
| metal | 120b structured_json effort_low | 50 | stable comparator, dedicated eviction block |
| metal | 120b structured_json effort_high | 50 | THE 89:11 cell, dedicated eviction block |
| cuda | 20b structured_json effort_low | 50 | the 0.990 one-flip cell |
| cuda | 20b open_generation effort_low | 20 | observer-affected at length — descriptive only |

- Request fields: `{"logprobs": true, "top_logprobs": 5}` — the
  probe-verified working spelling on 0.30.5, both boxes.
- Cell ids are suffixed `|logprobs` so no companion cell can ever
  collide with a frozen cell key.
- **Caveat carried on every output:** short-output cells were probed
  byte-neutral, so their margins plausibly describe the frozen
  trajectories; open-generation margins describe logprobs-perturbed
  trajectories, not the frozen ones. No margin from these runs is ever
  attached to a frozen-run record.
- 120b cells run LAST on Metal in their own per-model block (65 GB load
  evicts production residents — dedicated-window treatment, teardown
  re-pulses residents to 24h keep-alive and verifies box state).
- Records store compact per-token margin triples
  `[chosen_logprob, top1_logprob, top2_logprob]` (full raw top-5 arrays
  are not persisted; the triple is the analysis object and keeps records
  bounded).

### Endpoints (descriptive)

- Per-call minimum top1-minus-top2 margin and its token position.
- Fraction of token positions with margin below 0.001 / 0.01 / 0.1.
- For multi-variant cells: the margin at the fork position (the token
  where variants first diverge), within the companion run's own records.
- Stable-cell vs unstable-cell margin-distribution comparison
  (120b effort_low vs effort_high is the matched pair).

---

## Results (post-data, 2026-07-30 — reports are authoritative)

Executed same day: 402 A/B calls + 244 margins calls across both boxes,
zero failures, zero retries, all four runs complete
(`runs/local-study3-churn-ab-20260730T2102*`,
`runs/local-study3-margins-20260730T21*`).

- **Companion A registered primary: churn-minus-blocked +0.1000
  [+0.0412, +0.1588] on BOTH boxes — the direction OPPOSITE the stated
  hypothesis**, with the manipulation and cross-arm gates passing. The
  churn (fresh-reload) arm was byte-perfect, 100/100, on each box.
- The exploratory cache-state decomposition explains it with recorded
  evidence: the schedule bundled three prompt-cache states, and each is
  internally deterministic — fresh-load/no-cache (prefill ~199 ms CUDA /
  ~118 ms Metal): one variant, 100/100; cached steady state (~17 / ~29
  ms): one variant, 90/90; warmup-era first blocked block (~37 / ~106
  ms): one variant, 10/10. Every point of sub-ceiling modal share in the
  blocked arm is the state boundary, not noise. Read back, the section-6
  tension resolves the same way: the shuffled pilot mixed cache/load
  states (0.60, three variants, each plausibly a state), and the blocked
  confirmatory run's interleaved tasks kept the prefix cache overwritten
  — one state, 1.000. **Reload does not degrade determinism; state
  mixture masquerades as nondeterminism.** The launch smokes' cross-
  session drift reads the same way.
- **Companion B margins:** the cells with observed instability are the
  cells with razor margins — minimum top1-minus-top2 margin 0.0034
  (120b structured-JSON effort_high; 2 variants at n=50) and 0.0014
  (vl-32b open generation; 2 variants at n=20) versus ≥ 0.011 on every
  stable cell, with the 120b matched pair at 0.0034 (high) vs 0.846
  (low) — a ~250x gap between the unstable arm and its stable
  comparator. Observed forks occur at margins 0.005–0.019.

Full numbers: `reports/churn-ab-report-20260730T220525Z.*` and
`reports/margins-report-20260730T220540Z.*`.

## Companion C — cache-state A/B (pre-data plan, added 2026-07-30 evening)

**Status: exploratory-directional; this plan is committed BEFORE any
companion-C data.** Companion A's registered result came out opposite its
hypothesis, and the exploratory cache-state decomposition attributed the
inversion to prompt-cache coverage. That attribution deserves its own
manipulated test rather than remaining a post-hoc reading.

### Hypothesis (directional, stated pre-data)

With the model held loaded (no unloads anywhere), the prompt-prefix cache
state at call time determines the generated bytes: calls whose prefill is
cache-cold (immediately preceded by a different-prompt call) are
byte-identical to each other, calls whose prefill is cache-warm
(immediately preceded by an identical-prompt call) are byte-identical to
each other, and the two groups differ from each other.

### Design

- **Cell:** the same frozen cell (`gpt-oss-20b|open_generation|greedy|
  effort_low`, byte-identical measured bodies in both arms).
- **Arms (n=50 each):** `warm` = measured call immediately preceded by an
  identical measured call · `cold` = measured call immediately preceded
  by a flusher call carrying a DIFFERENT frozen prompt (the
  classification task's, short), which evicts the open-generation prefix
  from the single-slot context cache.
- **Schedule:** one warmup (load), then alternating mini-blocks
  C,W,C,W,… of 10: a cold block's measured calls each follow a flusher;
  a warm block's calls each follow a measured open-generation call (the
  preceding cold block ends with one, so warm blocks never start cold).
  Flushers carry meta `control=flusher` and are excluded from endpoints
  like warmups. Fixed schedule, single-flight.
- **Box:** CUDA only tonight (non-production box); Metal replication
  optional in a future window.
- **No unloads anywhere** — companion A already measured the load factor;
  this design isolates prefill-cache state with residency constant.

### Manipulation gate (arm void if failed)

Prefill time is the recorded discriminator (~10x separation observed in
companion A): every cold-arm measured call's `prompt_eval_duration_ns`
must exceed 3x the warm arm's median, and the warm arm's median must sit
below the cold arm's median/3.

### Endpoints (registered)

1. Within-arm byte determinism: modal share per arm (prediction: 1.0 in
   both arms).
2. Cross-arm modal equality (prediction: the arms' modal outputs DIFFER).

### Descriptive only (NOT registered — cross-session drift is already on
record, so sha-matching to history is reported, not predicted)

Which historical variants each arm's modal matches, against the four
distinct CUDA variants already committed for this cell: confirmatory
modal `13bae41c…` (previous server session) · companion-A B1 `20310cdd…`
· companion-A cached steady state `cf2c66c8…` · companion-A fresh-load
`45e27daf…`. Companion C runs in a NEW server session, so any match or
non-match is itself informative about session-scoped state.

### Interpretation (stated pre-data)

Both predictions holding = the cache-state attribution is confirmed as a
manipulated result on this box: outputs are deterministic conditional on
prefill-cache state, and state mixture fully accounts for companion A's
inversion. Prediction 1 holding with prediction 2 failing = cache state
does not matter in a fresh session (weakens the attribution). Prediction
1 failing = the state story is incomplete; report as such.

### Companion C results (post-data, 2026-07-30 late evening)

151/151 calls, zero failures
(`runs/local-study3-cache-ab-20260730T222842Z.*`,
`reports/cache-ab-report-20260730T223839Z.*`).

- **Manipulation gate FAILED — endpoint 2 honestly voided.** Warm-arm
  prefill median 35.7 ms vs cold-arm 36.3 ms: the cache-hit (~17 ms)
  prefill state never appeared in this fresh server session, so the
  flusher had nothing to evict and both arms ran in the same state. The
  registered cross-arm question (arms_differ) is therefore untested, per
  the gate — not answered.
- **Prediction 1 HELD at full n:** modal share 1.0 in both arms — all
  100 measured calls produced ONE variant.
- **Descriptive history match — the run's real contribution:** that
  variant is `20310cdd…`, companion A's B1/warmup-era class (prefill
  after a different-prompt call on a loaded model) — exactly the state
  class every companion-C call was in — reproduced byte-identically from
  a DIFFERENT server session. Combined with companion A, within-state
  determinism now spans two independent server sessions in the matching
  state class, which reframes the smoke-observed "cross-session drift"
  as state MISMATCH rather than time drift.
- Open (future window, not promised here): why the cache-hit state did
  not engage in this session (warm-up, VRAM pressure, or cache-admission
  behavior), and a pre-warm-until-cached design that would make the
  cross-arm endpoint testable.

## Companion D — pre-warmed cache-state A/B (pre-data plan, 2026-07-30 night)

**Status: exploratory-directional; this plan is committed BEFORE any
companion-D data.** Companion C's manipulation gate voided honestly: the
cached-prefill state never appeared in that fresh server session, so the
warm/cold contrast had nothing to contrast. Companion D is the same A/B
preceded by a **session-qualification gate** that proves the cached state
exists before any arm runs.

### Session-qualification prewarm (gate, not data)

- Repeatedly issue the EXACT frozen measured body (byte-identical to the
  A/B cell) and record `prompt_eval_duration_ns` per call.
- **Qualified** when 3 consecutive calls land below 25 ms — the threshold
  sits between the two observed prefill classes (~17 ms cached in
  companion A's session vs ~34–41 ms full-prefill in companions A and C).
- **Cap: 40 calls.** If the state has not engaged, the harness exits
  distinct (4) and the orchestration performs ONE full server restart and
  re-qualifies. If the second session also fails to qualify, NO A/B runs
  — the run is recorded as a session-qualification failure, which is
  itself a finding about state availability, not a silent skip.
- Prewarm calls are evidence, never analysis data: their prefill
  trajectory and output-sha sequence are written to
  `evidence/prewarm-cache-*.json` (the sha sequence captures any
  state-transition flip mid-prewarm — descriptive only).

### The A/B itself

Unchanged from companion C — same design, arms, byte-identical measured
bodies, flusher mechanics, manipulation gate (cold > 3x warm median;
warm median < cold median / 3), and registered endpoints (within-arm
modal shares predicted 1.0; arms_differ predicted true). The analyzer is
the committed `analysis/analyze_cache_ab.py`, untouched.

### Interpretation (stated pre-data)

- Qualification succeeds and both endpoints hold → the cache-state
  attribution is confirmed as a manipulated result: prose-length greedy
  output is deterministic conditional on prefill-cache state, which is
  the load-bearing claim behind the state-pinning pattern for
  reproducible long-form generation on owned hardware.
- Qualification succeeds, endpoint 2 fails (arms identical) → cache
  state does not change output in a qualified session; the companion-A
  decomposition needs a different explanation and the state-pinning
  claim is NOT supported at the prefill-cache layer.
- Qualification fails in both sessions → the cached state's availability
  is itself unstable; state pinning would have to pin the NO-cache
  regime instead (always-flushed), which companion C already measured
  internally deterministic at n=100.

### Companion D results (post-data, 2026-07-30 night)

Session QUALIFIED on attempt 1 in 4 calls; A/B ran 151/151, zero
failures (`runs/local-study3-cache-ab-20260730T231915Z.*`,
`reports/cache-ab-report-20260730T232908Z.*`,
`evidence/prewarm-cache-cuda-d1.json`).

- **A fourth outcome occurred, outside the three stated above:** the
  session qualified, but the A/B could not HOLD the cached state — the
  manipulation gate voided again (warm 35.8 ms ≈ cold 36.3 ms), so the
  registered cross-arm endpoint is untested for the second time. Both
  arms: modal 1.0, one shared variant.
- **The prewarm trajectory is the decisive artifact:** prefills
  [216.0, 17.8, 16.3, 17.9] ms — call 0 (fresh load) produced
  companion-A's fresh-load variant `45e27daf…`; calls 1–3 (cached)
  produced companion-A's cached-steady variant `cf2c66c8…`. The state
  transition and the byte flip occur at the same call, and both states
  reproduce the prior session's bytes.
- **The A/B's shared variant is the third state:** the serve logs
  identify it exactly — a persistent ~39-token context checkpoint is
  restored and 77 of 116 prompt tokens are evaluated (~36 ms) — and its
  bytes are `20310cdd…`, the same partial-state variant as companion A's
  B1, companion C, and the launch smoke: a third independent session.
- **Why the manipulation failed: the flusher is not the operative
  lever — inter-call timing is.** Back-to-back identical calls (the
  prewarm loop, no delay) reuse the full prompt KV (~17 ms); calls
  separated by the runner's anti-burst jitter (0.25–1.0 s) fall to the
  checkpoint state (~36 ms) in every position of every block, warm and
  cold alike.
- **Standing evidence after A+C+D:** three prefill states, each
  internally byte-deterministic at n=10–100, each observed in at least
  two of three independent server sessions, each producing the same
  bytes per state across sessions:
  fresh-load ~200–216 ms → `45e27daf…` · full-KV-cached ~16–18 ms →
  `cf2c66c8…` · checkpoint-partial ~34–41 ms → `20310cdd…`.
  The cross-arm prediction is directly evidenced descriptively (the
  three states' bytes differ) but remains unconfirmed under a
  registered manipulation gate. The follow-up that manipulates the
  discovered variable — inter-call timing — is companion E below.

## Companion E — timing-manipulated cache-state A/B (pre-data plan, 2026-07-30 night)

**Status: exploratory-directional; this plan is committed BEFORE any
companion-E data.** Companions C and D failed their manipulation gates
for the same instructive reason: the flusher was not the operative
variable. D's evidence identified the real one — **inter-call timing on
consecutive identical prompts**: back-to-back calls reuse the full
prompt KV (~16–18 ms prefill), while calls separated by the runner's
anti-burst jitter fall to a persistent-checkpoint partial state
(~34–41 ms). Companion E manipulates timing directly.

### Hypothesis (directional, stated pre-data)

With the model loaded and no flushers anywhere, the gap before a call
determines its prefill state and therefore its bytes: adjacent calls
(no deliberate gap) land in the full-KV state and are byte-identical to
each other; gapped calls (600 ms deliberate pause) land in the
checkpoint state and are byte-identical to each other; the two groups
differ from each other.

### Design

- **Cell:** the same frozen cell, byte-identical measured bodies in both
  arms (`gpt-oss-20b|open_generation|greedy|effort_low`).
- **Arms (n=50 each):** `adjacent` = pre-call sleep 0, anti-burst jitter
  suppressed · `gapped` = pre-call sleep 600 ms, jitter suppressed.
  Alternating mini-blocks of 10 (A,G,A,G,…), one warmup head, single
  flight, fixed schedule. No flushers — timing is the only manipulated
  variable.
- **One burn-in call** (the measured body, meta `control=burnin`,
  excluded from analysis like warmups) sits between the warmup and the
  first adjacent block: the warmup's different prompt would otherwise
  force the first adjacent call into the checkpoint state on a schedule
  technicality rather than the manipulated variable. Every adjacent call
  thereby follows a same-prompt call.
- **Session qualification:** the companion-D prewarm gate runs first
  (cached state must exist this session); one server-restart retry.
- **Box:** CUDA (non-production), same posture.

### Manipulation gate (arm void if failed)

Absolute thresholds from the three-session state classes: every
adjacent-arm call's prefill must land below 25 ms AND every gapped-arm
call's above 30 ms; cross-arm single request sha as always.

### Endpoints (registered)

1. Within-arm byte determinism: modal share per arm (prediction: 1.0 in
   both arms).
2. Cross-arm modal equality (prediction: the arms' modal outputs DIFFER).

Descriptive (not registered, consistent with C/D): history matching —
the standing three-session mapping expects adjacent → `cf2c66c8…` and
gapped → `20310cdd…`; any match or miss is reported as observed.

### Interpretation (stated pre-data)

Both endpoints holding under a passing gate = the state→bytes claim is
confirmed as a manipulated, registered result: prose-length greedy
output is a deterministic function of prefill-cache state, switchable by
call timing alone. Endpoint 2 failing under a passing gate = the states
share bytes in this session and the standing mapping is session-scoped
after all. Gate failing = timing does not control the state as D's
evidence indicated; report and stop — no further same-night designs.

### Companion E results (post-data, 2026-07-30 night) — TIMING FALSIFIED

Session qualified on attempt 1 (prewarm prefills [284.1, 16.3, 16.1,
16.5] ms — the fourth session reproducing both state variants'
bytes: `45e27daf…` fresh-load, `cf2c66c8…` cached). A/B ran 102/102,
zero failures (`runs/local-study3-cache-timing-20260730T233649Z.*`,
`reports/cache-timing-report-20260730T234514Z.*`).

- **Manipulation gate FAILED — the stated stop condition.** Adjacent
  median 36.1 ms ≈ gapped median 36.3 ms: truly back-to-back calls
  (pre-sleep 0, jitter suppressed) landed in the checkpoint state
  exactly like 600 ms-gapped calls. **Inter-call timing does not control
  the state; companion D's timing inference is falsified.** Both arms:
  modal 1.0, one shared variant — `20310cdd…` again.
- **The pattern that survives every run (A, C, D, E + prewarms):** the
  full-KV ~16–18 ms state exists only on a model instance that has NOT
  yet served a different prompt. The first different-prompt call (the
  warmup, a flusher) forms the persistent template-prefix checkpoint the
  serve logs show being restored, and every subsequent identical-prompt
  call lands at ~34–41 ms regardless of gap or repetition — until a
  reload resets the instance (companion A's post-churn blocks are the
  17 ms state for exactly this reason).
- **Standing hypothesis (for a future designed test, not tonight per the
  stop rule):** instance history, not timing — arms on fresh instances
  with the manipulation being a single interposed different-prompt call.
  Requires a warmup-free schedule variant.
- **What is now established descriptively across four server sessions:**
  three prefill states, each internally byte-deterministic (n up to
  100), each byte-stable per state across every session observed:
  fresh-load → `45e27daf…` · full-KV → `cf2c66c8…` · checkpoint →
  `20310cdd…`. The registered manipulated confirmation remains open.

## Companion F — fresh-instance manipulated confirmation (pre-data plan, 2026-07-31)

**Status: exploratory-directional; this plan is committed BEFORE any
companion-F data.** Companion E falsified inter-call timing; the standing
hypothesis after A+C+D+E is **instance history**: the full-KV ~16–18 ms
prefill state exists only on a model instance that has never served a
different prompt, and the first different-prompt call forms the
persistent template checkpoint the serve logs show being restored
(`evidence/serve-log-checkpoint-excerpts-20260730.txt`), pinning
~34–41 ms until reload. Companion F manipulates instance history
directly — fresh instances, with the manipulation being a single
interposed different-prompt call — the design companion E's results
prescribed.

### Hypothesis (directional, stated pre-data)

On a freshly reset instance that has only ever served the measured
prompt, same-prompt calls land in the full-KV state and are
byte-identical to each other; after ONE interposed different-prompt
call, subsequent same-prompt calls land in the checkpoint state and are
byte-identical to each other; the two groups differ from each other —
and the flip occurs exactly at the interposed call, in every cycle.

### Design

- **Unit = instance cycle; K = 5 cycles; CUDA (non-production box).**
  Each cycle:
  1. **Reset:** unload gpt-oss:20b (companion-A `pre_unload` machinery;
     absence confirmed via `/api/ps` before the next call).
  2. **Burn-in ×1:** the measured body itself — the fresh-load ~200 ms
     call (meta `control=burnin`, excluded from endpoints like warmups).
  3. **Arm P ("pure") ×10:** same-prompt measured calls, pre-sleep 0.
     Prediction: full-KV state (prefill < 25 ms), `cf2c66c8…`-class
     bytes.
  4. **Manipulation ×1:** ONE different-prompt call — the frozen
     classification flusher (meta `control=flusher`, excluded).
  5. **Arm C ("contaminated") ×10:** same-prompt measured calls,
     pre-sleep 0. Prediction: checkpoint state (prefill > 30 ms),
     `20310cdd…`-class bytes.
- **No warmup item anywhere** — a warmup's different prompt is itself
  the checkpoint trigger under test; the burn-in (same prompt) replaces
  it. All items carry pre-sleep 0 (timing was falsified in companion E;
  jitter stays suppressed so history is the only variable).
- **Cell:** the same frozen cell (`gpt-oss-20b|open_generation|greedy|
  effort_low`); measured bodies byte-identical across arms AND burn-ins.
- Totals: 50 measured calls/arm + 5 burn-ins + 5 flushers = 110 calls.
- Fixed schedule, single-flight; mode `study3-cache-instance`; no
  prewarm qualification — each cycle creates its own fresh instance.

### Manipulation gate (void if failed)

Every pure call's prefill below 25 ms AND every contaminated call's
prefill above 30 ms AND every cycle's reset confirmed
(`pre_unload_confirmed` on the burn-in record). Cross-arm negative
control: one request sha across burn-ins and both arms (flushers differ
by design and are excluded).

### Endpoints (registered)

1. Within-arm byte determinism pooled across cycles: modal share per arm
   (prediction: 1.0 in both arms).
2. Cross-arm modal equality (prediction: the arms' modal outputs
   DIFFER).
3. **Per-cycle flip at the interposed call** (prediction: in EVERY
   cycle, all pure calls match the pooled pure modal and all
   contaminated calls match the pooled contaminated modal — the state
   flip sits exactly at the flusher). This is the endpoint companions
   C, D, and E could not reach.

Sha expectations remain descriptive, per the four-session mapping
(fresh-load `45e27daf…` · full-KV `cf2c66c8…` · checkpoint
`20310cdd…`); any match or miss is reported as observed.

### Interpretation (stated pre-data)

- Gates pass + all three endpoints hold → state-pinning is a registered,
  manipulated result: instance history selects the prefill state, and
  the state selects the bytes.
- Gates pass, endpoint 3 fails → the hypothesis is wrong in an
  interesting way; report as such; paper #5 proceeds descriptive-only
  with the failure disclosed.
- Gate fails → stop and report; no same-day design iteration (the C/D/E
  lesson: three voids is the polite maximum).

## Execution + provenance

- Modes: `study3-churn-ab`, `study3-margins` — schema-3 records, manifest
  fields `exploratory: true`, `companion_plan: "FOLLOWUP-COMPANIONS.md"`,
  fixed schedules recorded by sha256.
- Runner code, tests, and this plan are committed before any companion
  call; raw runs, manifests, and reports are committed public after, as
  with every prior arc step.
- Production-box rules (unchanged): unload side-calls target gpt-oss:20b
  only; resident Qwens are never sent a keep_alive below 24h; teardown
  re-pulses residents and snapshots `/api/ps`.

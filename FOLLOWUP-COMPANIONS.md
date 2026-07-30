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

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
  measured call) AND a cold load: `load_duration_ns` > 10x the blocked
  arm's median.
- Blocked-arm records (post-warmup) must show warm loads: median
  `load_duration_ns` within the confirmatory run's warm range (~0.26 s).

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

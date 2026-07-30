# Pre-registration v3.0 (FROZEN): Local Open-Model Determinism Baseline — the Control Ceiling

**Status: FROZEN v3.0 — tag `prereg-v3`, pushed before the first
confirmatory call, same discipline as studies 1–2.** Four pilots preceded
this freeze (two CUDA, one Metal, one Q2 concurrency; n=10/cell; raw
records committed at `aeff3d5` and `2bc0132`); they are exploratory, form
no part of the confirmatory dataset, and every pilot-informed design
decision is recorded in section 6 and reflected in the sections below.

Lineage: study 1 (frozen `prereg-v1`, published) measured reproducibility on
one serving plane with zero determinism knobs; study 2 (frozen `prereg-v2`,
complete) attributed the thinking-mode effect across three serving planes —
the cost follows the AWS front door, and the two AWS doors are behaviorally
equivalent at 2pp while remaining distinguishable in latency profile
(exploratory addendum, `c5d99e7`). Both studies ran models we cannot
configure. Study 3 is the missing rung: **open weights on hardware we own**,
with temperature 0, a fixed seed, single-flight execution, and a warm model —
the *control ceiling*. Does byte-reproducibility actually reach 1.0 when you
control everything, and what breaks it? Because model and stack change
together relative to studies 1–2, **nothing here attributes API-plane
findings; cross-study comparisons are conceptual replications only.**

## 1. Questions

- **Q1 (primary, confirmatory — the ceiling).** Under full control — greedy
  decoding (temperature 0), fixed seed, single-flight, warm model, byte-
  identical request bodies — per-cell modal share of byte-identical response
  text on the frozen study-1 task ladder. **Registered hypothesis: the
  ceiling is NOT uniformly 1.0; deviations concentrate in structured JSON and
  open generation** (the tasks that discriminated in studies 1–2). Endpoint:
  per-cell modal share, Wilson 95% intervals — identical to studies 1–2 for
  cross-study comparability.
- **Q2 (confirmatory — concurrency).** Same cells at parallel load: 1 vs 4
  concurrent clients. Candidate mechanisms on record: continuous-batching
  non-associativity; MoE routing under load. **The registered contrast is
  MoE vs dense** (qwen3.5:122b-a10b and qwen3.6:35b-a3b vs qwen3-vl:32b
  dense): endpoint is the per-model concurrency effect (modal share at
  concurrency 4 minus concurrency 1) and the MoE-minus-dense
  difference-of-differences with 95% CI. Per study 2's recorded lesson, the
  estimator registered here is the estimator any power calculation must use:
  equal-weight mean of per-stratum differences over matched (task, sampling)
  strata with per-stratum binomial variances; pooled Wald computed as a
  labeled sensitivity only.
- **Q3 (confirmatory — the thinking analog).** Does thinking mode destabilize
  structured-JSON serialization under greedy decoding on non-Anthropic open
  weights? Qwen hybrid think on/off; gpt-oss reasoning effort low vs high.
  Endpoint: per-model thinking-on-minus-off (effort-high-minus-low)
  difference in structured-JSON modal share with 95% CI. Conceptual
  replication of the study-1/2 coin-flip mechanism under full control;
  either outcome is publishable and no direction is registered as expected.
  Models: qwen3.5-122b and qwen3.6-35b (bool think), gpt-oss-20b and
  gpt-oss-120b (effort). **qwen3-vl-32b is STRUCK from Q3** — pilot-verified
  non-hybrid (rejects `think: true` with a 400 on the pinned engine); it
  retains its core-grid and Q2 dense-comparator roles on `think: false`,
  which it accepts (32/32 in the Metal pilot).
- **Q4 (confirmatory — the one clean cross-comparison; arm CONFIRMED IN,
  owner decision 2026-07-29).** gpt-oss:20b, same engine and engine version,
  same weights digest, on two hardware stacks: Metal (Mac Pro M2 Ultra) vs
  CUDA (RTX 4090). (a) Within-box reproducibility per hardware; (b)
  **cross-box identity**: do the two boxes produce each other's modal bytes
  at identical settings? **Registered expectation (pilot-revised, recorded
  in section 6): within-box high; cross-box BYTE-IDENTICAL on short
  outputs, with divergence emerging as generation length grows** — the
  pilots found every short-output greedy cell byte-identical across Metal
  and CUDA while long generation diverged on Metal only. Exploratory
  alongside: per-box per-token decode-rate from recorded call latency —
  the known-ground-truth calibration for the study-2 latency-fingerprint
  readout (pilot: CUDA 159.9 vs Metal 97.9 tok/s on identical weights and
  engine).

Analysis code for the Q1–Q4 estimators and validity gates above is
implemented and committed BEFORE any confirmatory data
(`analysis/analyze_study3.py`, `a5443dc`): Q2's stratified estimator and
MoE-minus-dense DoD, Q3's Wald difference, Q4's cross-box identity readout,
the exact wire-hash negative control, warmup exclusion, and the temp07
positive-control firing check.

## 2. Design (confirmatory grid — reduced factorial, NOT full cross)

| Factor | Levels | Applied to |
|---|---|---|
| Model | qwen3.5:122b-a10b (MoE) · qwen3.6:35b-a3b-q8_0 (MoE) · qwen3-vl:32b-instruct-q8_0 (dense) · gpt-oss:20b | core grid |
| Task | frozen study-1 ladder (4), byte-identical prompts | all |
| Sampling | greedy temp 0 + fixed seed · UNSEEDED temp 0.7 (positive-control analog, per model) | all |
| Thinking | think on/off (Qwen hybrid) · effort low/high (gpt-oss) | Q3 cells only |
| Concurrency | 1 · 4 | Q2 cells only |
| Hardware | Metal (M2 Ultra) · CUDA (4090) | gpt-oss:20b only (Q4) |

- Wall-clock, not dollars, is the budget: **n=100/cell CONFIRMED for every
  model at freeze.** The 122b measured 46.6 tok/s in the Metal pilot, so
  its 800 core calls are affordable; n=100 matches studies 1–2, and the
  section-1 estimators at n=100 deliver the same per-cell Wilson precision
  those studies published.
- **Scheduling (pilot-informed, section 6): confirmatory runs execute in
  per-model blocks** (`apply_model_blocks` — a stable sort of the shuffled
  schedule, so the per-cell ordering control is unchanged; model was never
  a within-cell factor). The Metal pilot's fully-shuffled schedule spent
  most of its 92 minutes swapping models on an over-budget box.
- **Windows (recorded at freeze):** one Metal full window + one CUDA full
  window (core grid, n=100) · Q3 arm cells alongside the full windows ·
  one Q2 concurrency window (Metal) · ONE dedicated gpt-oss:120b window
  (Metal). Owner posture recorded 2026-07-29 ("low-use box; loading is
  fine"); each window runs with owner awareness, box state snapshotted
  before and after.
- gpt-oss:120b (65 GB): **REGISTERED single-dedicated-window arm (owner
  decision 2026-07-29; criterion: active community use of the open-weights
  release — Apache-2.0 weights on Hugging Face, multiple commodity
  inference providers, single-H100 deployment class).** Its 9-cell slice
  (4 tasks × 2 sampling at pinned effort_low, plus the structured-JSON
  effort_high cell) runs in exactly ONE dedicated window via the
  `study3-120b-window` mode, Metal box only; it never joins core-grid
  windows (loading 65 GB evicts the production residents).
- Windows: runs occur only in owner-approved windows on the production box
  (see section 6); the window schedule is recorded at freeze.

## 3. Machinery (carried forward, plus local-only upgrades)

Primary endpoint machinery identical to studies 1–2: per-cell modal share of
byte-identical response text, Wilson 95%; semantic-equality readout alongside
for structured JSON; exclusion rules; ordering control (seeded shuffle, no
same-cell concurrency outside Q2 cells, jitter); no silent retries, attempts
recorded per call; deterministic measurement with no model in the loop; raw
records, manifests, and reports committed to this public repository.

**Negative control, exact:** requests go over local HTTP (stdlib client, no
SDK) — the harness owns every byte. The canonical body is hashed per cell
and the bytes actually sent are hashed per call; they must be equal by
construction, and more than one distinct wire hash within a cell fails the
cell. This is strictly cleaner than the SDK planes of study 2.

**Drift control, upgraded to weights level:** the model digest (weights
hash as reported by the local runtime) is recorded per run and must be
constant within the confirmatory dataset — the control the API studies
structurally could not have. Engine name and version are recorded per run
and must be identical across both boxes for Q4.

**New local-only endpoints (pilot-resolved):** the engine decision is
CLOSED — Ollama exposes per-token logprobs
(`evidence/logprobs-probe-metal.json`); no llama.cpp fallback, one engine
for the whole confirmatory dataset. **First-divergence token index:
REGISTERED** — it is text-derived and needs no request-side capture.
**Logprob margin: EXPLORATORY COMPANION ONLY.** The freeze probe found the
logprobs request fields are NOT generation-neutral at length
(`evidence/logprobs-bytes-probe-cuda-4090.json`: three calls with logprobs
produced three distinct outputs on a cell that is 20/20 byte-identical
without them, while all short-output cells stayed byte-neutral). Margin
capture is therefore barred from the frozen confirmatory bodies; margins
come from separate, clearly-labeled companion runs that form no part of
the confirmatory dataset. The non-neutrality itself is disclosed as a
standalone observation.

**Box-state covariate:** resident models (`ollama ps`), engine version,
parallelism setting, keep-alive state, and (4090) GPU driver version are
snapshotted before and after every window; contention appears as a recorded
covariate, not silent noise.

**Latency capture:** per-call wall-clock latency and token usage are recorded
exactly as in studies 1–2, feeding the Q4 exploratory calibration readout.

## 4. Known differences to document, not hide

- Model identity includes quantization: the quant tag and weights digest are
  part of the registered model, and the same digest must serve both boxes in
  Q4. A digest mismatch at setup time is a freeze blocker, not a footnote.
- The two boxes differ in OS, GPU driver, and thermal envelope; Q4
  attributes only "hardware stack" as a bundle. Kernel-level attribution is
  out of scope.
- Local serving has no multi-tenant queue: studies 1–2's time-of-day windows
  do not map here. Load variation is introduced explicitly (Q2 concurrency),
  not ambiently.
- The production Mac Pro serves live workloads outside study windows;
  within-window residency is controlled and recorded (section 3 covariate).
- These are different models from studies 1–2. No result here attributes any
  Anthropic-model finding; the thinking analog (Q3) is a conceptual
  replication, registered as such.

## 5. Freeze checklist (CLOSED at freeze — every item verified)

- [x] `LocalPlane` client + `canonical_local_body()` builder + invariant
      tests committed (`c25b3b5`), with a live instrument smoke 6/6 PASS on
      the pinned CUDA box (engine 0.30.5, gpt-oss effort-level AND bool
      think acceptance verified on-engine, 404 classification verified;
      `evidence/smoke-local-cuda-4090.json`). Qwen-family field acceptance
      VERIFIED 2026-07-29 on the production box's resident models with
      keep_alive 24h and box-state proof of undisturbed residency
      (`evidence/smoke-local-metal-qwen36.json`,
      `evidence/smoke-local-metal-qwen35-122b.json`, both 6/6: bool think
      on/off accepted on qwen3.5-122b AND qwen3.6-35b; effort-level
      spelling also accepted, recorded). ⚠️ qwen3-vl:32b (not resident;
      loading it risks evicting production models) remains unverified —
      its Q3 membership is decided at freeze
- [x] Ollama for Windows installed on the 4090 — v0.30.5, **version-pinned
      to the Mac Pro's release** (versioned GitHub installer, not latest);
      gpt-oss:20b pulled; **weights digest identical across boxes
      (`17052f91a42e`)**; CUDA smoke generation PASS. No autostart
      registered — the server runs only inside study windows.
      (2026-07-29; arm confirmed IN the same day)
- [x] Logprobs exposure VERIFIED on the pinned engine (2026-07-29,
      `evidence/logprobs-probe-metal.json`, qwen3.6:35b on 0.30.5):
      top-level `logprobs: true` returns per-token logprobs and
      `top_logprobs: N` returns ranked alternatives with logprobs; the
      options-object spelling is accepted but inert. **Engine decision:
      Ollama, no llama.cpp fallback — the first-divergence and
      logprob-margin endpoints are REGISTERED, reading the top-level
      fields.** Grid bodies remain logprob-free; logprob capture rides on
      the registered request shape only if added at freeze as an explicit
      field of the frozen bytes — RESOLVED AT FREEZE: NOT added; the
      byte-neutrality probe failed at generation length (section 3), so
      margins run as an exploratory companion only
- [x] Pilots run clean on both boxes, all models (CUDA ×2 91/91; Metal
      364/364 with the 10 vl think-on rejections recorded; Q2 123/123) —
      disclosed in section 6; raw data committed `aeff3d5` + `2bc0132`
- [x] n=100/cell confirmed for every model (122b measured 46.6 tok/s in
      the Metal pilot); the section-1 estimators at n=100 match the
      per-cell precision studies 1–2 published at the same n
- [x] gpt-oss:120b arm REGISTERED (owner decision 2026-07-29, criterion:
      active community use — verified same day); dedicated-window mode
      implemented (`study3-120b-window`, single-flight, Metal only)
- [x] Window schedule recorded (section 2) under the owner posture of
      2026-07-29; per-model block scheduling adopted from the Metal
      pilot's swap-thrash finding
- [x] This file bumped to v3.0, tagged `prereg-v3`, pushed before any
      confirmatory call

## 6. Pilot disclosure (exploratory, not evidence)

Four pilots and two freeze probes preceded this freeze; all raw records,
manifests, reports, and probe evidence are committed (`aeff3d5`,
`2bc0132`, plus this freeze commit). n=10/cell throughout. Every
pilot-informed design decision is listed here; hypotheses and endpoints
changed only where explicitly recorded.

- **CUDA pilot #1 (91/91 clean):** the sampling arm as first drafted
  carried the fixed seed at temp 0.7 and the pinned engine reproduced it
  byte-for-byte — a seeded local sampler is not a divergence control.
  DESIGN CHANGE (recorded before any further runs): the temp-0.7
  positive-control arm is UNSEEDED. Also disclosed: the seeded-sj cell
  split 7/3 across byte-identical single-flight requests — sampling
  amplifies sub-argmax numeric jitter.
- **CUDA pilot #2 (91/91, all gates green):** positive control fires
  (open-generation temp07 10/10 distinct); ALL greedy cells at modal
  share 1.0.
- **Metal pilot (364/364, 10 recorded failures):** all 10 failures are
  the qwen3-vl think_on cell (400 "does not support thinking");
  think:false accepted 32/32 → DESIGN CHANGE: vl-32b struck from Q3,
  retained in the core grid and Q2 (section 1). Positive controls fired
  on every model. The single greedy cell below ceiling anywhere in the
  pilots: Metal gpt-oss open-generation (0.60, 3 variants) — the Q1
  hypothesis's predicted concentration in open generation, observed at
  pilot n. Wall clock was dominated by model swaps → DESIGN CHANGE:
  per-model block scheduling (section 2).
- **Q2 concurrency pilot (123/123 clean):** all 12 cells byte-identical
  at concurrency 4, including the 122b MoE; the 35b decode-rate
  bimodality confirms the load was genuinely parallel. No direction is
  inferred at n=10.
- **Cross-box (pilots #2 + Metal):** every short-output greedy cell
  byte-identical across Metal and CUDA; divergence only in long
  generation (Metal) and the by-construction-unseeded temp07 cells →
  DESIGN CHANGE: Q4 registered expectation revised (section 1). Decode
  calibration: CUDA 159.9 vs Metal 97.9 tok/s on identical weights,
  engine, and digest.
- **Logprobs byte-neutrality probe (freeze gate):** the logprobs request
  fields are NOT generation-neutral at length — three calls with
  logprobs yielded three distinct outputs on a cell that is 20/20
  byte-identical without them, while every short-output cell stayed
  byte-neutral (`evidence/logprobs-bytes-probe-cuda-4090.json`) →
  DESIGN CHANGE: margin capture exiled to exploratory companion runs
  (section 3). The non-neutrality is itself a disclosed observation.

## 7. Confirmatory run plan (frozen)

Runs execute inside owner-approved windows per the section-2 schedule —
Metal full window, CUDA full window, Q2 concurrency window, one dedicated
gpt-oss:120b window — in per-model blocks, single-flight except the Q2
arm, box state snapshotted before and after every window. Analysis runs
`analysis/analyze_study3.py` (committed pre-data, `a5443dc`) over
confirmatory files only; the four pilots and all probes are excluded as
disclosed in section 6. Raw records, manifests, and reports are committed
to this public repository.

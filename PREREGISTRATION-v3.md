# Pre-registration v3 (DRAFT): Local Open-Model Determinism Baseline — the Control Ceiling

**Status: DRAFT — NOT FROZEN.** Freeze = this file bumped to v3.0 and tag
`prereg-v3` pushed before the first confirmatory call, same discipline as
studies 1–2. Pilot runs are permitted before the freeze; pilot data forms no
part of the confirmatory dataset and any pilot-informed design decision is
recorded explicitly in this file at freeze time.

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
- **Q4 (confirmatory — the one clean cross-comparison; arm CONFIRMED IN,
  owner decision 2026-07-29).** gpt-oss:20b, same engine and engine version,
  same weights digest, on two hardware stacks: Metal (Mac Pro M2 Ultra) vs
  CUDA (RTX 4090). (a) Within-box reproducibility per hardware; (b)
  **cross-box identity**: do the two boxes produce each other's modal bytes
  at identical settings? **Registered expectation: within-box high,
  cross-box NOT byte-identical** (kernel and accumulation-order
  differences). Exploratory alongside: per-box per-token decode-rate from
  recorded call latency — the known-ground-truth calibration for the
  study-2 latency-fingerprint readout (same weights, same engine, different
  silicon: what slope difference does a hardware swap actually produce?).

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
| Sampling | greedy temp 0 + fixed seed · seeded temp 0.7 (positive-control analog, per model) | all |
| Thinking | think on/off (Qwen hybrid) · effort low/high (gpt-oss) | Q3 cells only |
| Concurrency | 1 · 4 | Q2 cells only |
| Hardware | Metal (M2 Ultra) · CUDA (4090) | gpt-oss:20b only (Q4) |

- Wall-clock, not dollars, is the budget: **repeats target n=100/cell on the
  small/fast models; n for qwen3.5:122b is set at freeze from pilot
  tokens/sec.** The registered estimator (section 1) is the basis for any
  n justification recorded at freeze.
- gpt-oss:120b (65 GB): **optional single-dedicated-window arm — owner
  decision OPEN.** Loading it evicts the production-resident models on the
  Mac Pro; if approved it runs in exactly one dedicated window and is
  registered (or struck) at freeze.
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

**New local-only endpoints (pilot-gated):** first-divergence token index and
logprob margin at the divergence point, for cells with byte-divergent
responses. ⚠️ These are registered ONLY IF the pilot verifies the runtime
exposes per-token logprobs; the fallback (llama.cpp server) changes the
engine and therefore the study — the engine decision is made once, before
freeze, and engines are never mixed within the confirmatory dataset.

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

## 5. Freeze checklist (OPEN — all must close before tag `prereg-v3`)

- [x] `LocalPlane` client + `canonical_local_body()` builder + invariant
      tests committed (`c25b3b5`), with a live instrument smoke 6/6 PASS on
      the pinned CUDA box (engine 0.30.5, gpt-oss effort-level AND bool
      think acceptance verified on-engine, 404 classification verified;
      `evidence/smoke-local-cuda-4090.json`). Qwen-family field acceptance
      still to verify in a Mac Pro window (smoke_local `--family qwen`)
- [x] Ollama for Windows installed on the 4090 — v0.30.5, **version-pinned
      to the Mac Pro's release** (versioned GitHub installer, not latest);
      gpt-oss:20b pulled; **weights digest identical across boxes
      (`17052f91a42e`)**; CUDA smoke generation PASS. No autostart
      registered — the server runs only inside study windows.
      (2026-07-29; arm confirmed IN the same day)
- [ ] Logprobs exposure verified in pilot → first-divergence/logprob-margin
      endpoints confirmed or struck; engine decision (Ollama vs llama.cpp)
      recorded — single engine for the whole confirmatory dataset
- [ ] Pilot (small n, both boxes, all models) run clean; disclosed in
      section 6 at freeze
- [ ] n/cell for qwen3.5:122b set from pilot tokens/sec; power basis recorded
      using the registered estimator
- [ ] gpt-oss:120b arm registered or struck (owner decision)
- [ ] Window schedule on the production box approved and recorded
- [ ] This file bumped to v3.0, tagged `prereg-v3`, pushed before any
      confirmatory call

## 6. Pilot disclosure

None yet. To be completed at freeze: pilot scope, results, and every
pilot-informed design decision, following the study-2 template.

## 7. Confirmatory run plan (to finalize at freeze)

Runs execute inside owner-approved windows on the Mac Pro (production box)
and on the 4090 for Q4 cells; box state is snapshotted before/after each
window. Analysis runs the existing pipeline over confirmatory files only
(pilot excluded by filename convention), with per-question estimators as
registered in section 1. Raw records, manifests, and reports are committed
to this public repository.

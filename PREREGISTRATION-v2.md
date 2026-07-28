# Pre-registration v2 (DRAFT): Cross-Plane Attribution of Inference Reproducibility Effects

**Status: DRAFT — NOT FROZEN. Freeze = tag `prereg-v2` after per-plane pilots.**
Successor to the completed study frozen at `prereg-v1` (see PREREGISTRATION.md and
the published results). Study 1 measured reproducibility on one serving plane
(AWS Bedrock). Its headline finding — adaptive thinking costing ~39 points of
byte-level reproducibility on structured JSON — is unattributed between the
models and the serving stack. Study 2 exists to attribute it.

## 1. Questions

- **Q1 (primary, confirmatory — attribution).** Does the thinking-mode effect on
  structured-JSON byte-reproducibility replicate on Anthropic-operated serving
  planes? Same models, three planes: Amazon Bedrock (AWS-operated), Claude
  Platform on AWS (Anthropic-operated behind AWS front door), Anthropic API
  (first-party). If the effect is model-intrinsic it appears on all three at
  comparable magnitude; if it is serving-stack behavior it attenuates or
  vanishes off Bedrock. Endpoint: per-plane adaptive-minus-disabled difference
  in structured-JSON modal share with 95% CI, plus the cross-plane
  difference-of-differences with 95% CI.
- **Q2 (confirmatory — plane equivalence).** Pairwise equivalence of pooled
  exact-match rates across the three planes, TOST at **delta = 2pp**, with the
  **stratified variance estimator registered as the primary test statistic** —
  the estimator the power analysis uses. This corrects study 1's recorded
  methodological miss (its frozen test used a conservative pooled-variance Wald
  SE roughly twice the stratified sampling error, and returned inconclusive at
  a margin the data plainly satisfied under the matched estimator).
- **Q3 (exploratory).** Streaming vs non-streaming delivery on structured JSON
  and open generation, per plane.
- **Q4 (exploratory).** Sparse input-length ladder (roughly 1k / 10k / 50k
  padded input tokens) on the extraction task, reduced repeats, cost-bounded.

## 2. Design

| Factor | Levels | Notes |
|---|---|---|
| Plane | 3 | Bedrock (`us.anthropic.claude-*`, InvokeModel) · Claude Platform on AWS (bare IDs, SigV4) · Anthropic API (bare IDs, API key) |
| Model | 2 + anchor | `claude-opus-5`, `claude-sonnet-5`; Haiku 4.5 as dated anchor + per-plane positive-control host |
| Task | 4 | The frozen study-1 ladder, byte-identical prompts |
| Thinking | 2 | adaptive vs disabled, effort pinned `medium` (5-family only) |
| Windows | 2 | peak + low UTC (study 1 measured time-of-day flat: 0.6923 / 0.6957 / 0.6929 — recorded justification for dropping to two) |
| Repeats | 100 per cell (pilot-adjusted) | Q4 arm reduced-n |

Estimated call volume ~13–14k; cost estimate $150–200 at list prices (Q4's
padded-input arm dominates; trimmed or cut before freeze if the pilot says so).

## 3. Carried-forward machinery (unchanged from study 1)

Primary endpoint: per-cell modal share of byte-identical response text, Wilson
95% intervals. Semantic-equality readout reported alongside for structured
JSON. Exclusion rules, ordering control (seeded shuffle, no same-cell
concurrency, jitter), negative control (one request SHA-256 per cell —
within-plane; request bodies necessarily differ across planes), version-drift
recording (returned model IDs; still blind on undated 5-family IDs — Haiku
anchor per plane), cache covariates, thinking-token accounting, deterministic
measurement with no model in the loop, raw records committed.

Positive control runs **per plane**: Haiku 4.5 at temperature 0.7 (sampling
parameters remain accepted there), n=100 on each plane; each plane's null is
interpretable only if its own control fires.

## 4. Known plane differences to document, not hide

- Request body shapes differ by construction: Bedrock InvokeModel carries
  `anthropic_version` and no `model` field; the Messages API planes carry
  `model` in the body. Cross-plane identity is defined at the semantic level
  (same prompts, same parameters); byte-identity claims are always within-cell.
- Auth differs (SigV4 vs API key). No global `ANTHROPIC_API_KEY` is ever set on
  the operator machine; credentials are scoped to the run.
- Routing scope existed only on Bedrock in study 1 (`us.` vs `global.`,
  bounded within ±2.6pp at 90% confidence). Study 2 pins Bedrock to `us.` and
  treats plane as the routing-analog factor.

## 5. Freeze checklist

- [ ] Claude Platform on AWS workspace enabled; `ANTHROPIC_AWS_WORKSPACE_ID`
      available (owner console step)
- [ ] First-party API credential minted, run-scoped (owner)
- [ ] Anthropic SDK client layer added to harness (`anthropic` package;
      runner-only dependency, analysis stays stdlib) with per-plane request
      builders and unit tests
- [ ] Per-plane smoke (expected-rejection cases included) committed to
      evidence/
- [ ] Per-plane pilot (n=20/cell, one window); final n, delta unchanged or
      widened with reasoning; Q4 arm kept, trimmed, or cut
- [ ] This file bumped to v2.0, tagged `prereg-v2`, pushed before any
      confirmatory call

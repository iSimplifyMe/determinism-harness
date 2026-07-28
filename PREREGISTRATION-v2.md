# Pre-registration v2.0 (FROZEN): Cross-Plane Attribution of Inference Reproducibility Effects

**Status: FROZEN v2.0 — tag `prereg-v2`, pushed before the first confirmatory
call.** Successor to the completed study frozen at `prereg-v1` (see
PREREGISTRATION.md and the published results). Study 1 measured reproducibility
on one serving plane (AWS Bedrock). Its headline finding — adaptive thinking
costing ~39 points of byte-level reproducibility on structured JSON — is
unattributed between the models and the serving stack. Study 2 exists to
attribute it.

The pilot preceding this freeze (n=20/cell, one window, 1,200 calls, committed
at `83578a7`) is **exploratory and forms no part of the confirmatory dataset**.
It is disclosed in section 6; the hypotheses and endpoints below are unchanged
from the pre-pilot draft (`15e79bf`) except where a pilot-informed design
decision is explicitly recorded.

## 1. Questions

- **Q1 (primary, confirmatory — attribution).** Does the thinking-mode effect on
  structured-JSON byte-reproducibility replicate on Anthropic-operated serving
  planes? Same models, three planes: Amazon Bedrock (AWS-operated), Claude
  Platform on AWS (Anthropic-operated behind AWS front door), Anthropic API
  (first-party). If the effect is model-intrinsic it appears on all three at
  comparable magnitude; if it is serving-stack behavior it attenuates or
  vanishes off Bedrock. Endpoint: per-plane adaptive-minus-disabled difference
  in structured-JSON modal share with 95% CI, plus the cross-plane
  difference-of-differences with 95% CI. Analysis code implemented and
  committed BEFORE any confirmatory data (`34ae0b8`:
  `q1_attribution__*` in analysis/analyze.py).
- **Q2 (confirmatory — plane equivalence).** Pairwise equivalence of pooled
  exact-match rates across the three planes, TOST at **delta = 2pp**, with the
  **stratified variance estimator registered as the primary test statistic**
  (equal-weight mean of per-stratum differences; per-stratum binomial
  variances; strata are matched (model, task, thinking, window) cells) — the
  estimator the power analysis uses. The cross-stratum pooled Wald TOST is
  computed as a **labeled sensitivity only**. This corrects study 1's recorded
  methodological miss. Implemented before confirmatory data (`34ae0b8`:
  `stratified_tost` in analysis/stats.py; `q2_plane__*`).
- **Q3 (exploratory — streamed delivery).** Streamed vs non-streamed delivery
  on structured JSON and open generation. Design frozen: opus-5 and sonnet-5,
  adaptive thinking, all three planes, n=100/cell, streamed arm only
  (`study2-q3-streaming`, 12 cells, control window); the non-streamed
  comparators are the main grid's own cells. Streamed requests are
  parameter-identical to their comparators (test-enforced); on the Messages
  planes the SDK adds the stream field to the wire body, so wire hashes are
  compared within-cell only, as everywhere.
- **Q4 (exploratory — sparse input-length ladder).** Kept at full registered
  size (decision recorded 2026-07-28, spend approved). Char-exact deterministic
  padding prepended to the extraction task at 3,700 / 37,000 / 185,000 chars
  (~1k / 10k / 50k tokens): opus-5 and sonnet-5, adaptive, all three planes,
  n=25/cell (`study2-q4-lengths`, 18 cells, control window).

## 2. Design (confirmatory grid)

| Factor | Levels | Notes |
|---|---|---|
| Plane | 3 | Bedrock (`us.anthropic.claude-*`, InvokeModel) · Claude Platform on AWS (bare IDs, SigV4, workspace us-east-1) · Anthropic API (bare IDs, run-scoped key) |
| Model | 2 + anchor | `claude-opus-5`, `claude-sonnet-5`; Haiku 4.5 as dated anchor (`claude-haiku-4-5-20251001` on the Messages planes) + per-plane positive-control host |
| Task | 4 | The frozen study-1 ladder, byte-identical prompts |
| Thinking | 2 | adaptive vs disabled, effort pinned `medium` (5-family only) |
| Windows | 2 | peak (15:00–19:00 UTC) + low (07:00–10:00 UTC); study 1 measured time-of-day flat (0.6923 / 0.6957 / 0.6929) — recorded justification for two windows |
| Repeats | **100 per cell — CONFIRMED at freeze** (pilot supported feasibility; no revision) | Q4 arm n=25 |

60 grid cells per window (confirmed by pilot manifest), 6,000 calls per
window, 12,000 confirmatory + 300 positive control + 1,200 Q3 + 450 Q4.
Total ≈ 13,950 calls; spend approved 2026-07-28 (~$150–200 at list prices).

## 3. Carried-forward machinery (unchanged from study 1, plus one upgrade)

Primary endpoint: per-cell modal share of byte-identical response text, Wilson
95% intervals. Semantic-equality readout reported alongside for structured
JSON. Exclusion rules, ordering control (seeded shuffle, no same-cell
concurrency, jitter), version-drift recording (returned model IDs; still blind
on undated 5-family IDs — dated Haiku anchor per plane), cache covariates,
thinking-token accounting, deterministic measurement with no model in the
loop, raw records committed.

**Negative control, upgraded:** the planned request is hashed canonically per
cell as in study 1, and on every plane the harness additionally records the
SHA-256 of the **bytes actually sent**, captured at the HTTP layer (httpx
request hook on the SDK planes; on Bedrock hashed == sent by construction).
More than one distinct wire hash within a cell fails the cell's negative
control (`gate_cell` wire check, `06e77bf`). Request bodies necessarily differ
ACROSS planes (Bedrock carries `anthropic_version`, no `model`; Messages
planes carry `model`, no `anthropic_version` — invariant-tested); byte-identity
claims are always within-cell, within-plane.

Positive control runs **per plane**: Haiku 4.5 at temperature 0.7 (sampling
parameters remain accepted there), n=100 on each plane
(`study2-positive-control`); each plane's null is interpretable only if its
own control fires.

## 4. Known plane differences to document, not hide

- Request body shapes differ by construction (see section 3). Cross-plane
  identity is defined at the semantic level (same prompts, same parameters).
- Auth differs (SigV4 vs API key). No global `ANTHROPIC_API_KEY` is ever set on
  the operator machine; credentials are scoped to the run's process.
- Routing scope existed only on Bedrock in study 1 (`us.` vs `global.`,
  bounded within ±2.6pp at 90% confidence). Study 2 pins Bedrock to `us.` and
  treats plane as the routing-analog factor.
- Retry classification: Bedrock retains study 1's code-name semantics
  (single-sourced); Messages planes classify by HTTP status. No silent
  retries anywhere; attempts recorded per call.

## 5. Freeze checklist (all complete at tag time)

- [x] Claude Platform on AWS workspace enabled, us-east-1
      (`wrkspc_…`, owner console step 2026-07-28)
- [x] First-party API credential minted, run-scoped (macOS Keychain;
      per-invocation process env)
- [x] Anthropic SDK client layer with per-plane request builders and unit
      tests (`9f6e3cf`)
- [x] Per-plane smoke incl. expected-rejection cases, committed to evidence/
      (`e4243c1` p_aws 7/7, `6b2e8ac` anthropic_api 7/7; Bedrock smoke from
      study 1 stands)
- [x] Per-plane pilot n=20/cell, one window — 1,200/1,200 clean (`83578a7`);
      **final n=100 and delta=2pp confirmed unchanged**; **Q4 kept at full
      size**; Q3 streamed arm built (`06e77bf`)
- [x] This file bumped to v2.0, tagged `prereg-v2`, pushed before any
      confirmatory call

## 6. Pilot disclosure (exploratory, not evidence)

The 2026-07-28 pilot ran clean (1,200/1,200, zero failures, all controls
green including the wire-byte gate on all 60 cells). Its point estimates are
disclosed for transparency and DO NOT modify the hypotheses above: the
thinking-mode effect on structured JSON appeared large on Bedrock and near
zero on the first-party plane for opus-5, while sonnet-5 showed its largest
effect on Claude Platform on AWS — heterogeneous plane dependence, in both
directions, at n=20 precision (±5pp per call). The confirmatory grid exists
to measure exactly this; no direction is registered as expected.

## 7. Confirmatory run plan

Control-window arms (positive control, Q3, Q4) run first, then the low window
(07:00–10:00 UTC) and the peak window (15:00–19:00 UTC), each as a single
compressed `study2-full` run inside its window, orchestrated unattended.
Analysis runs `analysis/analyze.py` over the confirmatory files only (pilot
excluded), with delta = 0.02. Raw records, manifests, and reports are
committed to this public repository.

# Pre-registration: Reproducibility of AWS Bedrock Inference on Models Without Sampling Controls

**Status: DRAFT v0.1 — NOT YET FROZEN.**
This document becomes binding at the freeze commit (tag `prereg-v1`). Confirmatory
windows run only after the freeze. The pilot may run before the freeze; its only
permitted influence on this document is the final per-cell sample size (section 5)
and any widening of the equivalence margin, both recorded here before freezing.
The commit timestamp of the freeze, made before any confirmatory data exists, is
what makes the eventual result checkable rather than asserted.

Analysis code referenced here (`analysis/analyze.py`, `analysis/stats.py`,
`analysis/metrics.py`) is committed in this repository and pinned by the freeze
commit. All measurement is deterministic code; no model is anywhere in the
measurement loop.

## 1. Background and questions

On the current Claude 5 family, `temperature`, `top_p`, and `top_k` are removed —
sending them returns a validation error (verified live on Bedrock 2026-07-27;
`evidence/smoke.json`, case `opus5-us-temp07-must-reject`). There is no
determinism knob left to misuse. The question is the reproducibility of a
frontier model that cannot be configured for determinism at all. The candidate
mechanism for residual nondeterminism is floating-point non-associativity under
dynamic batching: batch composition changes reduction order inside matrix
multiplications, low-order bits shift, and an argmax can flip at a token
boundary, after which everything diverges.

- **Q1 (descriptive).** With no sampling parameters available, what is the
  exact-match reproduction rate for byte-identical Bedrock requests, by model
  and task type?
- **Q2 (confirmatory, equivalence).** Does worldwide routing
  (`global.` inference profile) measurably change exact-match reproducibility
  versus US-bounded routing (`us.` profile)?
- **Q3 (confirmatory, difference).** Does adaptive thinking (the default on the
  Claude 5 family) change final-answer reproducibility versus
  `thinking: disabled`?
- **Q4 (descriptive).** Does divergence correlate with time-of-day load window?

### Why Q2 is routing-form vs routing-form

The original design compared the `us.` cross-region profile to single-region
on-demand invocation. That comparison cannot exist: `list-foundation-models`
reports `inferenceTypesSupported = ["INFERENCE_PROFILE"]` — no ON_DEMAND — for
all three study models (verified live 2026-07-27, account 024033896674,
us-east-1; `evidence/inference-profiles.json`). This is itself a finding: on
current Claude models, Bedrock offers no single-region on-demand path, so every
deployment chooses a routing scope. Both the `us.` and `global.` system-defined
profiles are ACTIVE for all three models, and the reshaped Q2 compares exactly
the choice every current deployment faces. Directionally, if routing scope
matters at all, the `global.` pool (larger, more heterogeneous hardware and
traffic mix) is expected to reduce reproducibility relative to `us.`; the
equivalence test is two-sided regardless.

## 2. Design

Full factorial where the API permits it, verified against live Bedrock behavior:

| Factor | Levels | Notes |
|---|---|---|
| Model | 3 | `us./global.anthropic.claude-opus-5`, `...claude-sonnet-5`, `...claude-haiku-4-5-20251001-v1:0` |
| Task | 4 | extraction, classification, structured JSON, open generation (frozen in `harness/tasks.py`) |
| Routing profile | 2 | `us.` vs `global.` |
| Thinking | 2 (Claude 5 family only) | `{"type":"adaptive"}` vs `{"type":"disabled"}`; Haiku 4.5 runs one arm with the field omitted |
| Repeats per cell | 100 (pilot-adjusted; section 5) | |
| Load windows | 3 | low / mid / peak (UTC; PROTOCOL.md) |

Grid: (2 models x 4 tasks x 2 profiles x 2 thinking) + (1 model x 4 tasks x
2 profiles x 1 arm) = **40 cells per window**, 4,000 calls per window at n=100,
12,000 across three windows, plus the positive control (100) and a follow-on
effort sweep (1,000; section 8).

Fixed parameters, and why they are forced rather than chosen:

- **Effort pinned at `medium` on the 5-family arms.** `thinking: disabled` is
  rejected above effort `high` on Opus 5 (verified live: `evidence/smoke.json`,
  case `opus5-us-disabled-xhigh-must-reject`). A fixed low-enough effort is the
  only way to hold thinking as a clean two-level factor. Haiku 4.5 rejects the
  effort parameter entirely and sends none.
- **Thinking factor restricted to the 5-family.** Haiku 4.5 predates adaptive
  thinking; its thinking form (`budget_tokens`) is a different manipulation, not
  a level of the same factor. Haiku's roles are dated-version anchor and
  positive-control host.
- **max_tokens 16,000 (5-family) / 8,192 (Haiku).** On the 5-family,
  `max_tokens` caps thinking plus response text together; headroom prevents
  truncation registering as false divergence. Any `stop_reason` other than
  `end_turn` is excluded and counted (section 6).

## 3. Endpoints

**Primary endpoint:** per-cell **modal share** — the fraction of valid calls
whose response text is byte-identical to the cell's modal (most frequent)
response. Response text is the concatenation of `text`-type content blocks.
Ties for modal break to the lexicographically smallest candidate
(deterministic). Reported with 95% Wilson score intervals.

**Secondary endpoints:** distinct response count; pairwise agreement
probability; first-divergence character index (and whitespace-token index)
versus the modal response; normalized Levenshtein distance versus modal (exact
via band-doubling below a cap of 512 edits, reported as capped otherwise);
output-token-count variance; thinking-token count mean and variance from
`usage.output_tokens_details.thinking_tokens` (observed live on the 5-family —
the hidden reasoning channel's size is measurable per call even though its
content is not returned).

## 4. Hypotheses and tests

- **Q2 (primary confirmatory).** Per 5-family model, pooling valid calls across
  tasks, thinking arms, and windows: equivalence of exact-match rates between
  `us.` and `global.` arms by TOST with **delta = 1 percentage point**,
  **alpha = 0.05** (two one-sided Wald z tests, unpooled SE; Anscombe-adjusted
  SE only when both observed proportions sit exactly on 0 or 1; implementation
  `analysis/stats.py::two_prop_tost`, unit-tested against hand-derived values).
  Claim on success: "a routing-scope penalty larger than 1pp is ruled out."
  Failure to reject is reported as inconclusive, never as equivalence.
  Per-task and per-window breakdowns are reported as secondary, with the same
  machinery, labeled non-confirmatory.
- **Q3.** Per 5-family model, pooled across tasks, profiles, and windows:
  two-sided 95% CI on the difference in exact-match rate between adaptive and
  disabled arms. This is a difference question, not an equivalence question; no
  TOST. Secondary: association between within-cell thinking-token variance and
  within-cell modal share.
- **Q1, Q4.** Descriptive: per-cell modal shares with Wilson intervals; the
  task-ladder profile by model; per-window pooled rates with intervals.
- Any claim that a rate differs from a chance-level benchmark uses the exact
  binomial test (`analysis/stats.py::binom_test`); none is pre-registered as
  primary.

## 5. Sample size

Pilot: n=20 per cell, single window, before the freeze. The final n per cell is
set from pilot Wilson half-widths (target: half-width at most 5pp at the
observed rates for the least reproducible 5-family cell) and recorded here as
v1.0 before the freeze; n=100 is the default if the pilot supports it. If the
pilot shows delta=1pp is underpowered at feasible n for Q2, the margin may be
widened (recorded here, with reasoning, before the freeze) — never after.

## 6. Validity gates and exclusions

Applied mechanically by `analysis/analyze.py` (unit-tested):

1. **Negative control (harness validity).** All calls in a cell must share one
   request SHA-256 — the hash of exactly the bytes sent. Any mismatch
   invalidates the entire cell and is reported; it means the harness, not
   Bedrock, introduced variance.
2. **Exclusions, counted never silent.** Errored calls (after bounded retries,
   with attempt counts recorded) and calls with `stop_reason != end_turn` are
   excluded per call and reported per cell. A cell with more than 10% of raw
   calls excluded is flagged.
3. **Version-drift control.** The response-body model ID is recorded per call.
   More than one value among a cell's valid calls flags drift; the run is split
   at the boundary and drift is reported as a secondary finding. Known
   limitation, disclosed: the 5-family returns undated IDs (`claude-opus-5`,
   `claude-sonnet-5` — observed live), so an in-place point-version roll is
   undetectable from responses on those models. Mitigations: windows compressed
   to hours; dated Haiku 4.5 in every window as the anchor; AWS request IDs
   recorded on every call so AWS can reconstruct the serving window if a result
   looks anomalous.
4. **Ordering control.** The (cell, repeat) schedule is shuffled with a
   recorded seed; no two calls from the same cell are in flight concurrently;
   per-call jitter of 0.25–1.0 s. This prevents a cell's repeats from riding in
   the same server-side batch and biasing toward artificial agreement.
5. **Cache covariate.** No cache_control is ever sent;
   `cache_read_input_tokens` and `cache_creation_input_tokens` are recorded per
   call and any nonzero cell is flagged and reported as a covariate.
6. **Service tier covariate.** `usage.service_tier` is recorded; a cell serving
   mixed tiers is reported as a covariate.

## 7. Instrument validity (positive control)

Sampling parameters are removed on the 5-family, so the divergence-detection
check runs where sampling still exists: **Haiku 4.5, temperature 0.7, open
generation, n=100, `us.` profile** (accepted live: `evidence/smoke.json`, case
`haiku-us-temp07-control-path`). Fired means at least ceil(n/10) distinct
responses (at least 10 of 100). If it does not fire, the measurement pipeline
cannot detect divergence and no null anywhere in the study is interpretable.

Disclosed limitation: this is a **cross-model** control. It validates that the
instrument detects divergence; it cannot validate that the Opus 5 path would
have diverged under sampling, because no same-model positive control can exist
on a model that rejects sampling parameters.

## 8. Follow-on (exploratory, separately reported)

Effort sweep: Opus 5, classification + open generation, efforts
low/medium/high/xhigh/max, adaptive thinking, `us.` profile, n=100 per cell.
Question: does higher effort stabilize or destabilize the final answer?
Exploratory; no confirmatory test.

## 9. What publishes

A null publishes when all four hold:

1. the positive control fired;
2. the negative control is clean in every analyzed cell;
3. TOST rejects effects larger than delta, so the claim is a bounded one
   ("we can rule out a routing penalty larger than 1pp"), not "we found
   nothing";
4. this pre-registration was frozen, with a commit timestamp earlier than all
   confirmatory data.

A positive result publishes with effect sizes and intervals by model and task.
Version drift observed mid-window publishes as a secondary finding in either
case. A design where only one outcome is interesting is a bad design; both
outcomes here change or validate an architecture decision, including the
default recorded in this firm's own production standards.

## 10. Deviations from the source spec (2026-07-26)

1. **Q2 reshaped** from `us.`-vs-bare-model-ID to `us.`-vs-`global.`:
   single-region on-demand does not exist for these models
   (`evidence/inference-profiles.json`). The spec's instruction for this case
   ("if only one profile form exists, Q2 is unanswerable and should be cut")
   does not apply: two routing forms exist; the factor is real, just different
   from the one drafted.
2. **Thinking factor restricted to the 5-family** (spec assumed a full
   3-model factorial): Haiku 4.5 has no adaptive thinking and rejects effort.
   Grid is 4,000 calls per window, not 4,800.
3. **Positive control task fixed to open generation** (spec said "one task
   type"): maximizes the divergence surface the control must demonstrably
   detect.
4. **max_tokens set per family** (16,000 / 8,192) rather than one global value,
   for the truncation-vs-thinking reason in section 2.

## Freeze checklist (to convert DRAFT to v1.0)

- [ ] Pilot run complete; final n (and delta, if widened) recorded in sections 4–5
- [x] Model access + request-shape assumptions verified live (`evidence/smoke.json`, 10/10)
- [x] Routing profiles verified live (`evidence/inference-profiles.json`)
- [ ] This file bumped to v1.0 and tagged `prereg-v1`
- [ ] Repository public (owner's call) so the freeze timestamp is third-party checkable
- [ ] Confirmatory windows scheduled (PROTOCOL.md)

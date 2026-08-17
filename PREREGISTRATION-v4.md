# Pre-registration v4.0 (FROZEN): Door Attribution on OpenAI Weights

**Status: FROZEN v4.0 — tag `prereg-v4`, pushed before the first
confirmatory call, same discipline as studies 1–3. Frozen 2026-08-17;
owner spend approval recorded the same day.** The 2026-08-17 discovery
session (n=20/cell, four doors, ~250 calls) is **exploratory and forms
no part of the confirmatory dataset**; it is disclosed in section 6 and
its raw records are committed at `9770eb6`
(`evidence/discovery-20260817/`).

Lineage: study 1 (`prereg-v1`, published) measured reproducibility on one
serving plane; study 2 (`prereg-v2`, published as *The Same Model, Three
Doors*) attributed the thinking-mode effect across three Anthropic-model
serving planes; study 3 (`prereg-v3`) established the local control ceiling
on open weights. Study 4 is the vendor-crossing replication: **the same
OpenAI flagship (`gpt-5.6-sol`) reached through five doors spanning two
vendors' infrastructure** — OpenAI first-party, Amazon's OpenAI-compatible
door, Amazon's translated door, Amazon's translated door under global
routing, and OpenAI's own agent harness billed to a consumer subscription.
Studies 1–2 asked whether the stack changes the model's behavior; study 4
asks it across a vendor boundary, where neither party controls the whole
path. All Claude-study comparisons are conceptual replications only.

## 1. Questions

- **Q1 (primary, confirmatory — door attribution of byte-level bias).**
  Discovery found the structured-JSON task emits exactly two byte-variants
  (`349.50` vs `349.5`, all else identical) on every door, with the mix
  differing by door (section 6). Does door membership shift the variant
  distribution at confirmatory n? Endpoint: per-door share of the
  study-wide modal byte-variant on structured JSON at pinned effort
  `none`, Wilson 95% per door; pairwise door contrasts as
  difference-of-proportions with 95% CI, plus Fisher exact p per pair.
  **Registered hypothesis (direction from discovery, disclosed):** the
  1P-vs-codex contrast is nonzero (discovery p=0.019 at n=20). **No
  direction is registered for any other pair** — discovery's n=20 middle
  pairs are individually noise-compatible, and the "raw doors agree,
  wrapped doors drift" reading is a post-hoc label the confirmatory grid
  exists to test, not assume.
- **Q2 (confirmatory — HTTP-door equivalence).** Pairwise equivalence of
  pooled exact-match rates across the four HTTP doors (1P, mantle,
  runtime-us, runtime-global), TOST at **delta = 2pp**, stratified
  variance estimator registered as the primary statistic (equal-weight
  mean of per-stratum differences, per-stratum binomial variances; strata
  = matched (task, effort, window) cells), pooled Wald TOST as labeled
  sensitivity only — the study-2 estimator, unchanged. **The codex door
  is EXCLUDED from Q2 by construction**: it injects a ~13.3K-token agent
  scaffold and is not parameter-identical to any HTTP door. It is a
  registered *harness door*, present in Q1 and Q3 as a labeled fifth arm
  ("what subscribers actually get"), never in equivalence claims.
- **Q3 (confirmatory — the effort analog of the thinking effect).** Study
  2's headline was adaptive thinking costing byte-reproducibility on
  structured JSON. Sol's analog control is `reasoning.effort`. Per door:
  effort `high` minus effort `none` difference in structured-JSON modal
  share with 95% CI; cross-door difference-of-differences with 95% CI.
  Discovery never pinned `high` on structured JSON (default burned zero
  reasoning there), so **no direction is registered as expected; either
  outcome is publishable.**
- **Q4 (exploratory — routing sub-axis).** runtime-us vs runtime-global
  (both invoke-verified 8/17), structured JSON + open generation at
  effort `none`, n=25/cell. Study 1 bounded Claude's routing effect at
  ±2.6pp/90%; this is the OpenAI-weights replication of that bound, at
  reduced n, exploratory.
- **Q5 (exploratory — adaptive-default burn).** Discovery observed the
  omitted-effort default burning 55–475 reasoning tokens across
  byte-identical requests (section 6). Default-effort arm, API doors,
  open generation + structured JSON, n=25/cell: report per-door
  reasoning-token dispersion (min/median/max, IQR) and the exploratory
  association between per-call reasoning spend and byte-divergence from
  the cell's modal output. Descriptive only; no hypothesis registered.

## 2. Design (confirmatory grid)

| Factor | Levels | Notes |
|---|---|---|
| Door | 4 confirmatory + 1 sub-axis | 1P (`api.openai.com/v1/responses`, `gpt-5.6-sol`) · mantle (`bedrock-mantle.us-east-1.api.aws/openai/v1/responses`, `openai.gpt-5.6-sol`, Bearer) · runtime-us (`us.openai.gpt-5.6-sol`, Converse, SigV4) · codex-sub (codex exec 0.147.0, ChatGPT Plus, harness door) — runtime-global in Q4 cells only |
| Model | 1 | `gpt-5.6-sol` only. Terra/Luna are OUT of study 4 (scope decision at draft: one model, five doors is the clean claim; a multi-model grid triples spend without touching the attribution question) |
| Task | 4 | The frozen study-1 ladder, byte-identical prompt strings across doors (envelopes differ by construction; section 3) |
| Effort | 2 pinned | `none` · `high`, pinned explicitly on every call (1P/mantle: `reasoning:{effort}`; runtime: `additionalModelRequestFields.reasoning.effort`; codex: `-c model_reasoning_effort`). The adaptive default is NOT a confirmatory arm (run-variable burn, section 6); it appears only in exploratory Q5 |
| Windows | 2 (HTTP doors) · 1 (codex) | peak (15:00–19:00 UTC) + low (07:00–10:00 UTC), study-2 definitions. **codex-sub runs ONE window's worth, batched across subscription rate-windows over several days (section 7) — registered asymmetry, justified by Plus rate limits; the window factor is therefore tested on HTTP doors only** |
| Repeats | 100 per cell | Q4/Q5 arms n=25 |

Grid: 3 HTTP doors × 4 tasks × 2 effort × 2 windows × 100 = **4,800** ·
codex 4 × 2 × 1 × 100 = **800** · Q4 routing 2 doors × 2 tasks × 25 =
**100** · Q5 default 3 doors × 2 tasks × 25 = **150**. Total ≈ **5,850
calls** (+ discovery's 250, disclosed, excluded). Spend estimate at Sol
list prices, output-dominated: **~$90–130** on HTTP doors; codex $0 if
completed before the ~Sep-17 subscription decision (registered deadline
pressure, section 7). Spend approval is a freeze-checklist item.

## 3. Machinery (carried forward from studies 1–3, plus door-specific)

Primary endpoint: per-cell modal share of byte-identical response text,
Wilson 95%; semantic-equality readout alongside for structured JSON (the
trailing-zero variants are semantically equal — the byte/semantic gap IS
a headline readout); exclusion rules, ordering control (seeded shuffle,
no same-cell concurrency, jitter), no silent retries — bounded backoff
with attempts recorded per call (1P showed a 4/15 transient-5xx burst in
discovery; the retry bound is 3 with exponential backoff, and a call
failing all attempts is a counted exclusion, never re-run singly).

**Request builders and the negative control:** 1P and mantle share one
Responses builder; their bodies are **byte-identical except the model
alias** — mantle rejects the bare `gpt-5.6-sol` id (`not_found_error`,
smoke-verified 2026-08-17), so each door carries its own alias and the
difference is exactly one field, invariant-tested (the study-2 pattern
for by-construction body differences). No `store` field is ever sent:
that 1P persists responses server-side by default while mantle has no
such field is disclosed as a door property, not neutralized. Converse
bodies differ by construction (Converse envelope;
`additionalModelRequestFields` carries the effort pin). Wire hashes: on
1P/mantle the harness records SHA-256 of the bytes actually sent
(stdlib HTTP, no SDK — hashed == sent by construction); Converse takes
structured params, so a botocore `before-send` hook captures the exact
serialized bytes (the study-2 SDK-plane pattern; the hook's firing is a
REQUIRED smoke check, since a silent event-name mismatch would void the
control). More than one distinct wire hash within a cell fails the
cell. **codex-sub has no wire control** — the harness cannot see the
bytes codex sends. Its `--json` mode emits no banner (smoke-verified),
so the effort-pin receipt is a registered **per-batch probe**: before
each batch, one plain-mode call per arm whose stderr banner must state
`reasoning effort: <arm>` (and the model); measured calls then run the
identical argv plus `--json` for the usage record. Per-call receipts
are the JSONL usage event and thread id; it is a harness door and its
claims are labeled accordingly.

**Positive control, redefined for this study:** no door accepts sampling
parameters on Sol (all four reject temperature/top_p; no seed anywhere —
discovery, section 6), so study 2's sampling-based control host does not
exist here. The registered instrument check is internal: **open
generation at effort `none` must show divergence within every door**
(discovery: 20/20 distinct on all doors tested). Any door whose
open-generation cell is ≥99% byte-identical at n=100 flags instrument
failure (caching, wrong plumbing) and gates that door's null results.

**Version-drift blindness, disclosed:** no door returns a dated
snapshot ID (`gpt-5.6-sol` / `openai.gpt-5.6-sol` aliases everywhere).
The dated-anchor mitigation of study 2 is structurally unavailable.
Mitigations: compressed windows (v3 rationale), served `model` string +
response IDs + relevant headers recorded per call, cross-window
consistency checks reported. A mid-study silent roll cannot be ruled
out; it is a registered limitation.

**Token accounting:** Responses doors record
`output_tokens_details.reasoning_tokens` and cache fields; Converse
records only aggregate `outputTokens` (reasoning invisible-but-billed —
door property, disclosed); codex records its own usage schema. Reasoning
spend is a recorded covariate on every call where the door exposes it.

## 4. Known door differences to document, not hide

- Envelopes differ by construction: Responses JSON (1P, mantle) vs
  Converse JSON (runtime) vs codex's agent scaffold (~13.3K standing
  input tokens, banner-receipted). Cross-door identity is defined at the
  prompt-string and pinned-parameter level.
- Auth differs: run-scoped API key (1P) vs IAM service-specific Bearer
  credential (mantle) vs SigV4 (runtime) vs ChatGPT OAuth (codex). No
  key is ever committed; 1P/mantle keys live outside the repo and enter
  as process-scoped env.
- Error surfaces differ: 1P returns OpenAI error JSON with guidance;
  Bedrock wraps model-layer errors in `ValidationException` and strips
  guidance text (discovery-receipted). Retry classification is by HTTP
  status on 1P/mantle, by code name on Converse (study-1 semantics),
  by exit code + stderr on codex.
- `store:true` is 1P-only default behavior; confirmatory responses will
  persist in the OpenAI org for its retention period. Disclosed, not
  neutralized (section 3).
- The codex door's context is bounded by a 1,048,576-character exec
  input cap (discovery-verified; openai/codex#33478's ~258K clamp
  REFUTED at 621,804 tokens through one call). No study-4 cell
  approaches either bound; recorded for scope.
- Input-length ladders (study-2 Q4 analog) are OUT of study 4 entirely.

## 5. Freeze checklist (OPEN — every item must close before tag)

- [x] Owner spend approval (~$90–130 HTTP + $0 codex pre-Sep-17) —
      approved 2026-08-17
- [x] Door request builders + invariant tests committed (Responses
      shared builder, byte-parity modulo model alias; Converse
      effort-pin encoding; codex measured/receipt argv pair; no-store
      and no-sampling assertions) — `tests/test_study4.py`, full suite
      green (328)
- [x] Per-door smoke incl. expected-rejection cases (temperature,
      `minimal` with enumerated set, flat `reasoning_effort`), wire-hook
      firing on Converse, mantle bare-alias probe (REJECTED ⇒ one-field
      body difference), codex per-batch effort receipts both arms —
      `evidence/smoke-study4.json`, 19/19 PASS 2026-08-17
- [x] Discovery raw records (2026-08-17, ~250 calls, four doors)
      imported to evidence/discovery-20260817/ and committed
- [x] codex batch plan verified against live Plus rate windows —
      dry batch 40/40 clean at 5.2 s/call (207 s wall), receipts both
      arms, zero rate-limit events across ~106 door calls same evening;
      the trailing-zero variant pair reproduced through the production
      door client (2 distinct texts at pinned `none`) —
      `evidence/codex-dry-batch-20260817.*`, `scripts/codex_dry_batch.py`
- [x] Analysis code for Q1–Q5 estimators committed BEFORE confirmatory
      data (per-door variant shares + Fisher, stratified TOST, effort
      DoD, Q4 routing bound, Q5 dispersion + association) —
      `analysis/analyze_study4.py` + `tests/test_analyze_study4.py`,
      full suite green (335)
- [x] This file bumped to v4.0, tagged `prereg-v4`, pushed before any
      confirmatory call (frozen 2026-08-17)

Pre-confirmatory machinery (may land after the tag; must be committed
and dry-run verified before the first confirmatory call; touches no
registered quantity):

- [x] study-4 runner modes (schedule builder, door dispatch, manifests,
      per-batch codex receipts) with a `--dry-run` schedule verified —
      completed 2026-08-17 post-tag as registered: modes
      `study4-full` (2,400 calls/24 cells per window) · `study4-codex`
      (800/8, driven in registered batches by
      `scripts/run_codex_batches.py`, resume-aware, receipt-gated) ·
      `study4-q4q5` (250/10, exploratory); dry-run manifests + a live
      10-call q4q5 shakedown at `--repeats 1` (10/10 clean, wire hash on
      every HTTP record, analyzer consumed the records end to end) in
      `evidence/study4-runner-shakedown/`; suite green (349)

## 6. Discovery disclosure (exploratory, not evidence — 2026-08-17)

Raw receipts + full narrative: `evidence/discovery-20260817/` (import
pending, section 5). n=20/cell, defaults unless noted. Every
design decision above that discovery informed:

- **Trailing-zero bifurcation:** structured JSON produced exactly two
  byte-variants on all four doors, differing only at `349.50`/`349.5`
  (char 102, 49 output tokens both). Mix: 1P 11/9 · Converse 11/9 ·
  mantle 7/13 · codex 3/17; ends differ (Fisher p=0.019). → Q1 cell,
  hypothesis restricted to the 1P–codex contrast.
- **Effort-default correction (recorded to keep us honest):** single
  probes first suggested 1P's default ≈ its high arm; n=15/door showed
  all API doors' default distributions coincide (medians 172–191) with
  ~4–5× run-to-run spread on identical requests. The single samples
  were tail draws. → default demoted to exploratory Q5; effort PINNED
  everywhere in the confirmatory grid.
- **Parameter posture (all doors):** temperature/top_p rejected
  (`unsupported_parameter`), seed unknown, effort set enumerated by the
  API as `none/low/medium/high/xhigh/max`, `minimal` gone; flat
  `reasoning_effort` rejected on every Responses-derived door; Bedrock
  accepts only nested Responses shapes via
  `additionalModelRequestFields`. → builder specs, section 3.
- **Ladder reproduces on OpenAI weights:** extraction 20/20
  byte-identical on all four doors; open generation 20/20 distinct on
  all doors tested. → internal positive control, section 3.
- **Fence rate 0/20 on every door** — study 1's AWS fence headline does
  not reproduce on OpenAI weights; model-family-specific, not
  door-intrinsic. Contrast paragraph for the paper; no study-4 cell.
- **codex door:** ~13.3K-token scaffold (9,984 cache-read), effort
  defaults to `none` (banner-receipted), `-c model_reasoning_effort`
  override verified, 1MB exec char cap, #33478 clamp refuted, prompt
  travels via stdin above ARG_MAX. → harness-door registration, Q2
  exclusion, batch plan.
- **1P reliability:** 4/15 transient 5xx in one burst (Bedrock doors
  195/195 same evening). → retry bound + counted-exclusion rule.
- **Classification task was NOT piloted** (discovery ran 3 of the 4
  ladder tasks). It enters the confirmatory grid on study-1/2 precedent
  with no discovery prior; disclosed.

## 7. Confirmatory run plan

Order: codex-sub batches FIRST (rate-window-paced, ~40 calls/window,
several days; must complete before the ~Sep-17 subscription decision so
the sub door's data is free regardless of the keep/cancel outcome),
then Q4/Q5 exploratory arms in a control window, then the low window,
then the peak window, each HTTP-door run as a single compressed run.
Analysis runs the committed Q1–Q5 code over confirmatory files only
(discovery excluded). Raw records, manifests, and reports are committed
to this public repository.

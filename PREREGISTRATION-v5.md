# Pre-registration v5 (DRAFT — NOT FROZEN): The Confidence-Signal Study — Is Ask-Twice an Error Detector?

**Status: DRAFT. Nothing in this document is frozen, no confirmatory
call has been made, and zero model calls have been made against the
fixture corpus.** The freeze checklist is section 10; freeze happens
only after (a) the collaborator decision resolves and the corpus freezes
(its own tagged commit), and (b) the pilot's power calculation is
computed for the estimator registered in section 6. Tag at freeze:
`prereg-v5`, pushed before the first confirmatory call, same
third-party-checkable ordering as studies 1–4.

Lineage: studies 1–2 measured frontier reproducibility across serving
planes (semantic reproducibility 100% on identical requests; the
instability that remains is serialization); study 3 established the
local control ceiling and found the one semantic fork under full greedy
control; study 4 replicated the door structure cross-vendor. The series'
routing doctrine ("ask twice, escalate disagreement" — The Routing
Table, section 6) currently rests on ONE datapoint: the "Item Corvid"
phrase forking Haiku (4–9.6% per plane), a local 20B under greedy
control (1/1,400), and sampled cells (~20% object-level). Study 5 asks
whether that advice is an instrument: **does disagreement between
semantically identical, independently worded asks predict which answers
are actually wrong?**

## 1. The design crux (why paraphrase, not resample)

Identical-request resample is dead on frontier doors by the series' own
data: every 5-family/GPT-5.6 door rejects temperature, and greedy
frontier answers are semantically identical at 100%. The subject of
study 5 is therefore the **disagreement generator**:

1. **Paraphrase pair (PRIMARY)** — two corpus-global instruction
   templates, semantically identical asks, independently worded. Works
   on every substrate including deterministic frontier doors; matches
   how an enterprise would deploy ask-twice (fixed prompt variants).
2. **Sampled resample (registered secondary baseline)** — one template
   repeated 5× under sampling where sampling exists (Haiku 4.5 at
   temperature 0.7; local unseeded 0.7). The semantic-entropy prior-art
   analog (Farquhar et al. 2024; Wang et al. self-consistency).
3. **Cross-door pair (registered falsification arm)** — Sonnet 1P vs
   Sonnet Bedrock, identical template and item. The series predicts
   doors move bytes, not meaning: **registered prediction NULL** (no
   detector signal). Derived from the paraphrase calls; zero extra
   spend.
4. **Cross-model pair (exploratory)** — Haiku vs Sonnet on 1P; measures
   model disagreement, not input ambiguity. Derived; labeled
   exploratory throughout.

## 2. Fixture corpus (the answer key)

`fixtures/study5/corpus.json` + `fixtures/study5/PROTOCOL.md`: n=150
catalog/inventory extraction items, shared schema (`item_name`,
`unit_price`, `quantity_in_stock`), 50/50/50 across an ambiguity
gradient assigned at authoring, blind to any model output — `clean` /
`near_tie` (one correct reading, a boundary token invites a specific
misread) / `ambiguous` (two defensible readings; the second recorded in
`acceptable_alternatives`). Five instruction templates `t1`–`t5`, each
naming all three schema keys and the null rule, no scope hints,
mechanically validated (25 tests).

- **Corpus freeze precedes every model call against corpus items** —
  `meta.frozen` flips in a dedicated tagged commit; the manifest of
  every run records the corpus file's sha256 and frozen flag.
- Circularity mitigations (PROTOCOL section 4): independent labeling by
  the collaborator (preferred; outreach sent 2026-08-31, ~2-week
  window), else the held-out natural-corpus question re-opens before
  freeze; gradient-frozen-pre-data always holds. Authored-to-fork items
  forking is not evidence; the registered claims live in the
  ERROR-PREDICTION contrasts, not in whether forks occur.

## 3. Substrates

| Substrate | Plane | Decode | Arms |
|---|---|---|---|
| haiku_1p | Anthropic API, dated id | API defaults; resample adds temperature 0.7 | paraphrase, resample |
| sonnet_1p | Anthropic API | thinking disabled, effort medium (study-2 encoding) | paraphrase |
| sonnet_bedrock | Bedrock `us.` profile | same | paraphrase (cross-door) |
| local_20b_cuda | gpt-oss:20b, RTX 4090, Ollama 0.30.5 | greedy seed 42; resample unseeded 0.7 | paraphrase, resample |
| local_qwen_metal | qwen3.6:35b, Mac Pro | greedy seed 42, think off | paraphrase |

Prompts are always `template + "\n\n" + document`. Request bodies come
from the studies-1–3 canonical builders; `max_tokens`/`num_predict`
512. Local runs single-flight; API runs rely on unique cells (the
engine's same-cell rule serializes the only repeated cells, the
resample arm).

## 4. Scoring (validation first, semantics only)

Per the paper-4 discipline, comparisons happen after validation and
canonicalization, never on bytes:

- Parse: bare JSON object, or JSON inside one markdown fence (counted
  as `fenced`); anything else is a validation failure — counted and
  excluded from every comparison, never scored as disagreement.
- Canonicalization (analysis/analyze_study5.py, frozen with this
  document): `item_name` trim + collapse whitespace, case preserved;
  `unit_price` numeric with $/, stripping; `quantity_in_stock` int
  accepting integral floats/digit strings; null stays null.
- **Strict correctness**: target field equals ground truth.
  **Lenient**: equals ground truth or any recorded alternative. Strict
  is primary; lenient is a registered sensitivity (on `ambiguous`
  items, strict treats the non-primary defensible reading as wrong —
  by construction ~a third of the corpus can produce "errors" that are
  really defensible readings; the lenient analysis and the per-gradient
  strata make that visible rather than hidden).
- Disagreement: canonical inequality on the target field between the
  pair's answers.

## 5. Endpoints and hypotheses

Unit of analysis: corpus item (one row per item per contrast).

- **H1 (primary).** Items where the paraphrase pair disagrees have
  higher error risk than items where it agrees. Endpoint: risk
  difference P(wrong | disagree) − P(wrong | agree), strict scoring,
  primary substrate (section 6 decision rule), pair (t1, t2).
- **Co-primary descriptives** (the numbers the doctrine needs): catch
  rate P(disagree | wrong) and false-alarm rate P(disagree | right),
  with 95% Wilson intervals.
- **H2 (secondary).** Catch rate rises with k (any-disagreement among
  k asks) at declining marginal value: registered k-sets k2=(t1,t2),
  k3=(t1,t2,t3), k5=(t1..t5); the catch-vs-k curve with per-k
  false-alarm rates is the study's cost line.
- **H3 (registered null).** The cross-door pair shows no detector
  signal (prediction: RR ≈ 1; equivalence bounds set at freeze with
  the power calc).
- **H4 (secondary).** The paraphrase-pair signal concentrates in
  `near_tie` and `ambiguous` strata; `clean` items contribute ~no
  disagreement (per-gradient strata reported for every contrast).
- **Baseline comparison (secondary).** On haiku_1p, paraphrase-pair
  catch/false-alarm vs sampled-resample catch/false-alarm on the same
  items (the prior-art baseline vs the enterprise-deployable
  instrument).
- **Exploratory, labeled:** cross-model pair; per-substrate
  replication table; mechanic-level splits from the corpus rationales.

## 6. Estimators and decision rules (the study-1 lesson lives here)

- Primary estimator: risk difference with Newcombe hybrid-Wilson 95%
  CI; significance: Fisher exact, two-sided, alpha .05, on the item ×
  {disagree, agree} × {wrong, right} table. **The pilot power
  calculation is computed for THIS estimator; if the power calc forces
  an estimator change, the change happens BEFORE freeze and this
  section is rewritten, never reinterpreted after.**
- **Primary-substrate decision rule (registered now, resolved at
  freeze):** compute projected power at n=150 from pilot proportions
  for sonnet_1p and haiku_1p; primary = sonnet_1p if projected power
  ≥ 0.80 (the doctrine substrate — judgment work routes to the frontier
  tier), else haiku_1p if ≥ 0.80 (the semantic-risk tier). If neither
  reaches 0.80 at n=150, the primary claim is registered on the pooled
  near_tie+ambiguous strata instead, with the pooling stated at freeze.
  Both substrates are always reported in full either way.
- No interim looks: the pilot informs the power calc and freeze; pilot
  records never enter the confirmatory dataset.
- Multiplicity: H1 carries the confirmatory alpha; H2–H4 are reported
  with CIs, no alpha claims; exploratory analyses are labeled and make
  no inferential claims.

## 7. Exclusions and validity gates

- Transport failures after the engine's bounded retries: excluded,
  counted, reported per substrate.
- Validation failures (section 4): excluded from comparisons, counted;
  **gate: parse-fail rate < 5% per substrate**, else that substrate's
  results are descriptive-only (registered demotion, not a stop).
- An item enters a contrast only with parsed answers on every ask in
  the contrast (missing-any exclusion, counted per contrast).
- **Positive control (sampling-in-effect):** each resample arm must
  produce ≥1 semantic target-field variation across its 5×n calls;
  a resample arm with zero variation anywhere fails the gate and the
  baseline comparison is void (paraphrase endpoints stand).
- **Ordering attestations:** corpus-freeze commit precedes the first
  model call against corpus items; `prereg-v5` tag precedes the first
  confirmatory call; every manifest carries corpus sha256 + schedule
  digest; wire/planned-request sha per call as in studies 2–4.
- Warmup records excluded by control flag (mechanical).

## 8. Schedule and scale

Runner modes (all dry-run verified): `study5-pilot-api` 420 calls /
`study5-full-api` 3,000 / `study5-full-local` cuda 1,501, metal 751.
Pilot = stratified 21 items (7 per gradient). Confirmatory total 5,252
calls ≈ 3,000 API + 2,252 local ($0). Spend: envelope ≤$100 list-price
(owner-approved 2026-08-31); pilot actuals and the projected
confirmatory number are recorded at freeze.

## 9. Reporting

Whatever the outcome, the report includes: the full 2×2s and rates per
contrast with raw counts; strict and lenient side by side; per-gradient
strata; the k-curve with false-alarm costs; the cross-door null result;
parse/transport accounting; and the negative spaces (what a null H1
would mean for the routing-table claim is written into the paper, not
argued away). Raw records, manifests, and reports are committed to the
public repo as in studies 1–4.

## 10. Freeze checklist (every box before `prereg-v5` is tagged)

- [ ] Collaborator decision resolved; corpus labeling path recorded
      (independent labels merged + adjudication log, or the solo
      mitigation resolved by owner decision)
- [ ] Corpus frozen: `meta.frozen: true`, dedicated commit, tagged,
      pushed — BEFORE any pilot call
- [ ] Pilot run (both credential families), records committed,
      labeled exploratory
- [ ] Power calculation for the section-6 estimator from pilot
      variance; primary substrate resolved by the section-6 rule
- [ ] H3 equivalence bounds set; section 6 finalized
- [ ] Spend actuals recorded; owner sign-off on the confirmatory number
- [ ] Any pilot-informed design change written into sections 1–8 with
      a dated note in the deviations ledger below
- [ ] Tag `prereg-v5` pushed; zero confirmatory calls precede it

## 11. Deviations ledger

(Empty at draft. Every post-freeze deviation lands here with date,
scope, and consequence — the study-3/4 pattern.)

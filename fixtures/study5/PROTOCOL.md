# Study 5 fixture corpus — labeling protocol (v0, 2026-08-31)

**Status: DRAFT — nothing frozen.** `corpus.json` `meta.frozen` is `false` and flips only in the
gate-2 freeze commit, after the collaborator decision resolves (see §4). Everything here is
revisable until PREREG v5 freeze; after that, the corpus and this protocol are immutable and the
freeze commit hash is the reference. **2026-09-09:** mitigation 2 (§4) is now built — a held-out
natural arm, §7, with its own file `natural_corpus.json` that freezes in the same commit.

Study context: does disagreement between semantically identical, independently worded asks
predict which answers are actually wrong? The corpus is the answer key that makes "wrong"
measurable. Design record: `determinism-study5-confidence-signal-design-2026-08-31.md`
(private repo notes); prereg will restate everything registered.

## 1. Corpus shape

- **Domain:** catalog/inventory extraction — synthetic listings in the "Item Corvid" family
  (the phrase class that forked Haiku 4.5, budget-tier API defaults, and a local 20B in
  studies 1 and 3). Invented bird-name products only; no real brands, no client-derived shapes.
- **Shared schema, every item:** `item_name` (string) · `unit_price` (number|null) ·
  `quantity_in_stock` (int|null). One schema across the corpus keeps canonicalization and
  validation identical everywhere; `target_field` marks where the item's graded difficulty lives.
- **Planned n = 150** (batch 1 = 30, authored 2026-08-31). Gradient balance goal: ~50/50/50
  across classes, target_field roughly balanced within each class.

## 2. Ambiguity gradient (assigned at authoring, blind to any model output)

| Class | Definition | Ground truth |
|---|---|---|
| `clean` | One reading; a careful reader cannot defend another | The stated values |
| `near_tie` | One CORRECT reading, but a boundary token invites a specific misread (listing prefix, case-vs-unit price, was/now, reserved stock, pack counts) | The correct reading; `rationale` names the distractor and why it is wrong |
| `ambiguous` | Experts could defend two readings; the document genuinely underdetermines the target field | Primary = the reading a careful cataloger would defend as best; every other defensible reading recorded in `acceptable_alternatives`; `rationale` states both |

Analysis consequence (registered at prereg, not here): error can be scored **strict**
(≠ ground_truth) and **lenient** (∉ {ground_truth} ∪ alternatives); the corpus carries both labels
so the frozen estimator can pick either and the other runs as sensitivity.

## 3. Disagreement generator: corpus-global instruction templates

Five templates (`t1`–`t5` in `corpus.json` meta), each a complete ask for the same three fields
with the same null rule, independently worded.

**Why global templates rather than per-item paraphrases:** (1) no phrasing can be tuned to a
specific item's ambiguity — the template authors never see the item when the pair is chosen,
killing the fixture-side circularity channel at the instruction level; (2) the k-ladder
(k = 2, 3, 5) needs k registered asks — pairs/sets are drawn from `t1`–`t5` at prereg;
(3) it matches deployment reality: an enterprise implements ask-twice as fixed prompt variants,
not per-request rewording.

Template discipline (mechanically enforced in `harness/study5_fixtures.py` +
`tests/test_study5_fixtures.py`): every template names all three schema keys and the null rule;
no template carries scope hints (`current`, `total`, `combined`, `per unit`, `on-hand`) that would
disambiguate what the documents leave open; templates are pairwise distinct; documents never
contain instruction text.

## 4. Circularity hazard and the labeling path (the honest part)

Authored fixtures authored to fork WILL fork — that alone proves nothing. Standing mitigations,
at least one required before freeze:

1. **Independent labeler** — the preferred path. Collaborator outreach (IBM Financial Services
   Market pair, contacted 2026-08-31, ~2-week window): if they engage, they independently
   re-label ground truth + gradient for every item (and may contribute a finance-calibrated
   second domain). Disagreements between labelers are adjudicated and disclosed, and items
   with unresolved labels are dropped before freeze.
2. **Held-out naturally-occurring documents** — if the collaborator path fails, this question
   RE-OPENS before freeze (recorded owner: Joe). Frozen gradient assignment alone is the
   weakest acceptable position and must not be the study's only mitigation by silent default.
   **Resolved 2026-09-09 (owner decision, collaborator unresolved at day 9): built as §7,
   collaborator-independent — if the collaborator engages, their labels layer on top.**
3. **Gradient frozen pre-data** — always on: `meta.frozen` flips in a dedicated commit, tagged,
   before any pilot call; the freeze commit precedes the first model output on these items,
   third-party-checkable in the public history.

Batch 1 disclosure: single author (the session), documents + ground truth + gradient + templates
all authored 2026-08-31 with zero model calls made against them. The author had read studies 1–4,
so near-tie classes deliberately extend the empirically observed fork family — that is the design,
and it is why mitigation 1 or 2 is required rather than optional.

## 5. Authoring rules (for every future batch)

- Ground truth + gradient + rationale written at the moment the document is written, before any
  model ever sees the item. Never revise a graded item after model output exists for it —
  drop and replace with a new id instead.
- Documents ≤1500 chars, unique, instruction-free (no "JSON", no imperative ask verbs);
  prices/quantities realistic; nulls used where the document is genuinely silent.
- `near_tie` requires exactly one defensible reading (the distractor must be a misread, not an
  alternative interpretation); `ambiguous` requires ≥1 recorded alternative that a careful expert
  could defend. When authoring blurs that line, the item is `ambiguous`.
- Run `python3 -m unittest tests.test_study5_fixtures` after every batch; the validator is the
  protocol's mechanical half and must stay green.

## 6. Freeze checklist (gate 2 exit — do not check any box early)

- [ ] Collaborator decision resolved (in with labels received, or out with mitigation 2 resolved by Joe)
- [ ] Full n authored, gradient balance recorded
- [ ] Independent labels merged + adjudication log committed (if path 1)
- [ ] Validator green on the full corpus
- [ ] Natural arm (§7): second-labeler agreement recorded in `natural_corpus.json`
      `meta.labeling.agreement`, adjudication noted, any excluded ids listed by date
- [ ] Natural arm (§7): `python3 -m harness.study5_natural check` CLEAN and
      `validate_natural_corpus` clean at the freeze commit
- [ ] `meta.frozen: true` in BOTH `corpus.json` and `natural_corpus.json`, one dedicated
      commit, tagged, pushed BEFORE any pilot call

## 7. Held-out natural arm (mitigation 2 — added 2026-09-09)

**Why:** §4 requires a mitigation beyond gradient-frozen-pre-data. With the collaborator path
unresolved nine days after outreach, the owner chose to build mitigation 2 now rather than freeze
solo without it. The arm is collaborator-independent: if the collaborator engages, their
independent labels apply to both corpora.

**What it is:** 50 inventory records nobody on the study wrote, drawn by a fixed procedure from
two public-domain government feeds, rendered verbatim, labeled under the conventions below, and
asked the SAME five instruction templates as the authored corpus (byte-identical, validator-
enforced). File: `natural_corpus.json`; code: `harness/study5_natural.py`; provenance and
licenses: `natural/SOURCES.md`; snapshots: `natural/snapshots/` (sha256 in the corpus meta).

### 7.1 Sources and snapshots

| Source | Feed | Snapshot | Rows | Eligible | Drawn |
|---|---|---|---|---|---|
| City of Austin, TX — Arterial Management Materials Warehouse Inventory | `data.austintexas.gov/hcaw-evi2` | latest `published_date` 2026-09-08T23:13:18 | 381 | 226 | 25 |
| Montgomery County, MD — ABS Store Inventory and Sale Items | `data.montgomerycountymd.gov/ib5t-5ncy` | live view, fetched 2026-09-09 | 7,037 | 2,768 | 25 |

Both feeds are public domain (quotes and URLs in `natural/SOURCES.md`). The snapshots are
committed so the draw is re-derivable by anyone; the feeds themselves update daily.

### 7.2 Procedure (fixed in code BEFORE the draw; nothing hand-picked)

- **Eligibility** (mechanical): Austin — `financial_name` non-empty, `unit_cost` present and
  > 0, `total_on_hand` present, integer, ≥ 0. Montgomery — `description` non-empty,
  `price` > 0, `totalinventory` present, integer, > 0 (currently stocked items).
- **Draw:** `random.Random(int(sha256(snapshot)[:16], 16)).sample(eligible sorted by row key,
  25)` per source — the seed is the snapshot's own hash, so there is no seed to choose.
  Items are numbered `s5n-001…050` in draw order, Austin first.
- **Rendering:** the feed's own column display names, `Label: value`, one field per line, in
  the feed's column order; values verbatim (outer whitespace trimmed); record-keeping columns
  excluded (Austin: `id`, `published_date`, `modified_date`). No rewording, nothing added.
- **Reproducibility:** `python3 -m harness.study5_natural check` re-derives the draw and every
  document from the committed snapshots and reports any divergence; the test suite runs it.

### 7.3 Labeling conventions (declared; applied to the rendered documents; zero model calls)

- Same schema and gradient definitions as §1–§2. Every item records a class **per field** in
  `field_classes`; the item's `gradient` is the highest class present; `target_field` is the
  highest-class field, ties rotated through schema order by item number (validator-enforced —
  the labeler chooses classes with rationales, never the target). `acceptable_alternatives`
  carries the target field's alternatives (§2 rule); alternatives on other fields are recorded
  in `other_field_alternatives` for disclosure and the second-labeler comparison.
- **Austin.** `item_name`: when Financial Name and Common Name differ → `ambiguous`, both
  recorded; **Common Name is primary** (adjudicated 2026-09-10, §7.4) unless the Common Name is
  a quoted sentence rather than a name (s5n-005), where Financial Name stays primary.
  `unit_price`: Unit Cost, exact figure, `clean` regardless of decimal places (adjudicated
  2026-09-10: rounding is a transcription risk, not a boundary-token misread; the rationale
  keeps the note; Total Value is the extended-value distractor). `quantity_in_stock`: Total On
  Hand as stated → `clean` (zero is stated, not silent); a pack-count token in a name against
  `EA` (e.g., "100 per box", 15 on hand) → `near_tie`, the multiplied count being the invited
  misread.
- **Montgomery.** `item_name`: Description verbatim is primary; size/pack tokens duplicated by
  the Size field → `ambiguous`, stripped reading(s) recorded, **and `null` recorded as an
  accepted alternative** (adjudicated 2026-09-10: the second labeler read these records as
  stating no item_name field — Description/Code/Category only). `unit_price`: Price →
  `clean`; Sale Price present → `ambiguous` (Price primary). `quantity_in_stock`: Total
  Inventory as stated → `clean`; pack SKUs whose inventory is a multiple of the pack count →
  `ambiguous` (per-pack and per-case readings recorded).
- **Realized distribution after adjudication (recorded, never re-balanced by selection):**
  ambiguous 32 / near_tie 0 / clean 18; target fields item_name 38 / quantity_in_stock 7 /
  unit_price 5; per source Austin 7 ambiguous / 18 clean, Montgomery 25 ambiguous. (Before
  adjudication: 32 / 11 / 7 — the 11 near_tie items were the decimal-cost convention, decision
  C.) The natural arm therefore has NO near_tie stratum; H4's near_tie contrast is a primary-
  corpus result only. Reported per stratum and per source (analyzer `by_source`). The
  Montgomery item_name mechanic is one mechanic on 25 items — a known concentration,
  disclosed, not a finding.

### 7.4 Second labeler (partial independence check)

The owner labels 20 items (the first 10 per source in draw order) from
`natural/SECOND-LABELER-SHEET.md` before reading §7.3 or the corpus file. Agreement — exact
match per field after the registered canonicalization, plus class agreement — is recorded in
`meta.labeling.agreement` at freeze; disagreements are adjudicated and disclosed. An item whose
label cannot be adjudicated is EXCLUDED by id in a dated note (the draw is never redrawn).
This is a partial check only: the first labeler is the same session labeler as batches 1–4, and
that is disclosed.

**Outcome (2026-09-10, raw → adjudicated; files `natural/agreement-raw-2026-09-10.json` and
`natural/agreement-post-adjudication-2026-09-10.json`; decisions verbatim in
`natural_corpus.json` `meta.labeling.adjudication`):** unit_price 20/20 strict; quantity_in_stock
18/18 strict (2 left blank); item_name 7/20 strict and 10/20 within the recorded readings before
adjudication → 10/20 strict and 20/20 within after; class 13/20 (κ 0.37) before → 17/20 (κ 0.71
three-class, 1.00 clean-vs-not) after. The disagreements were three systematic conventions, not
noise: (A) Montgomery item_name — the second labeler read `null` (no field named "name"), the
first read Description verbatim; both classed ambiguous ⇒ Description stays primary, `null` is
an accepted alternative on all 25. (B) Austin dual names — the second labeler chose Common Name
on every plain case ⇒ Common Name primary on 6 items, Financial Name on the quoted-sentence
case, class ambiguous on all 7. (C) decimal costs — the second labeler classed them clean ⇒
reclassed clean on 15 items (gradient/target re-derived on 11). Remaining: s5n-001..003 classed
near_tie by the second labeler vs ambiguous (owner kept ambiguous). No item excluded.

### 7.5 Role in the design

A registered replication arm (PREREGISTRATION-v5 §5, H5), not the powered primary. Substrates:
`haiku_1p`, `sonnet_1p` (paraphrase; haiku resample), `local_20b_cuda`, `local_qwen_metal`.
Calls: 750 API + 501 cuda + 251 metal. Confirmatory only — no pilot, no item reused anywhere —
run after the `prereg-v5` tag via `scripts/study5_api_run.sh natural` and
`scripts/study5_cuda_window.sh natural` (both refuse on an unfrozen natural corpus and without
the tag). `natural_corpus.json` `meta.frozen` flips in the same freeze commit as `corpus.json`.

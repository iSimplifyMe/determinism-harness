# Study 5 fixture corpus — labeling protocol (v0, 2026-08-31)

**Status: DRAFT — nothing frozen.** `corpus.json` `meta.frozen` is `false` and flips only in the
gate-2 freeze commit, after the collaborator decision resolves (see §4). Everything here is
revisable until PREREG v5 freeze; after that, the corpus and this protocol are immutable and the
freeze commit hash is the reference.

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
- [ ] `meta.frozen: true` in a dedicated commit, tagged, pushed BEFORE any pilot call

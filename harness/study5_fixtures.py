"""Study-5 fixture corpus: loading and protocol validation.

The corpus (fixtures/study5/corpus.json) is authored under
fixtures/study5/PROTOCOL.md — ground truth and an ambiguity-gradient class
assigned at authoring, blind to any model output. The disagreement
generator is a set of corpus-global instruction templates (semantically
identical asks, independently worded); they are global rather than
per-item so no phrasing can be tuned to an item's ambiguity, and so the
k-ladder has k registered asks.

This module is the single reader the runner, analyzer, and tests all
share, and `validate_corpus` is the mechanical enforcement of every
protocol property a program can check. Stdlib only, like the rest of the
harness.
"""
import json
from pathlib import Path

SCHEMA_KEYS = ("item_name", "unit_price", "quantity_in_stock")
GRADIENTS = ("clean", "near_tie", "ambiguous")
TEMPLATE_IDS = ("t1", "t2", "t3", "t4", "t5")

CORPUS_PATH = Path(__file__).resolve().parent.parent / "fixtures/study5/corpus.json"

_ITEM_FIELDS = (
    "id",
    "gradient",
    "target_field",
    "document",
    "ground_truth",
    "acceptable_alternatives",
    "rationale",
)

# Documents are catalog snippets, never prompts: an ask embedded in the
# document would make the instruction-template contrast cosmetic.
_INSTRUCTION_MARKERS = ("json", "return a", "extract the")

# Templates must not disambiguate what the documents leave open — the
# ambiguity under test lives in the documents alone.
_SCOPE_HINTS = ("current", "total", "combined", "per unit", "on-hand")


def load_corpus(path=None):
    with open(path or CORPUS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _check_templates(meta, errors):
    templates = meta.get("instruction_templates")
    if not isinstance(templates, dict):
        errors.append("meta.instruction_templates missing")
        return
    if tuple(sorted(templates)) != TEMPLATE_IDS:
        errors.append(
            f"template ids {sorted(templates)} != registered {list(TEMPLATE_IDS)}"
        )
    texts = set()
    for template_id, text in templates.items():
        if not isinstance(text, str) or not text.strip():
            errors.append(f"template {template_id}: empty")
            continue
        for key in SCHEMA_KEYS:
            if key not in text:
                errors.append(f"template {template_id}: omits key {key}")
        if "null" not in text.lower():
            errors.append(f"template {template_id}: omits the null rule")
        lowered = text.lower()
        for hint in _SCOPE_HINTS:
            if hint in lowered:
                errors.append(f"template {template_id}: scope hint {hint!r}")
        stripped = text.strip()
        if stripped in texts:
            errors.append(f"template {template_id}: duplicate text")
        texts.add(stripped)


def _check_ground_truth(item, errors):
    gt = item.get("ground_truth")
    if not isinstance(gt, dict) or set(gt) != set(SCHEMA_KEYS):
        errors.append(f"{item.get('id')}: ground_truth keys != schema keys")
        return
    if not isinstance(gt["item_name"], str) or not gt["item_name"].strip():
        errors.append(f"{item['id']}: ground_truth item_name not a nonempty str")
    price = gt["unit_price"]
    if price is not None and not isinstance(price, (int, float)):
        errors.append(f"{item['id']}: ground_truth unit_price not number|null")
    quantity = gt["quantity_in_stock"]
    if quantity is not None and not (
        isinstance(quantity, int) and not isinstance(quantity, bool)
    ):
        errors.append(f"{item['id']}: ground_truth quantity_in_stock not int|null")


def _check_alternatives(item, errors):
    alts = item.get("acceptable_alternatives")
    if not isinstance(alts, list):
        errors.append(f"{item.get('id')}: acceptable_alternatives not a list")
        return
    if item.get("gradient") == "ambiguous":
        if not alts:
            errors.append(
                f"{item['id']}: ambiguous item needs >=1 acceptable_alternative"
            )
        for alt in alts:
            target = item.get("target_field")
            if not isinstance(alt, dict) or target not in alt:
                errors.append(
                    f"{item['id']}: alternative missing target_field {target}"
                )
            elif alt[target] == item["ground_truth"].get(target):
                errors.append(
                    f"{item['id']}: alternative equals ground truth on {target}"
                )
    elif alts:
        errors.append(
            f"{item['id']}: acceptable_alternatives only allowed on ambiguous"
        )


def validate_corpus(corpus):
    """Return a list of human-readable protocol violations (empty = clean)."""
    errors = []
    meta = corpus.get("meta", {})
    if meta.get("study") != "study5-confidence-signal":
        errors.append("meta.study missing or wrong")
    if not isinstance(meta.get("frozen"), bool):
        errors.append("meta.frozen must be a bool")
    _check_templates(meta, errors)

    items = corpus.get("items", [])
    seen_ids, seen_docs = set(), set()
    for item in items:
        item_id = item.get("id", "<no id>")
        missing = [f for f in _ITEM_FIELDS if f not in item]
        if missing:
            errors.append(f"{item_id}: missing fields {missing}")
            continue
        if item_id in seen_ids:
            errors.append(f"{item_id}: duplicate id")
        seen_ids.add(item_id)

        if item["gradient"] not in GRADIENTS:
            errors.append(f"{item_id}: gradient {item['gradient']!r} invalid")
        if item["target_field"] not in SCHEMA_KEYS:
            errors.append(f"{item_id}: target_field not a schema key")

        doc = item["document"]
        if not isinstance(doc, str) or not doc.strip():
            errors.append(f"{item_id}: document empty")
        else:
            if len(doc) > 1500:
                errors.append(f"{item_id}: document over 1500 chars")
            if doc in seen_docs:
                errors.append(f"{item_id}: duplicate document")
            seen_docs.add(doc)
            lowered = doc.lower()
            for marker in _INSTRUCTION_MARKERS:
                if marker in lowered:
                    errors.append(
                        f"{item_id}: document contains instruction marker "
                        f"{marker!r}"
                    )

        _check_ground_truth(item, errors)
        _check_alternatives(item, errors)

        if item["gradient"] != "clean" and not str(item["rationale"]).strip():
            errors.append(f"{item_id}: rationale required off clean")
    return errors

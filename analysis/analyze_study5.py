"""Study-5 analyzer: does disagreement predict error?

Consumes study-5 run records (meta: substrate/arm/item_id/template_id)
plus the fixture corpus, and computes the design's endpoints:

- validation first (paper-4 discipline): a response must parse to the
  registered schema before it enters any comparison; parse failures are
  counted and excluded, never scored as disagreement.
- disagreement is SEMANTIC, on canonicalized field values — never bytes.
- correctness against ground truth, scored two ways: strict
  (target field == ground truth) and lenient (== ground truth or any
  recorded acceptable_alternative).
- per generator pair: catch rate P(disagree | wrong), false-alarm rate
  P(disagree | right), and the relative error risk of disagreeing vs
  agreeing items. k-sets generalize pairs to "any disagreement among k
  asks".

Point estimates and raw counts only: interval estimators are registered
at prereg v5 with the power calc (study-1 lesson — never bolt an
estimator on after the fact). Stdlib only.
"""
import json
import re
from collections import defaultdict

FENCE_RE = re.compile(r"\A\s*```(?:json)?\s*\n(.*?)\n?```\s*\Z", re.DOTALL)

SCHEMA_KEYS = ("item_name", "unit_price", "quantity_in_stock")


def parse_response(text):
    """Return (mode, dict) — mode 'bare', 'fenced', or 'fail'.

    Same validation stance as the published semantic addendum
    (analysis/semantic_sj.py): a fence is a formatting variant the
    validator strips; anything else non-JSON is a validation failure."""
    if not isinstance(text, str):
        return "fail", None
    candidate = text.strip()
    mode = "bare"
    match = FENCE_RE.match(text)
    if match:
        candidate = match.group(1).strip()
        mode = "fenced"
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return "fail", None
    if not isinstance(obj, dict):
        return "fail", None
    return mode, obj


_WS_RE = re.compile(r"\s+")
_PRICE_STRIP_RE = re.compile(r"[$,\s]")


def canonical_field(field, value):
    """Registered canonicalization (v0, revisable until prereg freeze):

    - item_name: trim + collapse internal whitespace; case preserved
      (models copy names from the document; case edits are content).
    - unit_price: number, or numeric string after stripping $ , and
      whitespace; integral floats compare equal to ints.
    - quantity_in_stock: int, accepting integral floats and digit strings.
    - null/None stays None everywhere.
    Unconvertible values return ("invalid", raw) so they compare unequal
    to everything except an identical raw value.
    """
    if value is None:
        return None
    if field == "item_name":
        if not isinstance(value, str):
            return ("invalid", repr(value))
        return _WS_RE.sub(" ", value).strip()
    if field == "unit_price":
        if isinstance(value, bool):
            return ("invalid", repr(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = _PRICE_STRIP_RE.sub("", value)
            try:
                return float(stripped)
            except ValueError:
                return ("invalid", value)
        return ("invalid", repr(value))
    if field == "quantity_in_stock":
        if isinstance(value, bool):
            return ("invalid", repr(value))
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else ("invalid", repr(value))
        if isinstance(value, str):
            stripped = value.strip().replace(",", "")
            try:
                return int(stripped)
            except ValueError:
                return ("invalid", value)
        return ("invalid", repr(value))
    raise ValueError(f"unknown field {field}")


def answer_correct(corpus_item, answer_obj, lenient=False):
    """Correctness on the item's target field."""
    field = corpus_item["target_field"]
    given = canonical_field(field, answer_obj.get(field))
    truth = canonical_field(field, corpus_item["ground_truth"][field])
    if given == truth:
        return True
    if lenient:
        for alt in corpus_item.get("acceptable_alternatives", []):
            if field in alt and given == canonical_field(field, alt[field]):
                return True
    return False


def fields_disagree(obj_a, obj_b, field):
    return canonical_field(field, obj_a.get(field)) != canonical_field(
        field, obj_b.get(field)
    )


def index_records(records):
    """(substrate, item_id, template_id) -> [parsed dicts]; plus parse
    accounting. Warmup/control records are excluded here."""
    parsed = defaultdict(list)
    counts = {"in": 0, "excluded_control": 0, "parse_fail": 0, "fenced": 0}
    for record in records:
        meta = record.get("meta", {})
        if meta.get("control"):
            counts["excluded_control"] += 1
            continue
        counts["in"] += 1
        mode, obj = parse_response(record.get("response_text"))
        if mode == "fail":
            counts["parse_fail"] += 1
            continue
        if mode == "fenced":
            counts["fenced"] += 1
        key = (meta["substrate"], meta["item_id"], meta["template_id"])
        parsed[key].append(obj)
    return parsed, counts


def _rates(rows):
    """rows: list of (disagree: bool, wrong: bool). Returns the endpoint
    block with raw counts alongside every rate."""
    n = len(rows)
    wrong = [r for r in rows if r[1]]
    right = [r for r in rows if not r[1]]
    disagree = [r for r in rows if r[0]]
    agree = [r for r in rows if not r[0]]
    caught = sum(1 for r in wrong if r[0])
    false_alarms = sum(1 for r in right if r[0])
    wrong_in_disagree = sum(1 for r in disagree if r[1])
    wrong_in_agree = sum(1 for r in agree if r[1])
    p_wrong_disagree = wrong_in_disagree / len(disagree) if disagree else None
    p_wrong_agree = wrong_in_agree / len(agree) if agree else None
    if p_wrong_disagree is None or p_wrong_agree in (None, 0):
        relative_risk = None
    else:
        relative_risk = p_wrong_disagree / p_wrong_agree
    return {
        "n_items": n,
        "n_wrong": len(wrong),
        "n_disagree": len(disagree),
        "catch_rate": (caught / len(wrong)) if wrong else None,
        "caught": caught,
        "false_alarm_rate": (false_alarms / len(right)) if right else None,
        "false_alarms": false_alarms,
        "p_wrong_given_disagree": p_wrong_disagree,
        "p_wrong_given_agree": p_wrong_agree,
        "relative_risk": relative_risk,
    }


def _first_answer(parsed, substrate, item_id, template_id):
    answers = parsed.get((substrate, item_id, template_id), [])
    return answers[0] if answers else None


def kset_analysis(parsed, corpus_items, substrate, template_ids,
                  lenient=False, reference_template=None):
    """Disagreement-any among the k templates vs correctness of the
    reference answer (default: first template in the set). Items missing
    any template's parsed answer are excluded (counted)."""
    reference_template = reference_template or template_ids[0]
    rows = []
    excluded = 0
    per_gradient = defaultdict(list)
    for item in corpus_items:
        answers = [
            _first_answer(parsed, substrate, item["id"], t)
            for t in template_ids
        ]
        if any(a is None for a in answers):
            excluded += 1
            continue
        field = item["target_field"]
        reference = _first_answer(
            parsed, substrate, item["id"], reference_template
        )
        disagree = any(
            fields_disagree(answers[0], other, field) for other in answers[1:]
        )
        wrong = not answer_correct(item, reference, lenient=lenient)
        rows.append((disagree, wrong))
        per_gradient[item["gradient"]].append((disagree, wrong))
    out = _rates(rows)
    out["k"] = len(template_ids)
    out["templates"] = list(template_ids)
    out["reference_template"] = reference_template
    out["excluded_missing"] = excluded
    out["by_gradient"] = {g: _rates(r) for g, r in sorted(per_gradient.items())}
    return out


def cross_pair_analysis(parsed, corpus_items, substrate_a, substrate_b,
                        template_id, lenient=False):
    """Cross-door / cross-model generator: same template, two substrates.
    Correctness is scored on substrate_a's answer."""
    rows = []
    excluded = 0
    for item in corpus_items:
        a = _first_answer(parsed, substrate_a, item["id"], template_id)
        b = _first_answer(parsed, substrate_b, item["id"], template_id)
        if a is None or b is None:
            excluded += 1
            continue
        field = item["target_field"]
        rows.append((
            fields_disagree(a, b, field),
            not answer_correct(item, a, lenient=lenient),
        ))
    out = _rates(rows)
    out["substrates"] = [substrate_a, substrate_b]
    out["template"] = template_id
    out["excluded_missing"] = excluded
    return out


def resample_analysis(parsed_resample, corpus_items, substrate,
                      template_id, lenient=False):
    """Resample generator: R sampled answers per item; disagreement = any
    semantic difference among them; correctness scored on the first."""
    rows = []
    excluded = 0
    for item in corpus_items:
        answers = parsed_resample.get((substrate, item["id"], template_id), [])
        if len(answers) < 2:
            excluded += 1
            continue
        field = item["target_field"]
        disagree = any(
            fields_disagree(answers[0], other, field)
            for other in answers[1:]
        )
        rows.append((
            disagree,
            not answer_correct(item, answers[0], lenient=lenient),
        ))
    out = _rates(rows)
    out["substrate"] = substrate
    out["template"] = template_id
    out["excluded_missing"] = excluded
    return out

"""Second-labeler agreement for the study-5 natural arm (PROTOCOL 7.4).

The owner labels 20 natural-arm items from natural/SECOND-LABELER-SHEET.md
without reading the first labeler's conventions or the corpus file. This
module parses the filled sheet, compares it with the committed labels, and
reports agreement three ways per field:

- strict: the second labeler's primary reading equals the first labeler's
  ground truth after the registered canonicalization (analysis.canonical_field);
- within: the second labeler's primary reading is one of the readings the
  first labeler recorded (ground truth + acceptable_alternatives on the
  target field + other_field_alternatives elsewhere);
- covered: the first labeler's ground truth appears among ALL readings the
  second labeler listed (an ambiguous item lists several, primary first).

Plus class agreement (clean / near_tie / ambiguous) with Cohen's kappa. The
report is a disclosure, not a gate: disagreements are adjudicated by hand
and the outcome written into the corpus meta at freeze (--write records the
computed summary there; it never touches an item). Stdlib only.

Sheet grammar (one block per item):

    ## s5n-012
    ```
    <document>
    ```
    - item_name: <reading>[ | <reading>...]
    - unit_price: <number>[ | <number>...]
    - quantity_in_stock: <integer>[ | <integer>...]
    - class: clean | near_tie | ambiguous
    - note: <free text>

`null` (any case) is the null reading; an empty value = unlabeled.
"""
import argparse
import json
import re
from datetime import datetime, timezone

from analysis.analyze_study5 import canonical_field
from harness.study5_fixtures import GRADIENTS, NATURAL_CORPUS_PATH, SCHEMA_KEYS, load_corpus
from harness.study5_natural import NATURAL_DIR

SHEET_PATH = NATURAL_DIR / "SECOND-LABELER-SHEET.md"
ALT_SEP = " | "
SHEET_FIELDS = SCHEMA_KEYS + ("class", "note")

_SECTION_RE = re.compile(r"^## (s5n-\d{3})\s*$", re.MULTILINE)
_LINE_RE = re.compile(
    r"^- (item_name|unit_price|quantity_in_stock|class|note):[ \t]*(.*?)[ \t]*$",
    re.MULTILINE,
)


def _readings(raw):
    """Split a sheet value into readings (primary first); [] = unlabeled."""
    raw = raw.strip()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(ALT_SEP)]
    out = []
    for part in parts:
        if not part:
            continue
        out.append(None if part.lower() in ("null", "none") else part)
    return out


def parse_sheet(text):
    """{item_id: {field: [readings], 'class': str|None, 'note': str}} in
    sheet order. A block without a field line has that field unlabeled."""
    blocks = {}
    matches = list(_SECTION_RE.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        # Field lines live after the fenced document; a document line that
        # happens to start with "- item_name:" is impossible (documents are
        # "Label: value" lines), but strip the fence to be safe.
        fence_end = body.rfind("```")
        tail = body[fence_end + 3:] if fence_end != -1 else body
        entry = {field: [] for field in SCHEMA_KEYS}
        entry["class"] = None
        entry["note"] = ""
        for line in _LINE_RE.finditer(tail):
            field, value = line.group(1), line.group(2)
            if field == "class":
                value = value.strip().lower().replace("-", "_").replace(" ", "_")
                entry["class"] = value or None
            elif field == "note":
                entry["note"] = value.strip()
            else:
                entry[field] = _readings(value)
        item_id = match.group(1)
        if item_id in blocks:
            raise ValueError(f"duplicate sheet block {item_id}")
        blocks[item_id] = entry
    return blocks


def _canon(field, value):
    return canonical_field(field, value)


def _recorded_readings(item, field):
    readings = [_canon(field, item["ground_truth"][field])]
    if field == item["target_field"]:
        for alt in item.get("acceptable_alternatives", []):
            if field in alt:
                readings.append(_canon(field, alt[field]))
    for value in item.get("other_field_alternatives", {}).get(field, []):
        readings.append(_canon(field, value))
    return readings


def cohen_kappa(pairs):
    """Unweighted Cohen's kappa over (a, b) label pairs; None if undefined."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = 0.0
    for label in labels:
        pa = sum(1 for a, _ in pairs if a == label) / n
        pb = sum(1 for _, b in pairs if b == label) / n
        expected += pa * pb
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def compare(corpus, sheet):
    """Agreement summary between the corpus labels (first labeler) and a
    parsed sheet (second labeler)."""
    by_id = {item["id"]: item for item in corpus["items"]}
    unknown = [item_id for item_id in sheet if item_id not in by_id]
    if unknown:
        raise ValueError(f"sheet ids not in corpus: {unknown}")
    fields = {
        field: {"n": 0, "strict": 0, "within": 0, "covered": 0, "unlabeled": 0}
        for field in SCHEMA_KEYS
    }
    disagreements = []
    class_pairs = []
    class_disagreements = []
    class_unlabeled = 0
    items_labeled = 0
    for item_id, entry in sheet.items():
        item = by_id[item_id]
        any_field = False
        for field in SCHEMA_KEYS:
            theirs = entry[field]
            if not theirs:
                fields[field]["unlabeled"] += 1
                continue
            any_field = True
            fields[field]["n"] += 1
            recorded = _recorded_readings(item, field)
            ours = recorded[0]
            theirs_canon = [_canon(field, value) for value in theirs]
            strict = theirs_canon[0] == ours
            within = theirs_canon[0] in recorded
            covered = ours in theirs_canon
            fields[field]["strict"] += strict
            fields[field]["within"] += within
            fields[field]["covered"] += covered
            if not strict:
                disagreements.append({
                    "id": item_id,
                    "field": field,
                    "first_labeler": item["ground_truth"][field],
                    "second_labeler": theirs,
                    "within_recorded": within,
                    "covered": covered,
                    "target_field": field == item["target_field"],
                })
        theirs_class = entry["class"]
        if theirs_class is None:
            class_unlabeled += 1
        else:
            any_field = True
            if theirs_class not in GRADIENTS:
                raise ValueError(f"{item_id}: class {theirs_class!r} not in {GRADIENTS}")
            class_pairs.append((item["gradient"], theirs_class))
            if theirs_class != item["gradient"]:
                class_disagreements.append({
                    "id": item_id,
                    "first_labeler": item["gradient"],
                    "second_labeler": theirs_class,
                    "note": entry["note"],
                })
        items_labeled += any_field
    per_field = {}
    for field, counts in fields.items():
        n = counts["n"]
        per_field[field] = {
            "n": n,
            "strict": counts["strict"],
            "within": counts["within"],
            "covered": counts["covered"],
            "unlabeled": counts["unlabeled"],
            "strict_rate": counts["strict"] / n if n else None,
            "within_rate": counts["within"] / n if n else None,
            "covered_rate": counts["covered"] / n if n else None,
        }
    n_class = len(class_pairs)
    class_agree = sum(1 for a, b in class_pairs if a == b)
    collapsed = [(a == "clean", b == "clean") for a, b in class_pairs]
    return {
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "sheet_items": len(sheet),
        "items_labeled": items_labeled,
        "fields": per_field,
        "class": {
            "n": n_class,
            "agree": class_agree,
            "agree_rate": class_agree / n_class if n_class else None,
            "kappa_3class": cohen_kappa(class_pairs),
            "kappa_clean_vs_not": cohen_kappa(collapsed),
            "unlabeled": class_unlabeled,
        },
        "disagreements": disagreements,
        "class_disagreements": class_disagreements,
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(summary):
    lines = [
        "| field | n | strict | within recorded | covered | unlabeled |",
        "|---|---|---|---|---|---|",
    ]
    for field, block in summary["fields"].items():
        lines.append(
            f"| {field} | {block['n']} | {block['strict']} ({_fmt(block['strict_rate'])}) "
            f"| {block['within']} ({_fmt(block['within_rate'])}) "
            f"| {block['covered']} ({_fmt(block['covered_rate'])}) | {block['unlabeled']} |"
        )
    cls = summary["class"]
    lines.append("")
    lines.append(
        f"class: {cls['agree']}/{cls['n']} agree ({_fmt(cls['agree_rate'])}), "
        f"kappa 3-class {_fmt(cls['kappa_3class'])}, "
        f"kappa clean-vs-not {_fmt(cls['kappa_clean_vs_not'])}, "
        f"unlabeled {cls['unlabeled']}"
    )
    if summary["disagreements"]:
        lines.append("")
        lines.append("field disagreements (strict):")
        for d in summary["disagreements"]:
            lines.append(
                f"- {d['id']} {d['field']}: first={d['first_labeler']!r} "
                f"second={d['second_labeler']!r} within={d['within_recorded']} "
                f"covered={d['covered']}"
            )
    if summary["class_disagreements"]:
        lines.append("")
        lines.append("class disagreements:")
        for d in summary["class_disagreements"]:
            note = f" ({d['note']})" if d["note"] else ""
            lines.append(
                f"- {d['id']}: first={d['first_labeler']} second={d['second_labeler']}{note}"
            )
    return "\n".join(lines)


def write_agreement(corpus_path, summary):
    """Record the summary under meta.labeling.agreement; items untouched."""
    corpus = load_corpus(corpus_path)
    corpus["meta"].setdefault("labeling", {})["agreement"] = summary
    with open(corpus_path, "w", encoding="utf-8") as handle:
        json.dump(corpus, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Study-5 natural arm: second-labeler agreement")
    parser.add_argument("--sheet", default=str(SHEET_PATH))
    parser.add_argument("--corpus", default=str(NATURAL_CORPUS_PATH))
    parser.add_argument("--write", action="store_true",
                        help="record the summary in the corpus meta.labeling.agreement")
    parser.add_argument("--json", default=None, help="also write the summary JSON here")
    args = parser.parse_args(argv)
    with open(args.sheet, encoding="utf-8") as handle:
        sheet = parse_sheet(handle.read())
    corpus = load_corpus(args.corpus)
    summary = compare(corpus, sheet)
    print(render_markdown(summary))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"summary -> {args.json}")
    if summary["items_labeled"] == 0:
        print("sheet is unlabeled - nothing to record")
        return 1
    if args.write:
        write_agreement(args.corpus, summary)
        print(f"agreement recorded in {args.corpus} meta.labeling.agreement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

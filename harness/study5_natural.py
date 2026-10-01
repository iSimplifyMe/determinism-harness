"""Study-5 held-out NATURAL arm: sampling, rendering, and reproducibility.

PROTOCOL.md section 4 requires at least one circularity mitigation beyond
gradient-frozen-pre-data. This module is mitigation 2: a held-out set of
naturally-occurring inventory records that nobody on the study authored.
Two public-domain government feeds, snapshotted into
fixtures/study5/natural/snapshots/ (sha256 recorded in the corpus meta):

- Austin, TX — Arterial Management Materials Warehouse Inventory
  (data.austintexas.gov/hcaw-evi2): traffic-signal stock; unit cost,
  total on hand, two name fields, re-order and value distractors.
- Montgomery County, MD — ABS Store Inventory and Sale Items
  (data.montgomerycountymd.gov/ib5t-5ncy): retail beverage catalog;
  description with embedded size/pack tokens, price and sale price,
  total inventory.

Everything discretionary is fixed here, BEFORE the draw, so the corpus
is re-derivable by anyone from the committed snapshots:

- eligibility: a mechanical rule per source (below), no hand selection;
- the draw: `random.Random(seed).sample` over eligible rows sorted by
  the source's row key, with the seed derived from the snapshot file's
  own sha256 — no discretionary seed;
- rendering: the source's own column display names, "Label: value",
  one field per line, in the source's column order, values verbatim
  (outer whitespace trimmed), record-keeping columns (ids, timestamps)
  excluded; no rewording, no field invented or dropped beyond that.

Ground truth, gradient, and target_field are then assigned at labeling
under PROTOCOL sections 2/5/7 — after the draw, before any model call.
`check_corpus` re-derives the draw and the documents and reports any
divergence; the test suite runs it against the committed corpus.
Stdlib only.
"""
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NATURAL_DIR = REPO / "fixtures/study5/natural"
SNAPSHOT_DIR = NATURAL_DIR / "snapshots"
NATURAL_CORPUS_PATH = REPO / "fixtures/study5/natural_corpus.json"

SAMPLE_PER_SOURCE = 25
ID_PREFIX = "s5n-"


def _num(value):
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _positive(value):
    number = _num(value)
    return number is not None and number > 0


def _nonneg_int(value):
    number = _num(value)
    return number is not None and number >= 0 and number.is_integer()


def _positive_int(value):
    number = _num(value)
    return number is not None and number > 0 and number.is_integer()


def _austin_eligible(row):
    return (
        bool(str(row.get("financial_name", "")).strip())
        and _positive(row.get("unit_cost"))
        and _nonneg_int(row.get("total_on_hand"))
    )


def _montgomery_eligible(row):
    return (
        bool(str(row.get("description", "")).strip())
        and _positive(row.get("price"))
        and _positive_int(row.get("totalinventory"))
    )


SOURCES = (
    {
        "key": "austin",
        "dataset": "data.austintexas.gov/hcaw-evi2",
        "name": "Arterial Management Materials Warehouse Inventory",
        "publisher": "City of Austin, Texas",
        "snapshot": "austin-hcaw-evi2.json",
        "row_key": "stock_number",
        "eligible": _austin_eligible,
        "eligibility_rule": (
            "latest published_date snapshot; financial_name non-empty; "
            "unit_cost present and > 0; total_on_hand present, integer, >= 0"
        ),
        # (field, Socrata display name) in the dataset's column order.
        "render": (
            ("financial_name", "Financial Name"),
            ("common_name", "Common Name"),
            ("tracking_type", "Tracking Type"),
            ("stock_number", "Stock Number"),
            ("category", "Category"),
            ("object", "Object"),
            ("unit_cost", "Unit Cost"),
            ("unit_of_measure", "Unit of Measure"),
            ("status", "Status"),
            ("re_order_threshold", "Re-Order Threshold"),
            ("total_on_hand", "Total On Hand"),
            ("total_value", "Total Value"),
            ("re_order_turnaround_time", "Re-Order Turnaround Time"),
            ("re_order_status", "Re-Order Status"),
        ),
        "excluded": ("id", "published_date", "modified_date"),
    },
    {
        "key": "montgomery",
        "dataset": "data.montgomerycountymd.gov/ib5t-5ncy",
        "name": "ABS Store Inventory and Sale Items",
        "publisher": "Montgomery County, MD",
        "snapshot": "montgomery-ib5t-5ncy.json",
        "row_key": "code",
        "eligible": _montgomery_eligible,
        "eligibility_rule": (
            "description non-empty; price > 0; totalinventory present, "
            "integer, > 0 (currently stocked items)"
        ),
        "render": (
            ("code", "Code"),
            ("category", "Category"),
            ("description", "Description"),
            ("size", "Size"),
            ("totalinventory", "Total Inventory"),
            ("price", "Price"),
            ("saleprice", "Sale Price"),
            ("saleenddate", "Sale End Date"),
        ),
        "excluded": (),
    },
)


def source_by_key(key):
    for source in SOURCES:
        if source["key"] == key:
            return source
    raise KeyError(key)


def snapshot_path(source):
    return SNAPSHOT_DIR / source["snapshot"]


def load_snapshot(source):
    raw = snapshot_path(source).read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def seed_from_sha(sha256_hex):
    """The draw's seed IS the snapshot hash — nothing to choose."""
    return int(sha256_hex[:16], 16)


def eligible_rows(source, rows):
    keep = [row for row in rows if source["eligible"](row)]
    keys = [str(row[source["row_key"]]) for row in keep]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{source['key']}: duplicate row keys among eligible rows")
    return sorted(keep, key=lambda row: str(row[source["row_key"]]))


def draw(source, rows, sha256_hex, n=SAMPLE_PER_SOURCE):
    pool = eligible_rows(source, rows)
    rng = random.Random(seed_from_sha(sha256_hex))
    return rng.sample(pool, n)


def render_document(source, row):
    lines = []
    for field, label in source["render"]:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _skeleton_item(source, row, item_id):
    return {
        "id": item_id,
        "gradient": "",
        "target_field": "",
        "document": render_document(source, row),
        "ground_truth": {
            "item_name": None,
            "unit_price": None,
            "quantity_in_stock": None,
        },
        "acceptable_alternatives": [],
        "rationale": "",
        "source": {
            "dataset": source["dataset"],
            "row_key": source["row_key"],
            "key": str(row[source["row_key"]]),
            "raw": row,
        },
    }


def build_skeleton(primary_meta):
    """Unlabeled natural corpus: provenance + documents, labels empty.
    Templates are copied from the primary corpus verbatim — the natural
    arm asks the same five questions of different documents."""
    sources_meta = []
    items = []
    counter = 0
    for source in SOURCES:
        snapshot, sha = load_snapshot(source)
        rows = snapshot["rows"]
        chosen = draw(source, rows, sha)
        sources_meta.append({
            "key": source["key"],
            "dataset": source["dataset"],
            "name": source["name"],
            "publisher": source["publisher"],
            "snapshot": f"natural/snapshots/{source['snapshot']}",
            "snapshot_sha256": sha,
            "snapshot_fetched_utc": snapshot.get("fetched_utc"),
            "snapshot_rows": len(rows),
            "eligible_rows": len(eligible_rows(source, rows)),
            "eligibility_rule": source["eligibility_rule"],
            "draw": (
                f"random.Random(int(sha256[:16], 16)).sample(eligible sorted "
                f"by {source['row_key']}, {SAMPLE_PER_SOURCE})"
            ),
            "render": "column display names, 'Label: value' per line, "
                      "column order, values verbatim (outer whitespace "
                      "trimmed), excluded: "
                      + (", ".join(source["excluded"]) or "none"),
            "sampled": SAMPLE_PER_SOURCE,
        })
        for row in chosen:
            counter += 1
            items.append(_skeleton_item(source, row, f"{ID_PREFIX}{counter:03d}"))
    return {
        "meta": {
            "study": primary_meta["study"],
            "arm": "natural",
            "planned_n": SAMPLE_PER_SOURCE * len(SOURCES),
            "frozen": False,
            "sources": sources_meta,
            "instruction_templates": dict(primary_meta["instruction_templates"]),
            "labeling": {
                "primary_labeler": "",
                "second_labeler": "",
                "agreement": None,
            },
        },
        "items": items,
    }


def check_corpus(corpus):
    """Re-derive the draw and every document from the committed snapshots;
    return a list of divergences (empty = the corpus is exactly what the
    declared procedure produces)."""
    errors = []
    meta = corpus.get("meta", {})
    if meta.get("arm") != "natural":
        errors.append("meta.arm != 'natural'")
    by_key = {s["key"]: s for s in meta.get("sources", [])}
    expected_items = []
    for source in SOURCES:
        try:
            snapshot, sha = load_snapshot(source)
        except FileNotFoundError:
            errors.append(f"{source['key']}: snapshot missing")
            continue
        declared = by_key.get(source["key"])
        if not declared:
            errors.append(f"{source['key']}: absent from meta.sources")
        elif declared.get("snapshot_sha256") != sha:
            errors.append(f"{source['key']}: snapshot sha256 mismatch")
        for row in draw(source, snapshot["rows"], sha):
            expected_items.append((source, row))
    items = corpus.get("items", [])
    if len(items) != len(expected_items):
        errors.append(f"item count {len(items)} != draw {len(expected_items)}")
        return errors
    for index, (item, (source, row)) in enumerate(zip(items, expected_items), 1):
        item_id = item.get("id")
        if item_id != f"{ID_PREFIX}{index:03d}":
            errors.append(f"item {index}: id {item_id!r} out of draw order")
        src = item.get("source", {})
        if src.get("dataset") != source["dataset"]:
            errors.append(f"{item_id}: dataset != {source['dataset']}")
        if src.get("key") != str(row[source["row_key"]]):
            errors.append(f"{item_id}: row key != drawn row")
        if src.get("raw") != row:
            errors.append(f"{item_id}: raw row differs from snapshot")
        if item.get("document") != render_document(source, row):
            errors.append(f"{item_id}: document != rendered snapshot row")
    return errors


def main(argv=None):
    import argparse
    import sys

    from harness.study5_fixtures import load_corpus

    parser = argparse.ArgumentParser(description="Study-5 natural arm")
    parser.add_argument("command", choices=("skeleton", "check"))
    parser.add_argument("--out", default=str(NATURAL_CORPUS_PATH))
    args = parser.parse_args(argv)
    if args.command == "skeleton":
        primary = load_corpus()
        corpus = build_skeleton(primary["meta"])
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(corpus, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
        print(f"skeleton: {len(corpus['items'])} items -> {args.out}")
        return 0
    corpus = load_corpus(args.out)
    errors = check_corpus(corpus)
    for error in errors:
        print(f"DIVERGENCE: {error}", file=sys.stderr)
    print("natural corpus check:", "CLEAN" if not errors else f"{len(errors)} divergences")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

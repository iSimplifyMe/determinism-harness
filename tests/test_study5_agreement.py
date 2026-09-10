"""Second-labeler agreement (PROTOCOL 7.4): sheet grammar, canonical
comparison, kappa, and the --write path leave the items untouched."""
import copy
import json
import os
import tempfile
import unittest

from analysis.study5_agreement import (
    SHEET_PATH,
    cohen_kappa,
    compare,
    parse_sheet,
    render_markdown,
    write_agreement,
)
from harness.study5_fixtures import (
    NATURAL_CORPUS_PATH,
    load_corpus,
    validate_natural_corpus,
)
from harness.study5_natural import check_corpus


def _fill(sheet_text, values):
    """Fill the shipped sheet: values = {item_id: {field: str}}."""
    out = []
    current = None
    for line in sheet_text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
        for field in ("item_name", "unit_price", "quantity_in_stock", "class", "note"):
            if line.startswith(f"- {field}:") and current in values \
                    and field in values[current]:
                line = f"- {field}: {values[current][field]}"
        out.append(line)
    return "\n".join(out)


class TestSheetGrammar(unittest.TestCase):
    def test_shipped_sheet_parses_unlabeled(self):
        sheet = parse_sheet(SHEET_PATH.read_text())
        self.assertEqual(len(sheet), 20)
        for entry in sheet.values():
            self.assertEqual(entry["item_name"], [])
            self.assertEqual(entry["unit_price"], [])
            self.assertEqual(entry["quantity_in_stock"], [])
            self.assertIsNone(entry["class"])

    def test_readings_null_and_alternatives(self):
        text = (
            "## s5n-001\n```\nX: y\n```\n- item_name: A | B\n- unit_price: $1,234.50\n"
            "- quantity_in_stock: NULL\n- class: Near-Tie\n- note: hm\n"
        )
        sheet = parse_sheet(text)
        entry = sheet["s5n-001"]
        self.assertEqual(entry["item_name"], ["A", "B"])
        self.assertEqual(entry["unit_price"], ["$1,234.50"])
        self.assertEqual(entry["quantity_in_stock"], [None])
        self.assertEqual(entry["class"], "near_tie")
        self.assertEqual(entry["note"], "hm")

    def test_duplicate_block_rejected(self):
        text = "## s5n-001\n```\nX\n```\n- class: clean\n## s5n-001\n```\nX\n```\n"
        with self.assertRaises(ValueError):
            parse_sheet(text)


class TestKappa(unittest.TestCase):
    def test_perfect_and_empty(self):
        self.assertEqual(cohen_kappa([("a", "a"), ("b", "b")]), 1.0)
        self.assertIsNone(cohen_kappa([]))

    def test_known_value(self):
        # 20 A/A, 5 A/B, 10 B/A, 15 B/B: po=0.70, pe=0.25*0.30+0.75*0.70... computed
        pairs = [("A", "A")] * 20 + [("A", "B")] * 5 + [("B", "A")] * 10 + [("B", "B")] * 15
        po = 35 / 50
        pa_a, pb_a = 25 / 50, 30 / 50
        pa_b, pb_b = 25 / 50, 20 / 50
        pe = pa_a * pb_a + pa_b * pb_b
        self.assertAlmostEqual(cohen_kappa(pairs), (po - pe) / (1 - pe))


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(NATURAL_CORPUS_PATH)
        cls.sheet_text = SHEET_PATH.read_text()
        cls.by_id = {it["id"]: it for it in cls.corpus["items"]}
        cls.ids = list(parse_sheet(cls.sheet_text))

    def _perfect_values(self):
        values = {}
        for item_id in self.ids:
            item = self.by_id[item_id]
            gt = item["ground_truth"]
            values[item_id] = {
                "item_name": gt["item_name"],
                "unit_price": str(gt["unit_price"]),
                "quantity_in_stock": str(gt["quantity_in_stock"]),
                "class": item["gradient"],
            }
        return values

    def test_perfect_agreement(self):
        sheet = parse_sheet(_fill(self.sheet_text, self._perfect_values()))
        summary = compare(self.corpus, sheet)
        self.assertEqual(summary["items_labeled"], 20)
        for field, block in summary["fields"].items():
            self.assertEqual(block["n"], 20, field)
            self.assertEqual(block["strict_rate"], 1.0, field)
        self.assertEqual(summary["class"]["agree"], 20)
        self.assertEqual(summary["class"]["kappa_3class"], 1.0)
        self.assertEqual(summary["disagreements"], [])
        self.assertIn("| item_name | 20 | 20 (1.00)", render_markdown(summary))

    def test_alternative_first_is_within_and_covered(self):
        values = self._perfect_values()
        # An ambiguous Montgomery item: the second labeler leads with the
        # stripped reading and lists the verbatim one second.
        target = next(
            it for it in self.corpus["items"]
            if it["id"] in self.ids and it["gradient"] == "ambiguous"
            and it["target_field"] == "item_name" and it["acceptable_alternatives"]
        )
        alt = target["acceptable_alternatives"][0]["item_name"]
        values[target["id"]]["item_name"] = f"{alt} | {target['ground_truth']['item_name']}"
        # A rounded price and a wrong class elsewhere.
        other = next(it for it in self.corpus["items"] if it["id"] in self.ids and it["id"] != target["id"])
        values[other["id"]]["unit_price"] = "1.00"
        values[other["id"]]["class"] = "clean" if other["gradient"] != "clean" else "ambiguous"
        sheet = parse_sheet(_fill(self.sheet_text, values))
        summary = compare(self.corpus, sheet)
        name = summary["fields"]["item_name"]
        self.assertEqual(name["strict"], 19)
        self.assertEqual(name["within"], 20)
        self.assertEqual(name["covered"], 20)
        price = summary["fields"]["unit_price"]
        self.assertEqual(price["strict"], 19)
        self.assertEqual(price["within"], 19)
        self.assertEqual(summary["class"]["agree"], 19)
        ids = {(d["id"], d["field"]) for d in summary["disagreements"]}
        self.assertEqual(ids, {(target["id"], "item_name"), (other["id"], "unit_price")})
        self.assertEqual(
            [d["id"] for d in summary["class_disagreements"]], [other["id"]]
        )

    def test_unlabeled_counts_and_unknown_id(self):
        values = self._perfect_values()
        del values[self.ids[0]]
        sheet = parse_sheet(_fill(self.sheet_text, values))
        summary = compare(self.corpus, sheet)
        self.assertEqual(summary["items_labeled"], 19)
        self.assertEqual(summary["fields"]["item_name"]["unlabeled"], 1)
        self.assertEqual(summary["class"]["unlabeled"], 1)
        with self.assertRaises(ValueError):
            compare(self.corpus, {"s5n-999": sheet[self.ids[1]]})

    def test_write_records_meta_only(self):
        sheet = parse_sheet(_fill(self.sheet_text, self._perfect_values()))
        summary = compare(self.corpus, sheet)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "natural_corpus.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(copy.deepcopy(self.corpus), fh, indent=1, ensure_ascii=False)
            write_agreement(path, summary)
            written = load_corpus(path)
            self.assertEqual(
                written["meta"]["labeling"]["agreement"]["class"]["agree"], 20
            )
            self.assertEqual(written["items"], self.corpus["items"])
            self.assertEqual(check_corpus(written), [])
            self.assertEqual(validate_natural_corpus(written), [])


if __name__ == "__main__":
    unittest.main()

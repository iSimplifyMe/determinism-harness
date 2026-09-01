"""Study-5 analyzer: validation-first parsing, canonicalization,
correctness (strict/lenient), and the catch-rate machinery — exercised
on constructed truth tables so every rate is hand-checkable."""
import unittest

from analysis.analyze_study5 import (
    answer_correct,
    canonical_field,
    cross_pair_analysis,
    fields_disagree,
    index_records,
    kset_analysis,
    parse_response,
    resample_analysis,
)


def item(item_id, target="item_name", gradient="near_tie", alts=None,
         name="Corvid 3000 Optical Mouse", price=18.99, qty=44):
    return {
        "id": item_id,
        "gradient": gradient,
        "target_field": target,
        "document": f"doc {item_id}",
        "ground_truth": {
            "item_name": name, "unit_price": price, "quantity_in_stock": qty,
        },
        "acceptable_alternatives": alts or [],
        "rationale": "r",
    }


def record(substrate, item_id, template_id, text, control=None, ok=True):
    """Runner-shaped record: flattened meta_* keys + text (Engine base)."""
    rec = {
        "ok": ok,
        "meta_substrate": substrate,
        "meta_arm": "paraphrase",
        "meta_item_id": item_id,
        "meta_template_id": template_id,
        "text": text,
    }
    if control:
        rec["meta_control"] = control
    return rec


def answer(name="Corvid 3000 Optical Mouse", price=18.99, qty=44):
    import json
    return json.dumps({
        "item_name": name, "unit_price": price, "quantity_in_stock": qty,
    })


class TestParseResponse(unittest.TestCase):
    def test_bare_json(self):
        mode, obj = parse_response('{"item_name": "X"}')
        self.assertEqual(mode, "bare")
        self.assertEqual(obj["item_name"], "X")

    def test_fenced_json(self):
        mode, obj = parse_response('```json\n{"item_name": "X"}\n```')
        self.assertEqual(mode, "fenced")
        self.assertEqual(obj["item_name"], "X")

    def test_fail_modes(self):
        self.assertEqual(parse_response("not json")[0], "fail")
        self.assertEqual(parse_response('["a", "b"]')[0], "fail")
        self.assertEqual(parse_response(None)[0], "fail")


class TestCanonicalField(unittest.TestCase):
    def test_name_whitespace_collapse_only(self):
        self.assertEqual(
            canonical_field("item_name", "  Wren   Desk\tOrganizer "),
            "Wren Desk Organizer",
        )
        self.assertNotEqual(
            canonical_field("item_name", "wren desk organizer"),
            canonical_field("item_name", "Wren Desk Organizer"),
        )

    def test_price_forms(self):
        self.assertEqual(canonical_field("unit_price", "$14.50"), 14.5)
        self.assertEqual(canonical_field("unit_price", 14.5), 14.5)
        self.assertEqual(canonical_field("unit_price", 14), 14.0)
        self.assertEqual(canonical_field("unit_price", "1,150.00"), 1150.0)
        self.assertIsNone(canonical_field("unit_price", None))
        self.assertEqual(
            canonical_field("unit_price", "twelve")[0], "invalid"
        )

    def test_quantity_forms(self):
        self.assertEqual(canonical_field("quantity_in_stock", "40"), 40)
        self.assertEqual(canonical_field("quantity_in_stock", 40.0), 40)
        self.assertEqual(
            canonical_field("quantity_in_stock", 40.5)[0], "invalid"
        )
        self.assertEqual(
            canonical_field("quantity_in_stock", True)[0], "invalid"
        )


class TestCorrectness(unittest.TestCase):
    def test_strict_and_lenient(self):
        it = item(
            "s5-001", target="unit_price", gradient="ambiguous",
            price=17.5, alts=[{"unit_price": 21.0}],
        )
        import json
        promo = json.loads(answer(price=17.5))
        listp = json.loads(answer(price=21.0))
        other = json.loads(answer(price=19.0))
        self.assertTrue(answer_correct(it, promo))
        self.assertFalse(answer_correct(it, listp))
        self.assertTrue(answer_correct(it, listp, lenient=True))
        self.assertFalse(answer_correct(it, other, lenient=True))

    def test_null_ground_truth(self):
        it = item("s5-002", target="unit_price", price=None)
        import json
        self.assertTrue(
            answer_correct(it, json.loads(answer(price=None)))
        )
        self.assertFalse(
            answer_correct(it, json.loads(answer(price=0.0)))
        )


class TestDisagree(unittest.TestCase):
    def test_semantic_not_bytes(self):
        a = {"unit_price": "$14.50"}
        b = {"unit_price": 14.5}
        self.assertFalse(fields_disagree(a, b, "unit_price"))
        self.assertTrue(
            fields_disagree(a, {"unit_price": 14.0}, "unit_price")
        )
        self.assertTrue(
            fields_disagree({"unit_price": None}, b, "unit_price")
        )


class TestIndexRecords(unittest.TestCase):
    def test_controls_and_failures_counted_not_scored(self):
        records = [
            record("haiku_1p", "warmup", "warmup", "ok", control="warmup"),
            record("haiku_1p", "s5-001", "t1", answer()),
            record("haiku_1p", "s5-001", "t2", "garbage"),
            record("haiku_1p", "s5-001", "t3",
                   "```json\n" + answer() + "\n```"),
            record("haiku_1p", "s5-001", "t4", answer(), ok=False),
        ]
        parsed, counts = index_records(records)
        self.assertEqual(counts["excluded_control"], 1)
        self.assertEqual(counts["not_ok"], 1)
        self.assertEqual(counts["in"], 3)
        self.assertEqual(counts["parse_fail"], 1)
        self.assertEqual(counts["fenced"], 1)
        self.assertEqual(len(parsed[("haiku_1p", "s5-001", "t1")]), 1)
        self.assertNotIn(("haiku_1p", "s5-001", "t2"), parsed)
        self.assertNotIn(("haiku_1p", "s5-001", "t4"), parsed)

    def test_dict_meta_records_also_accepted(self):
        rec = {
            "meta": {
                "substrate": "haiku_1p", "arm": "paraphrase",
                "item_id": "s5-001", "template_id": "t1",
            },
            "response_text": answer(),
        }
        parsed, counts = index_records([rec])
        self.assertEqual(counts["in"], 1)
        self.assertEqual(len(parsed[("haiku_1p", "s5-001", "t1")]), 1)


class TestKsetAnalysis(unittest.TestCase):
    """Four items, hand-built truth table on t1/t2:
    - s5-001: agree, right    - s5-002: agree, wrong
    - s5-003: disagree, right - s5-004: disagree, wrong
    catch = 1/2, false alarm = 1/2, RR = (1/2)/(1/2) = 1.0
    """
    def setUp(self):
        self.items = [
            item("s5-001"), item("s5-002"),
            item("s5-003"), item("s5-004"),
        ]
        wrong = answer(name="Item Corvid 3000 Optical Mouse")
        right = answer()
        records = [
            record("haiku_1p", "s5-001", "t1", right),
            record("haiku_1p", "s5-001", "t2", right),
            record("haiku_1p", "s5-002", "t1", wrong),
            record("haiku_1p", "s5-002", "t2", wrong),
            record("haiku_1p", "s5-003", "t1", right),
            record("haiku_1p", "s5-003", "t2", wrong),
            record("haiku_1p", "s5-004", "t1", wrong),
            record("haiku_1p", "s5-004", "t2", right),
        ]
        self.parsed, _ = index_records(records)

    def test_pair_rates(self):
        out = kset_analysis(
            self.parsed, self.items, "haiku_1p", ("t1", "t2")
        )
        self.assertEqual(out["n_items"], 4)
        self.assertEqual(out["n_wrong"], 2)
        self.assertEqual(out["n_disagree"], 2)
        self.assertEqual(out["catch_rate"], 0.5)
        self.assertEqual(out["false_alarm_rate"], 0.5)
        self.assertEqual(out["relative_risk"], 1.0)
        self.assertEqual(out["excluded_missing"], 0)
        self.assertIn("near_tie", out["by_gradient"])

    def test_missing_template_excludes_item(self):
        items = self.items + [item("s5-005")]
        out = kset_analysis(self.parsed, items, "haiku_1p", ("t1", "t2"))
        self.assertEqual(out["excluded_missing"], 1)
        self.assertEqual(out["n_items"], 4)

    def test_k3_disagreement_any(self):
        records = [
            record("haiku_1p", "s5-001", "t1", answer()),
            record("haiku_1p", "s5-001", "t2", answer()),
            record("haiku_1p", "s5-001", "t3",
                   answer(name="Corvid 3000")),
        ]
        parsed, _ = index_records(records)
        out = kset_analysis(
            parsed, [item("s5-001")], "haiku_1p", ("t1", "t2", "t3")
        )
        self.assertEqual(out["k"], 3)
        self.assertEqual(out["n_disagree"], 1)


class TestCrossAndResample(unittest.TestCase):
    def test_cross_pair(self):
        records = [
            record("sonnet_1p", "s5-001", "t1", answer()),
            record("sonnet_bedrock", "s5-001", "t1",
                   answer(name="Corvid 3000")),
        ]
        parsed, _ = index_records(records)
        out = cross_pair_analysis(
            parsed, [item("s5-001")], "sonnet_1p", "sonnet_bedrock", "t1"
        )
        self.assertEqual(out["n_disagree"], 1)
        self.assertEqual(out["substrates"], ["sonnet_1p", "sonnet_bedrock"])

    def test_resample(self):
        base = [
            record("haiku_1p", "s5-001", "t1", answer())
            for _ in range(4)
        ]
        base.append(
            record("haiku_1p", "s5-001", "t1", answer(name="Corvid 3000"))
        )
        for r in base:
            r["meta_arm"] = "resample"
        parsed, _ = index_records(base)
        out = resample_analysis(
            parsed, [item("s5-001")], "haiku_1p", "t1"
        )
        self.assertEqual(out["n_items"], 1)
        self.assertEqual(out["n_disagree"], 1)

    def test_resample_needs_two(self):
        records = [record("haiku_1p", "s5-001", "t1", answer())]
        records[0]["meta_arm"] = "resample"
        parsed, _ = index_records(records)
        out = resample_analysis(parsed, [item("s5-001")], "haiku_1p", "t1")
        self.assertEqual(out["excluded_missing"], 1)
        self.assertEqual(out["n_items"], 0)


if __name__ == "__main__":
    unittest.main()

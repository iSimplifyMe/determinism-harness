"""Study-5 fixture-corpus validators (confidence-signal study).

The corpus is the study's answer key: every item carries ground truth and an
ambiguity-gradient class assigned at authoring, blind to any model output.
The disagreement generator is a set of corpus-global instruction templates
(semantically identical asks, independently worded) — global rather than
per-item so no phrasing can be tuned to an item's ambiguity, and so the
k-ladder (k=2,3,5) has k registered asks.

These tests are the mechanical half of the labeling protocol
(fixtures/study5/PROTOCOL.md) — an item that loads clean here is
protocol-conformant in every property a program can check.
"""
import unittest
from pathlib import Path

from harness.study5_fixtures import (
    CORPUS_PATH,
    GRADIENTS,
    SCHEMA_KEYS,
    TEMPLATE_IDS,
    load_corpus,
    validate_corpus,
)


class TestModuleContract(unittest.TestCase):
    def test_schema_keys_fixed(self):
        self.assertEqual(
            SCHEMA_KEYS, ("item_name", "unit_price", "quantity_in_stock")
        )

    def test_gradient_classes_fixed(self):
        self.assertEqual(GRADIENTS, ("clean", "near_tie", "ambiguous"))

    def test_template_ids_fixed(self):
        self.assertEqual(TEMPLATE_IDS, ("t1", "t2", "t3", "t4", "t5"))

    def test_corpus_path_in_repo(self):
        self.assertTrue(str(CORPUS_PATH).endswith("fixtures/study5/corpus.json"))


class TestCorpusLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus()
        cls.items = cls.corpus["items"]

    def test_file_exists_and_parses(self):
        self.assertTrue(Path(CORPUS_PATH).is_file())
        self.assertIsInstance(self.items, list)
        self.assertGreater(len(self.items), 0)

    def test_meta_present(self):
        meta = self.corpus["meta"]
        self.assertEqual(meta["study"], "study5-confidence-signal")
        self.assertIn("planned_n", meta)
        # The corpus must not claim frozen until gate-2 freeze actually
        # happens (freeze flips this bit in its own commit).
        self.assertIsInstance(meta["frozen"], bool)

    def test_validator_passes_on_shipped_corpus(self):
        self.assertEqual(validate_corpus(self.corpus), [])


class TestInstructionTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates = load_corpus()["meta"]["instruction_templates"]

    def test_exactly_the_registered_ids(self):
        self.assertEqual(tuple(sorted(self.templates)), TEMPLATE_IDS)

    def test_each_names_every_schema_key_and_the_null_rule(self):
        """Semantically identical asks: same three keys, same missing-value
        rule. A template that omits a key or the null rule is a different
        question, not a paraphrase."""
        for template_id, text in self.templates.items():
            for key in SCHEMA_KEYS:
                self.assertIn(key, text, template_id)
            self.assertIn("null", text.lower(), template_id)

    def test_pairwise_distinct(self):
        texts = [t.strip() for t in self.templates.values()]
        self.assertEqual(len(texts), len(set(texts)))

    def test_no_scope_hints(self):
        """Templates must not disambiguate what the documents leave open
        (e.g. 'current' price, 'total' stock) — the ambiguity under test
        lives in the documents alone."""
        for template_id, text in self.templates.items():
            lowered = text.lower()
            for hint in ("current", "total", "combined", "per unit", "on-hand"):
                self.assertNotIn(hint, lowered, template_id)


class TestItemInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = load_corpus()["items"]

    def test_ids_unique_and_formatted(self):
        ids = [i["id"] for i in self.items]
        self.assertEqual(len(ids), len(set(ids)))
        for item_id in ids:
            self.assertRegex(item_id, r"^s5-\d{3}$")

    def test_gradient_valid(self):
        for item in self.items:
            self.assertIn(item["gradient"], GRADIENTS, item["id"])

    def test_ground_truth_schema(self):
        for item in self.items:
            gt = item["ground_truth"]
            self.assertEqual(set(gt), set(SCHEMA_KEYS), item["id"])
            self.assertIsInstance(gt["item_name"], str, item["id"])
            self.assertTrue(
                gt["unit_price"] is None
                or isinstance(gt["unit_price"], (int, float)),
                item["id"],
            )
            self.assertTrue(
                gt["quantity_in_stock"] is None
                or isinstance(gt["quantity_in_stock"], int),
                item["id"],
            )

    def test_target_field_is_schema_key(self):
        for item in self.items:
            self.assertIn(item["target_field"], SCHEMA_KEYS, item["id"])

    def test_documents_nonempty_bounded_unique(self):
        docs = []
        for item in self.items:
            doc = item["document"]
            self.assertTrue(doc.strip(), item["id"])
            self.assertLessEqual(len(doc), 1500, item["id"])
            docs.append(doc)
        self.assertEqual(len(docs), len(set(docs)))

    def test_ambiguous_items_carry_alternatives(self):
        """Ambiguous = experts could defend two readings, so the second
        reading must be recorded; clean/near-tie have exactly one."""
        for item in self.items:
            alts = item.get("acceptable_alternatives", [])
            if item["gradient"] == "ambiguous":
                self.assertGreaterEqual(len(alts), 1, item["id"])
                for alt in alts:
                    self.assertIn(item["target_field"], alt, item["id"])
                    self.assertNotEqual(
                        alt[item["target_field"]],
                        item["ground_truth"][item["target_field"]],
                        item["id"],
                    )
            else:
                self.assertEqual(alts, [], item["id"])

    def test_rationale_required_off_clean(self):
        for item in self.items:
            if item["gradient"] != "clean":
                self.assertTrue(item["rationale"].strip(), item["id"])

    def test_document_does_not_leak_instructions(self):
        """Documents are catalog snippets, not prompts — a document that
        embeds the ask would make the instruction-template contrast
        cosmetic."""
        for item in self.items:
            lowered = item["document"].lower()
            for marker in ("json", "return a", "extract the"):
                self.assertNotIn(marker, lowered, item["id"])


class TestValidatorCatchesViolations(unittest.TestCase):
    """validate_corpus must be usable on candidate batches, not only the
    shipped file — feed it broken corpora and expect named errors."""

    @staticmethod
    def _minimal_item(**overrides):
        item = {
            "id": "s5-001",
            "gradient": "clean",
            "target_field": "item_name",
            "document": "Wren Desk Organizer. Price per unit: $14.50. In stock: 40.",
            "ground_truth": {
                "item_name": "Wren Desk Organizer",
                "unit_price": 14.50,
                "quantity_in_stock": 40,
            },
            "acceptable_alternatives": [],
            "rationale": "",
        }
        item.update(overrides)
        return item

    @staticmethod
    def _templates():
        base = (
            "Reply with one JSON object holding item_name, unit_price and "
            "quantity_in_stock; use null when unstated. Variant "
        )
        return {tid: base + tid for tid in TEMPLATE_IDS}

    def _corpus_with(self, items, templates=None):
        return {
            "meta": {
                "study": "study5-confidence-signal",
                "planned_n": 150,
                "frozen": False,
                "instruction_templates": (
                    self._templates() if templates is None else templates
                ),
            },
            "items": items,
        }

    def test_accepts_minimal_valid(self):
        self.assertEqual(
            validate_corpus(self._corpus_with([self._minimal_item()])), []
        )

    def test_flags_duplicate_ids(self):
        errors = validate_corpus(
            self._corpus_with([self._minimal_item(), self._minimal_item()])
        )
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_flags_bad_gradient(self):
        errors = validate_corpus(
            self._corpus_with([self._minimal_item(gradient="fuzzy")])
        )
        self.assertTrue(any("gradient" in e for e in errors))

    def test_flags_ambiguous_without_alternatives(self):
        errors = validate_corpus(
            self._corpus_with(
                [self._minimal_item(gradient="ambiguous", rationale="two reads")]
            )
        )
        self.assertTrue(any("alternative" in e for e in errors))

    def test_flags_template_missing_schema_key(self):
        templates = self._templates()
        templates["t3"] = "Answer with item_name and unit_price as null JSON."
        errors = validate_corpus(
            self._corpus_with([self._minimal_item()], templates=templates)
        )
        self.assertTrue(any("template" in e for e in errors))

    def test_flags_missing_template_id(self):
        templates = self._templates()
        del templates["t5"]
        errors = validate_corpus(
            self._corpus_with([self._minimal_item()], templates=templates)
        )
        self.assertTrue(any("template" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

"""Study-5 schedule builder invariants."""
import json
import unittest

from harness.study5_fixtures import TEMPLATE_IDS
from harness.study5_schedule import (
    RESAMPLE_N,
    RESAMPLE_TEMPLATE,
    RESAMPLE_TEMPERATURE,
    STUDY5_SUBSTRATES,
    SUBSTRATE_ORDER,
    build_prompt,
    build_study5_items,
    schedule_digest,
)

TEMPLATES = {
    tid: (
        f"Variant {tid}: reply with one JSON object holding item_name, "
        "unit_price and quantity_in_stock; use null when unstated."
    )
    for tid in TEMPLATE_IDS
}


def tiny_corpus(n=2):
    items = []
    for i in range(n):
        items.append({
            "id": f"s5-{i + 1:03d}",
            "gradient": "clean",
            "target_field": "item_name",
            "document": f"Wren Tester {i}, oak. $10.00 each. Stock: {i + 5}.",
            "ground_truth": {
                "item_name": f"Wren Tester {i}",
                "unit_price": 10.0,
                "quantity_in_stock": i + 5,
            },
            "acceptable_alternatives": [],
            "rationale": "",
        })
    return {
        "meta": {
            "study": "study5-confidence-signal",
            "planned_n": n,
            "frozen": False,
            "instruction_templates": TEMPLATES,
        },
        "items": items,
    }


class TestScheduleShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = tiny_corpus(2)
        cls.schedule = build_study5_items(cls.corpus)

    def test_deterministic(self):
        again = build_study5_items(tiny_corpus(2))
        self.assertEqual(
            schedule_digest(self.schedule), schedule_digest(again)
        )

    def test_block_order_and_warmups(self):
        substrates = [it["meta"]["substrate"] for it in self.schedule]
        # blocks appear in SUBSTRATE_ORDER, contiguous
        seen = []
        for s in substrates:
            if not seen or seen[-1] != s:
                seen.append(s)
        self.assertEqual(seen, list(SUBSTRATE_ORDER))
        for substrate, cfg in STUDY5_SUBSTRATES.items():
            block = [
                it for it in self.schedule
                if it["meta"]["substrate"] == substrate
            ]
            if cfg.get("warmup"):
                self.assertEqual(block[0]["meta"].get("control"), "warmup")
                self.assertEqual(
                    sum(1 for it in block
                        if it["meta"].get("control") == "warmup"),
                    1,
                )
            else:
                self.assertTrue(
                    all(not it["meta"].get("control") for it in block)
                )

    def test_paraphrase_coverage(self):
        """Every substrate with the paraphrase arm carries items x all
        templates, once each."""
        for substrate, cfg in STUDY5_SUBSTRATES.items():
            if "paraphrase" not in cfg["arms"]:
                continue
            block = [
                it for it in self.schedule
                if it["meta"]["substrate"] == substrate
                and it["meta"]["arm"] == "paraphrase"
                and not it["meta"].get("control")
            ]
            expected = {
                (item["id"], tid)
                for item in self.corpus["items"]
                for tid in TEMPLATE_IDS
            }
            got = {
                (it["meta"]["item_id"], it["meta"]["template_id"])
                for it in block
            }
            self.assertEqual(got, expected, substrate)
            self.assertEqual(len(block), len(expected), substrate)

    def test_resample_repeats_and_temperature(self):
        for substrate, cfg in STUDY5_SUBSTRATES.items():
            block = [
                it for it in self.schedule
                if it["meta"]["substrate"] == substrate
                and it["meta"]["arm"] == "resample"
            ]
            if "resample" not in cfg["arms"]:
                self.assertEqual(block, [])
                continue
            self.assertEqual(
                len(block), len(self.corpus["items"]) * RESAMPLE_N, substrate
            )
            self.assertTrue(
                all(it["meta"]["template_id"] == RESAMPLE_TEMPLATE
                    for it in block)
            )
            for it in block:
                payload = it["payload"]
                body = (
                    json.loads(payload.decode("ascii"))
                    if isinstance(payload, bytes) else payload
                )
                if cfg["kind"] == "local":
                    self.assertEqual(
                        body["options"]["temperature"], RESAMPLE_TEMPERATURE
                    )
                    self.assertNotIn("seed", body["options"], substrate)
                else:
                    self.assertEqual(
                        body["temperature"], RESAMPLE_TEMPERATURE
                    )

    def test_paraphrase_requests_deterministic_settings(self):
        """No sampling fields on the paraphrase arm; local keeps its seed."""
        for it in self.schedule:
            if it["meta"]["arm"] != "paraphrase":
                continue
            payload = it["payload"]
            body = (
                json.loads(payload.decode("ascii"))
                if isinstance(payload, bytes) else payload
            )
            cfg = STUDY5_SUBSTRATES[it["meta"]["substrate"]]
            if cfg["kind"] == "local":
                self.assertEqual(body["options"]["temperature"], 0)
                self.assertEqual(body["options"]["seed"], 42)
            else:
                self.assertNotIn("temperature", body)

    def test_prompt_carries_template_then_document(self):
        prompt = build_prompt("ASK", "DOC")
        self.assertEqual(prompt, "ASK\n\nDOC")
        sample = next(
            it for it in self.schedule
            if it["meta"]["arm"] == "paraphrase"
            and not it["meta"].get("control")
            and STUDY5_SUBSTRATES[it["meta"]["substrate"]]["kind"]
            == "messages"
        )
        text = sample["payload"]["messages"][0]["content"][0]["text"]
        self.assertIn(
            self.corpus["items"][0]["document"].split(".")[0].split(",")[0],
            text.split("\n\n", 1)[1],
        )
        self.assertTrue(
            text.startswith(TEMPLATES[sample["meta"]["template_id"]])
        )

    def test_meta_carries_join_keys(self):
        for it in self.schedule:
            if it["meta"].get("control"):
                continue
            self.assertIn("item_id", it["meta"])
            self.assertIn("template_id", it["meta"])
            self.assertIn("gradient", it["meta"])
            self.assertIn("target_field", it["meta"])
            self.assertTrue(it["sha"])

    def test_substrate_filter_and_limit(self):
        schedule = build_study5_items(
            tiny_corpus(2), substrates=("haiku_1p",), items_limit=1
        )
        self.assertTrue(
            all(it["meta"]["substrate"] == "haiku_1p" for it in schedule)
        )
        paraphrase = [
            it for it in schedule if it["meta"]["arm"] == "paraphrase"
        ]
        self.assertEqual(len(paraphrase), len(TEMPLATE_IDS))


if __name__ == "__main__":
    unittest.main()

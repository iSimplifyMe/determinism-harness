"""The gpt-oss:120b dedicated-window arm (registered 2026-07-29)."""
import json
import unittest

from harness.config import GPT_OSS_120B, REPEATS_FULL
from harness.runner import MODES, build_schedule, study3_run_settings


class Test120bWindow(unittest.TestCase):
    def test_mode_exists_and_is_single_flight(self):
        self.assertIn("study3-120b-window", MODES)
        settings = study3_run_settings("study3-120b-window")
        self.assertEqual(settings["concurrency"], 1)
        self.assertFalse(settings["allow_same_cell_concurrency"])

    def test_nine_cells_on_metal_only(self):
        items = build_schedule("study3-120b-window", box="metal", repeats=5)
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 9)  # 4 tasks x 2 sampling + sj high arm
        self.assertEqual(len(items), 9 * 5)
        with self.assertRaises(ValueError):
            build_schedule("study3-120b-window", box="cuda")

    def test_arms_and_tag(self):
        items = build_schedule("study3-120b-window", box="metal", repeats=1)
        high = [it for it in items if it["meta"]["thinking"] == "effort_high"]
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0]["meta"]["task"], "structured_json")
        for it in items:
            body = json.loads(it["payload"])
            self.assertEqual(body["model"], GPT_OSS_120B["tag"])
            self.assertEqual(it["meta"]["model"], "gpt-oss-120b")
            self.assertEqual(it["model_id"], "gpt-oss:120b")

    def test_default_repeats_full(self):
        items = build_schedule("study3-120b-window", box="metal")
        self.assertEqual(len(items), 9 * REPEATS_FULL)

    def test_core_modes_unaffected(self):
        # the 120b lives outside LOCAL_MODELS: core metal grid stays 32 cells
        cells = {it["cell"] for it in build_schedule("study3-full", box="metal")}
        self.assertEqual(len(cells), 32)
        self.assertFalse(any("120b" in c for c in cells))


if __name__ == "__main__":
    unittest.main()

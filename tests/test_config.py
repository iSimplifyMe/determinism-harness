"""Tests for the frozen grid, task set, and schedule construction.

The grid shape is a pre-registered quantity: (2 Claude-5 models x 4 tasks x
2 profiles x 2 thinking arms) + (Haiku 4.5 x 4 tasks x 2 profiles x 1 arm)
= 40 cells. Haiku 4.5 carries one thinking arm because adaptive thinking
does not exist on that model (verified against the Claude API reference).
"""
import unittest

from harness.config import (
    EFFORT_SWEEP,
    MODELS,
    POSITIVE_CONTROL,
    PROFILES,
    REPEATS_FULL,
    REPEATS_PILOT,
    cell_key,
    grid_cells,
)
from harness.runner import build_schedule
from harness.tasks import TASKS


class TestTasks(unittest.TestCase):
    def test_ladder_order(self):
        self.assertEqual(
            list(TASKS.keys()),
            ["extraction", "classification", "structured_json", "open_generation"],
        )

    def test_prompts_ascii_and_nonempty(self):
        for key, task in TASKS.items():
            self.assertTrue(task["prompt"].strip(), key)
            task["prompt"].encode("ascii")  # raises if non-ascii


class TestGrid(unittest.TestCase):
    def test_cell_count_is_40(self):
        self.assertEqual(len(list(grid_cells())), 40)

    def test_cell_keys_unique(self):
        keys = [cell_key(c) for c in grid_cells()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_profile_ids_have_expected_forms(self):
        for model_key, cfg in MODELS.items():
            self.assertEqual(set(cfg["profiles"].keys()), set(PROFILES))
            for form, pid in cfg["profiles"].items():
                self.assertTrue(
                    pid.startswith(f"{form}.anthropic.claude"), f"{model_key}:{pid}"
                )

    def test_haiku_is_anchor_and_control_host(self):
        haiku = MODELS["haiku-4-5"]
        self.assertEqual(haiku["thinking_arms"], ("none",))
        self.assertIsNone(haiku["effort"])
        self.assertTrue(haiku["supports_sampling"])
        self.assertTrue(haiku["dated_id"])

    def test_5family_rejects_sampling_and_pins_medium(self):
        for key in ("opus-5", "sonnet-5"):
            cfg = MODELS[key]
            self.assertFalse(cfg["supports_sampling"])
            self.assertEqual(cfg["effort"], "medium")
            self.assertEqual(cfg["thinking_arms"], ("adaptive", "disabled"))
            self.assertFalse(cfg["dated_id"])

    def test_positive_control_runs_on_sampling_model(self):
        self.assertTrue(MODELS[POSITIVE_CONTROL["model"]]["supports_sampling"])
        self.assertIn("temperature", POSITIVE_CONTROL["extra"])

    def test_effort_sweep_shape(self):
        self.assertEqual(len(EFFORT_SWEEP["tasks"]), 2)
        self.assertEqual(
            EFFORT_SWEEP["efforts"], ("low", "medium", "high", "xhigh", "max")
        )
        self.assertEqual(EFFORT_SWEEP["thinking"], "adaptive")


class TestSchedule(unittest.TestCase):
    def test_pilot_size(self):
        self.assertEqual(len(build_schedule("pilot")), 40 * REPEATS_PILOT)

    def test_full_size(self):
        self.assertEqual(len(build_schedule("full")), 40 * REPEATS_FULL)

    def test_positive_control_size(self):
        self.assertEqual(
            len(build_schedule("positive-control")), POSITIVE_CONTROL["repeats"]
        )

    def test_effort_sweep_size(self):
        self.assertEqual(
            len(build_schedule("effort-sweep")),
            len(EFFORT_SWEEP["tasks"])
            * len(EFFORT_SWEEP["efforts"])
            * EFFORT_SWEEP["repeats"],
        )

    def test_bodies_identical_within_cell(self):
        by_cell = {}
        for item in build_schedule("pilot"):
            by_cell.setdefault(item["cell"], set()).add(item["sha"])
        for cell, shas in by_cell.items():
            self.assertEqual(len(shas), 1, cell)

    def test_haiku_bodies_have_no_thinking_or_effort(self):
        import json

        for item in build_schedule("pilot"):
            if item["meta"]["model"] == "haiku-4-5":
                body = json.loads(item["body"])
                self.assertNotIn("thinking", body)
                self.assertNotIn("output_config", body)

    def test_effort_sweep_cells_carry_effort_in_key(self):
        keys = {item["cell"] for item in build_schedule("effort-sweep")}
        self.assertEqual(len(keys), 10)
        for key in keys:
            self.assertIn("effort=", key)


if __name__ == "__main__":
    unittest.main()

"""Study-5 runner wiring: mode registry, schedule building through
build_schedule, run settings, and a real dry-run through main() (writes
only a dryrun manifest — zero credentials, zero calls)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from harness.runner import (
    MODES,
    STUDY5_LOCAL_MODES,
    STUDY5_MODES,
    build_schedule,
    study5_run_settings,
)
from harness.study5_fixtures import GRADIENTS
from harness.study5_schedule import (
    PILOT_PER_CLASS,
    STUDY5_API_SUBSTRATES,
    pilot_corpus,
)
from harness.study5_fixtures import load_corpus


class TestModeRegistry(unittest.TestCase):
    def test_modes_registered(self):
        for mode in STUDY5_MODES:
            self.assertIn(mode, MODES)

    def test_run_settings(self):
        self.assertEqual(
            study5_run_settings("study5-pilot-local"), {"concurrency": 1}
        )
        self.assertEqual(study5_run_settings("study5-pilot-api"), {})
        self.assertIsNone(study5_run_settings("study2-pilot"))


class TestPilotCorpus(unittest.TestCase):
    def test_stratified_and_deterministic(self):
        sub = pilot_corpus(load_corpus())
        by_gradient = {}
        for it in sub["items"]:
            by_gradient.setdefault(it["gradient"], []).append(it["id"])
        self.assertEqual(sorted(by_gradient), sorted(GRADIENTS))
        for gradient, ids in by_gradient.items():
            self.assertEqual(len(ids), PILOT_PER_CLASS, gradient)
            self.assertEqual(ids, sorted(ids))
        again = pilot_corpus(load_corpus())
        self.assertEqual(
            [i["id"] for i in sub["items"]],
            [i["id"] for i in again["items"]],
        )


class TestBuildSchedule(unittest.TestCase):
    def test_api_mode_builds_api_substrates_only(self):
        schedule = build_schedule("study5-pilot-api")
        substrates = {it["meta"]["substrate"] for it in schedule}
        self.assertEqual(substrates, set(STUDY5_API_SUBSTRATES))
        self.assertTrue(
            all(not it["meta"].get("control") for it in schedule)
        )
        # pilot: 21 items x 5 templates x 3 paraphrase substrates
        # + 21 x 5 resample on haiku
        expected = 21 * 5 * 3 + 21 * 5
        self.assertEqual(len(schedule), expected)

    def test_local_mode_requires_box(self):
        with self.assertRaises(ValueError):
            build_schedule("study5-pilot-local")
        schedule = build_schedule("study5-pilot-local", box="cuda")
        substrates = {it["meta"]["substrate"] for it in schedule}
        self.assertEqual(substrates, {"local_20b_cuda"})
        warmups = [
            it for it in schedule if it["meta"].get("control") == "warmup"
        ]
        self.assertEqual(len(warmups), 1)
        # warmup + 21x5 paraphrase + 21x5 resample
        self.assertEqual(len(schedule), 1 + 21 * 5 + 21 * 5)

    def test_metal_box_gets_qwen_paraphrase_only(self):
        schedule = build_schedule("study5-full-local", box="metal")
        substrates = {it["meta"]["substrate"] for it in schedule}
        self.assertEqual(substrates, {"local_qwen_metal"})
        arms = {it["meta"]["arm"] for it in schedule}
        self.assertEqual(arms, {"paraphrase"})
        self.assertEqual(len(schedule), 1 + 150 * 5)


class TestDryRun(unittest.TestCase):
    def test_pilot_api_dry_run_writes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "harness.runner",
                    "--mode", "study5-pilot-api",
                    "--window", "pilot",
                    "--out", tmp,
                    "--dry-run",
                ],
                capture_output=True, text=True,
                cwd=os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                ),
                timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            manifests = [
                f for f in os.listdir(tmp)
                if f.endswith(".dryrun.manifest.json")
            ]
            self.assertEqual(len(manifests), 1)
            with open(os.path.join(tmp, manifests[0])) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["schema"], 5)
            self.assertTrue(manifest["pilot"])
            self.assertFalse(manifest["corpus_frozen"])
            self.assertEqual(manifest["corpus_n_total"], 150)
            self.assertEqual(manifest["items_in_run"], 21)
            self.assertEqual(
                sorted(manifest["substrates"]),
                sorted(STUDY5_API_SUBSTRATES),
            )
            self.assertIn("corpus_sha256", manifest)
            self.assertEqual(manifest["resample"]["n"], 5)


if __name__ == "__main__":
    unittest.main()

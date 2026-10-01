"""Study-5 held-out natural arm (PROTOCOL section 7): the committed corpus
is exactly what the declared procedure produces from the committed
snapshots, every item is labeled under the arm's rules, the schedule and
runner modes carry the registered call counts, and the primary corpus
schedule is byte-for-byte untouched."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

from harness.runner import (
    MODES,
    STUDY5_LOCAL_MODES,
    STUDY5_NATURAL_MODES,
    build_schedule,
    study5_run_settings,
)
from harness.study5_fixtures import (
    GRADIENTS,
    NATURAL_CORPUS_PATH,
    NATURAL_ID_PREFIX,
    SCHEMA_KEYS,
    load_corpus,
    validate_corpus,
    validate_natural_corpus,
)
from harness.study5_natural import (
    ID_PREFIX,
    NATURAL_DIR,
    SAMPLE_PER_SOURCE,
    SOURCES,
    check_corpus,
    draw,
    eligible_rows,
    load_snapshot,
    render_document,
    seed_from_sha,
    snapshot_path,
    source_by_key,
)
from harness.study5_schedule import (
    STUDY5_NATURAL_API_SUBSTRATES,
    build_study5_items,
    schedule_digest,
)

# The primary corpus full-schedule digest as shipped at 3060ec1 — the
# natural arm must not move it (recorded 2026-09-09 before the arm's
# code landed; re-derived here from the live builder).
PRIMARY_FULL_DIGEST = (
    "b5fdb6a5018bb9274f5f9011d3baf2e73cd250869d07164fc985385ff8aa58b3"
)
PRIMARY_FULL_CALLS = 5252

NATURAL_N = SAMPLE_PER_SOURCE * len(SOURCES)
NATURAL_API_CALLS = NATURAL_N * 5 * 2 + NATURAL_N * 5   # 750
NATURAL_CUDA_CALLS = 1 + NATURAL_N * 5 + NATURAL_N * 5  # 501
NATURAL_METAL_CALLS = 1 + NATURAL_N * 5                  # 251


class TestSnapshots(unittest.TestCase):
    def test_snapshots_committed_and_hashed_in_meta(self):
        corpus = load_corpus(NATURAL_CORPUS_PATH)
        declared = {s["key"]: s for s in corpus["meta"]["sources"]}
        for source in SOURCES:
            path = snapshot_path(source)
            self.assertTrue(path.exists(), path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(declared[source["key"]]["snapshot_sha256"], sha)
            snapshot, sha_again = load_snapshot(source)
            self.assertEqual(sha, sha_again)
            self.assertEqual(
                declared[source["key"]]["snapshot_rows"], len(snapshot["rows"])
            )

    def test_eligibility_rules_are_mechanical(self):
        austin = source_by_key("austin")
        rows = [
            {"stock_number": "1", "financial_name": "A", "unit_cost": "0",
             "total_on_hand": "3"},                       # zero cost
            {"stock_number": "2", "financial_name": "B", "unit_cost": "5.5"},
            {"stock_number": "3", "financial_name": "C", "unit_cost": "5.5",
             "total_on_hand": "0"},                       # zero stock OK
            {"stock_number": "4", "financial_name": " ", "unit_cost": "5.5",
             "total_on_hand": "1"},                       # blank name
        ]
        self.assertEqual(
            [r["stock_number"] for r in eligible_rows(austin, rows)], ["3"]
        )
        montgomery = source_by_key("montgomery")
        rows = [
            {"code": "1", "description": "X", "price": "9.99",
             "totalinventory": "0"},                      # not stocked
            {"code": "2", "description": "Y", "price": "9.99",
             "totalinventory": "12"},
            {"code": "3", "description": "Z", "price": "0",
             "totalinventory": "12"},                     # zero price
        ]
        self.assertEqual(
            [r["code"] for r in eligible_rows(montgomery, rows)], ["2"]
        )

    def test_draw_is_seeded_by_the_snapshot_hash(self):
        for source in SOURCES:
            snapshot, sha = load_snapshot(source)
            first = draw(source, snapshot["rows"], sha)
            again = draw(source, snapshot["rows"], sha)
            self.assertEqual(
                [r[source["row_key"]] for r in first],
                [r[source["row_key"]] for r in again],
            )
            self.assertEqual(len(first), SAMPLE_PER_SOURCE)
            other = draw(source, snapshot["rows"], "f" * 64)
            self.assertNotEqual(
                [r[source["row_key"]] for r in first],
                [r[source["row_key"]] for r in other],
            )
        self.assertEqual(seed_from_sha("00" * 32), 0)

    def test_render_uses_display_names_and_drops_bookkeeping(self):
        austin = source_by_key("austin")
        row = {
            "financial_name": " Widget ", "common_name": "Widget",
            "unit_cost": "1.5", "total_on_hand": "2", "id": "abc",
            "published_date": "2026-01-01", "modified_date": "x",
        }
        doc = render_document(austin, row)
        self.assertEqual(
            doc,
            "Financial Name: Widget\nCommon Name: Widget\nUnit Cost: 1.5\n"
            "Total On Hand: 2",
        )
        self.assertNotIn("abc", doc)
        self.assertNotIn("2026-01-01", doc)


class TestCommittedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_corpus(NATURAL_CORPUS_PATH)
        cls.primary = load_corpus()

    def test_reproducible_from_snapshots(self):
        self.assertEqual(check_corpus(self.corpus), [])

    def test_validators_clean(self):
        self.assertEqual(validate_corpus(self.corpus), [])
        self.assertEqual(validate_natural_corpus(self.corpus, self.primary), [])

    def test_shape(self):
        items = self.corpus["items"]
        self.assertEqual(len(items), NATURAL_N)
        self.assertEqual(
            [it["id"] for it in items],
            [f"{ID_PREFIX}{i:03d}" for i in range(1, NATURAL_N + 1)],
        )
        self.assertEqual(ID_PREFIX, NATURAL_ID_PREFIX)
        per_source = {}
        for it in items:
            per_source[it["source"]["dataset"]] = (
                per_source.get(it["source"]["dataset"], 0) + 1
            )
        self.assertEqual(
            per_source, {s["dataset"]: SAMPLE_PER_SOURCE for s in SOURCES}
        )

    def test_every_item_labeled(self):
        for it in self.corpus["items"]:
            self.assertIn(it["gradient"], GRADIENTS, it["id"])
            self.assertIn(it["target_field"], SCHEMA_KEYS, it["id"])
            self.assertTrue(it["ground_truth"]["item_name"].strip(), it["id"])
            self.assertIsNotNone(it["ground_truth"]["unit_price"], it["id"])
            self.assertIsNotNone(it["ground_truth"]["quantity_in_stock"], it["id"])
            self.assertTrue(it["rationale"].strip() or it["gradient"] == "clean")

    def test_templates_identical_to_primary(self):
        self.assertEqual(
            self.corpus["meta"]["instruction_templates"],
            self.primary["meta"]["instruction_templates"],
        )

    def test_unfrozen_until_the_freeze_commit(self):
        self.assertFalse(self.corpus["meta"]["frozen"])

    def test_second_labeler_sheet_lists_the_declared_sample(self):
        sheet = (NATURAL_DIR / "SECOND-LABELER-SHEET.md").read_text()
        by_source = {}
        for it in self.corpus["items"]:
            by_source.setdefault(it["source"]["dataset"], []).append(it)
        expected = []
        for dataset in sorted(by_source):
            expected += by_source[dataset][:10]
        self.assertEqual(len(expected), 20)
        for it in expected:
            self.assertIn(f"## {it['id']}", sheet)
            self.assertIn(it["document"], sheet)
        self.assertNotIn("ground_truth", sheet)
        self.assertNotIn("rationale", sheet)


class TestNaturalSchedule(unittest.TestCase):
    def test_primary_schedule_untouched(self):
        schedule = build_study5_items()
        self.assertEqual(len(schedule), PRIMARY_FULL_CALLS)
        self.assertEqual(schedule_digest(schedule), PRIMARY_FULL_DIGEST)
        self.assertFalse(any("source" in it["meta"] for it in schedule))

    def test_registered_counts(self):
        natural = load_corpus(NATURAL_CORPUS_PATH)
        api = build_study5_items(natural, substrates=STUDY5_NATURAL_API_SUBSTRATES)
        self.assertEqual(len(api), NATURAL_API_CALLS)
        self.assertEqual(
            {it["meta"]["substrate"] for it in api},
            set(STUDY5_NATURAL_API_SUBSTRATES),
        )
        self.assertTrue(all("source" in it["meta"] for it in api))
        cuda = build_study5_items(natural, substrates=("local_20b_cuda",))
        self.assertEqual(len(cuda), NATURAL_CUDA_CALLS)
        metal = build_study5_items(natural, substrates=("local_qwen_metal",))
        self.assertEqual(len(metal), NATURAL_METAL_CALLS)

    def test_runner_modes(self):
        for mode in STUDY5_NATURAL_MODES:
            self.assertIn(mode, MODES)
        self.assertIn("study5-natural-local", STUDY5_LOCAL_MODES)
        self.assertEqual(
            study5_run_settings("study5-natural-local"), {"concurrency": 1}
        )
        self.assertEqual(study5_run_settings("study5-natural-api"), {})
        api = build_schedule("study5-natural-api")
        self.assertEqual(len(api), NATURAL_API_CALLS)
        cuda = build_schedule("study5-natural-local", box="cuda")
        self.assertEqual(len(cuda), NATURAL_CUDA_CALLS)
        metal = build_schedule("study5-natural-local", box="metal")
        self.assertEqual(len(metal), NATURAL_METAL_CALLS)

    def test_natural_api_dry_run_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "harness.runner",
                    "--mode", "study5-natural-api",
                    "--window", "peak",
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
            self.assertIn(f"DRY RUN: {NATURAL_API_CALLS} calls", proc.stdout)
            manifests = [
                f for f in os.listdir(tmp)
                if f.endswith(".dryrun.manifest.json")
            ]
            self.assertEqual(len(manifests), 1)
            with open(os.path.join(tmp, manifests[0])) as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["schema"], 5)
            self.assertEqual(manifest["corpus_arm"], "natural")
            self.assertEqual(manifest["corpus_file"], "natural_corpus.json")
            self.assertFalse(manifest["pilot"])
            self.assertFalse(manifest["corpus_frozen"])
            self.assertEqual(manifest["corpus_n_total"], NATURAL_N)
            self.assertEqual(manifest["items_in_run"], NATURAL_N)
            self.assertEqual(
                sorted(manifest["substrates"]),
                sorted(STUDY5_NATURAL_API_SUBSTRATES),
            )
            sha = hashlib.sha256(NATURAL_CORPUS_PATH.read_bytes()).hexdigest()
            self.assertEqual(manifest["corpus_sha256"], sha)


if __name__ == "__main__":
    unittest.main()

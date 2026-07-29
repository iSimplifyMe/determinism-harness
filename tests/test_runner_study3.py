"""Study-3 runner integration: local grid scheduling and engine dispatch.

Stub/fake planes only — no network, no Ollama; must pass on a stdlib-only
interpreter. The local plane never constructs an SDK client, so injecting
plane_clients keeps everything offline.
"""
import io
import json
import os
import tempfile
import unittest

from harness.config import (
    LOCAL_MODELS,
    REPEATS_FULL,
    REPEATS_STUDY3_PILOT,
    local_on_arm,
    local_pinned_arm,
)
from harness.planes import LocalPlane
from harness.runner import (
    Engine,
    build_schedule,
    build_warmup_items,
    study3_run_settings,
)


class TestStudy3Schedule(unittest.TestCase):
    def test_full_counts_metal(self):
        items = build_schedule("study3-full", box="metal")
        cells = {it["cell"] for it in items}
        # 4 models x 4 tasks x 2 sampling arms, thinking pinned
        self.assertEqual(len(cells), 32)
        self.assertEqual(len(items), 32 * REPEATS_FULL)

    def test_full_counts_cuda_gpt_oss_only(self):
        items = build_schedule("study3-full", box="cuda")
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 8)  # 1 model x 4 tasks x 2 sampling
        for it in items:
            self.assertEqual(it["meta"]["model"], "gpt-oss-20b")
            self.assertEqual(it["meta"]["hardware"], "cuda")

    def test_pilot_adds_q3_on_arm_cells(self):
        metal = {it["cell"] for it in build_schedule("study3-pilot", box="metal")}
        self.assertEqual(len(metal), 36)  # 32 core + 4 thinking-on sj cells
        cuda = {it["cell"] for it in build_schedule("study3-pilot", box="cuda")}
        self.assertEqual(len(cuda), 9)  # 8 core + 1
        items = build_schedule("study3-pilot", box="cuda")
        self.assertEqual(len(items), 9 * REPEATS_STUDY3_PILOT)

    def test_core_grid_pins_low_arm_per_family(self):
        for it in build_schedule("study3-full", box="metal"):
            body = json.loads(it["payload"])
            model_key = it["meta"]["model"]
            arm = it["meta"]["thinking"]
            self.assertEqual(arm, local_pinned_arm(LOCAL_MODELS[model_key]))
            if model_key.startswith("qwen"):
                self.assertIs(body["think"], False)
            else:
                self.assertEqual(body["think"], "low")

    def test_sampling_arms_encode_temperature_and_seed(self):
        seen = {"greedy": None, "temp07": None}
        for it in build_schedule("study3-full", box="cuda"):
            body = json.loads(it["payload"])
            sampling = it["meta"]["sampling"]
            if seen[sampling] is None:
                seen[sampling] = body["options"]
        self.assertEqual(seen["greedy"]["temperature"], 0)
        self.assertEqual(seen["temp07"]["temperature"], 0.7)
        self.assertIn("seed", seen["greedy"])
        # the positive-control analog must stay unseeded: a seeded local
        # engine reproduces temp-0.7 byte-for-byte (pilot #1, 2026-07-29)
        self.assertNotIn("seed", seen["temp07"])
        self.assertIn("num_predict", seen["greedy"])

    def test_q3_thinking_mode_is_on_arm_sj_greedy_only(self):
        items = build_schedule("study3-q3-thinking", box="metal")
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 4)
        for it in items:
            meta = it["meta"]
            self.assertEqual(meta["task"], "structured_json")
            self.assertEqual(meta["sampling"], "greedy")
            self.assertEqual(
                meta["thinking"], local_on_arm(LOCAL_MODELS[meta["model"]])
            )

    def test_q2_concurrency_is_qwen_trio_on_metal(self):
        items = build_schedule("study3-q2-concurrency", box="metal")
        cells = {it["cell"] for it in items}
        self.assertEqual(len(cells), 12)  # 3 qwen x 4 tasks
        for it in items:
            self.assertTrue(it["cell"].endswith("|c4"), it["cell"])
            self.assertEqual(it["meta"]["concurrency"], 4)
            self.assertTrue(it["meta"]["model"].startswith("qwen"))
        with self.assertRaises(ValueError):
            build_schedule("study3-q2-concurrency", box="cuda")

    def test_box_is_required_for_study3(self):
        with self.assertRaises(ValueError):
            build_schedule("study3-full")

    def test_payloads_are_canonical_local_bytes(self):
        from harness.request_builder import sha256_hex

        for it in build_schedule("study3-pilot", box="cuda")[:20]:
            self.assertEqual(it["plane"], "local")
            self.assertIsInstance(it["payload"], bytes)
            body = json.loads(it["payload"])
            self.assertIs(body["stream"], False)
            self.assertEqual(body["model"], LOCAL_MODELS["gpt-oss-20b"]["tag"])
            self.assertEqual(it["sha"], sha256_hex(it["payload"]))

    def test_planned_sha_constant_within_cell(self):
        by_cell = {}
        for it in build_schedule("study3-pilot", box="metal"):
            by_cell.setdefault(it["cell"], set()).add(it["sha"])
        for cell, shas in by_cell.items():
            self.assertEqual(len(shas), 1, cell)

    def test_warmup_one_item_per_model(self):
        schedule = build_schedule("study3-pilot", box="metal")
        warmups = build_warmup_items(schedule)
        self.assertEqual(len(warmups), 4)
        for it in warmups:
            self.assertEqual(it["meta"]["control"], "warmup")
            self.assertEqual(it["plane"], "local")
            self.assertTrue(it["cell"].startswith("warmup|"))


class TestStudy3RunSettings(unittest.TestCase):
    def test_core_modes_enforce_single_flight(self):
        for mode in ("study3-pilot", "study3-full", "study3-q3-thinking"):
            settings = study3_run_settings(mode)
            self.assertEqual(settings["concurrency"], 1, mode)
            self.assertFalse(settings["allow_same_cell_concurrency"], mode)

    def test_q2_mode_runs_same_cell_parallel(self):
        settings = study3_run_settings("study3-q2-concurrency")
        self.assertEqual(settings["concurrency"], 4)
        self.assertTrue(settings["allow_same_cell_concurrency"])

    def test_non_study3_modes_have_no_settings(self):
        self.assertIsNone(study3_run_settings("study2-full"))


class TestEngineSameCellConcurrency(unittest.TestCase):
    def _items(self):
        return [
            {"cell": "x", "meta": {}, "plane": "local", "payload": b"{}",
             "sha": "s", "model_id": "m", "repeat": r}
            for r in range(2)
        ]

    def test_default_blocks_same_cell_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                self._items(), os.path.join(tmp, "o.jsonl"), 2, 1,
                plane_clients={"local": object()},
            )
            self.assertEqual(engine._next_index(), 0)
            self.assertEqual(engine._next_index(), -1)  # sibling in flight
            engine.out.close()

    def test_flag_allows_same_cell_in_flight(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = Engine(
                self._items(), os.path.join(tmp, "o.jsonl"), 2, 1,
                plane_clients={"local": object()},
                allow_same_cell_concurrency=True,
            )
            self.assertEqual(engine._next_index(), 0)
            self.assertEqual(engine._next_index(), 1)
            engine.out.close()


def _ok_payload():
    return {
        "model": "gpt-oss:20b",
        "message": {"role": "assistant", "content": "out"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 7,
        "eval_duration": 1000,
    }


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestEngineLocalDispatch(unittest.TestCase):
    def test_records_carry_schema3_and_hardware(self):
        def opener(request, timeout=None):
            return _FakeResponse(json.dumps(_ok_payload()).encode())

        items = build_schedule("study3-pilot", box="cuda")[:3]
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o.jsonl")
            engine = Engine(
                items, out, 1, 7,
                run_info={"schema": 3, "box": "cuda", "mode": "study3-pilot"},
                plane_clients={"local": LocalPlane(opener=opener, name="local_cuda")},
            )
            summary = engine.run()
            self.assertEqual(summary["done"], 3)
            self.assertEqual(summary["failures"], 0)
            with open(out, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
        for record in records:
            self.assertEqual(record["schema"], 3)
            self.assertEqual(record["box"], "cuda")
            self.assertEqual(record["plane"], "local")
            self.assertEqual(record["meta_hardware"], "cuda")
            self.assertTrue(record["ok"])
            self.assertEqual(record["wire_sha256"], record["request_sha256"])


if __name__ == "__main__":
    unittest.main()

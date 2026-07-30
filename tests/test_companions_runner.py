"""Companion modes (FOLLOWUP-COMPANIONS.md): reload-churn A/B + margins.

Runner-side units: LocalPlane.unload, fixed schedules, run settings, engine
dispatch hooks. Stub openers only — no network, stdlib-only interpreter.
"""
import io
import json
import os
import sys
import tempfile
import unittest

from harness.config import (
    CHURN_AB,
    GPT_OSS_120B,
    LOCAL_KEEP_ALIVE,
    LOCAL_MODELS,
    LOCAL_SAMPLING,
    MARGINS_BATTERY,
    MARGINS_LOGPROB_FIELDS,
)
from harness.planes import LocalPlane
from harness.request_builder import canonical_local_body, sha256_hex
from harness.runner import (
    COMPANION_MODES,
    Engine,
    FIXED_SCHEDULE_MODES,
    build_schedule,
    schedule_digest,
    study3_run_settings,
)
from harness.tasks import TASKS


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _ScriptedOpener:
    """Routes requests by URL path; records every call for assertions."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append(request)
        payload = self.handler(request)
        return _FakeResponse(json.dumps(payload).encode())


TAG = LOCAL_MODELS["gpt-oss-20b"]["tag"]


class TestLocalPlaneUnload(unittest.TestCase):
    def test_unload_posts_canonical_body_and_confirms_via_ps(self):
        def handler(request):
            if request.full_url.endswith("/api/chat"):
                return {"done": True, "done_reason": "unload"}
            if request.full_url.endswith("/api/ps"):
                return {"models": []}
            raise AssertionError(request.full_url)

        opener = _ScriptedOpener(handler)
        plane = LocalPlane(opener=opener, name="local_metal")
        result = plane.unload(TAG, wait_timeout=5)
        self.assertTrue(result["unloaded"])
        self.assertIsInstance(result["wait_ms"], int)
        self.assertGreaterEqual(result["ps_polls"], 1)
        chat = opener.calls[0]
        self.assertTrue(chat.full_url.endswith("/api/chat"))
        body = json.loads(chat.data)
        self.assertEqual(
            body, {"keep_alive": 0, "messages": [], "model": TAG}
        )

    def test_unload_times_out_when_model_stays_resident(self):
        def handler(request):
            if request.full_url.endswith("/api/chat"):
                return {"done": True}
            return {"models": [{"name": TAG}]}

        plane = LocalPlane(opener=_ScriptedOpener(handler))
        result = plane.unload(TAG, wait_timeout=0.3, poll_interval=0.05)
        self.assertFalse(result["unloaded"])

    def test_model_resident(self):
        def handler(request):
            return {"models": [{"name": TAG}]}

        plane = LocalPlane(opener=_ScriptedOpener(handler))
        self.assertTrue(plane.model_resident(TAG))
        self.assertFalse(plane.model_resident("other:1b"))


class TestChurnSchedule(unittest.TestCase):
    def test_counts_arms_and_mini_blocks(self):
        items = build_schedule("study3-churn-ab", box="metal")
        n = CHURN_AB["n_per_arm"]
        self.assertEqual(len(items), 1 + 2 * n)
        self.assertEqual(items[0]["meta"].get("control"), "warmup")
        measured = items[1:]
        arms = [it["meta"]["arm"] for it in measured]
        self.assertEqual(arms.count("blocked"), n)
        self.assertEqual(arms.count("churn"), n)
        # alternating mini-blocks of CHURN_AB["mini_block"], blocked first
        block = CHURN_AB["mini_block"]
        for i, arm in enumerate(arms):
            expected = "blocked" if (i // block) % 2 == 0 else "churn"
            self.assertEqual(arm, expected, f"position {i}")

    def test_measured_bodies_byte_identical_across_arms_and_match_frozen(self):
        items = build_schedule("study3-churn-ab", box="metal")
        measured = items[1:]
        shas = {it["sha"] for it in measured}
        self.assertEqual(len(shas), 1)
        frozen = canonical_local_body(
            TAG,
            TASKS["open_generation"]["prompt"],
            "effort_low",
            options=LOCAL_SAMPLING["greedy"],
            keep_alive=LOCAL_KEEP_ALIVE,
        )
        self.assertEqual(measured[0]["payload"], frozen)
        self.assertEqual(shas.pop(), sha256_hex(frozen))

    def test_churn_items_flag_pre_unload_blocked_do_not(self):
        items = build_schedule("study3-churn-ab", box="cuda")
        for it in items[1:]:
            if it["meta"]["arm"] == "churn":
                self.assertTrue(it.get("pre_unload"))
                self.assertTrue(it["cell"].endswith("|arm=churn"))
            else:
                self.assertFalse(it.get("pre_unload"))
                self.assertTrue(it["cell"].endswith("|arm=blocked"))
            self.assertEqual(it["meta"]["hardware"], "cuda")

    def test_repeats_override_scales_arms(self):
        items = build_schedule("study3-churn-ab", box="metal", repeats=4)
        arms = [it["meta"]["arm"] for it in items[1:]]
        self.assertEqual(arms.count("blocked"), 4)
        self.assertEqual(arms.count("churn"), 4)

    def test_unknown_box_rejected(self):
        with self.assertRaises(ValueError):
            build_schedule("study3-churn-ab", box="tpu")
        with self.assertRaises(ValueError):
            build_schedule("study3-churn-ab")


class TestMarginsSchedule(unittest.TestCase):
    def test_metal_battery_counts_order_and_warmups(self):
        items = build_schedule("study3-margins", box="metal")
        expected_measured = sum(c["n"] for c in MARGINS_BATTERY["metal"])
        warmups = [it for it in items if it["meta"].get("control") == "warmup"]
        self.assertEqual(len(items), expected_measured + len(warmups))
        self.assertEqual(len(warmups), 3)  # 20b, vl, 120b
        # 120b block is last; block heads are warmups
        model_seq = []
        for it in items:
            m = it["meta"]["model"]
            if not model_seq or model_seq[-1] != m:
                model_seq.append(m)
        self.assertEqual(model_seq[-1], GPT_OSS_120B["key"])
        self.assertEqual(len(model_seq), len(set(model_seq)))
        starts = [0] + [
            i for i in range(1, len(items))
            if items[i]["meta"]["model"] != items[i - 1]["meta"]["model"]
        ]
        for start in starts:
            self.assertEqual(items[start]["meta"].get("control"), "warmup")

    def test_cells_suffixed_and_bodies_carry_logprob_fields(self):
        items = build_schedule("study3-margins", box="metal")
        for it in items:
            if it["meta"].get("control") == "warmup":
                continue
            self.assertTrue(it["cell"].endswith("|logprobs"), it["cell"])
            self.assertTrue(it.get("capture_logprobs"))
            body = json.loads(it["payload"])
            for key, value in MARGINS_LOGPROB_FIELDS.items():
                self.assertEqual(body[key], value)
            self.assertEqual(body["options"]["temperature"], 0)
            self.assertIn("seed", body["options"])

    def test_effort_high_cell_present_on_metal_only(self):
        metal = {it["cell"] for it in build_schedule("study3-margins", box="metal")}
        self.assertIn(
            "gpt-oss-120b|structured_json|greedy|effort_high|logprobs", metal
        )
        cuda_items = build_schedule("study3-margins", box="cuda")
        cuda = {it["cell"] for it in cuda_items}
        self.assertEqual(
            cuda,
            {
                "warmup|gpt-oss-20b",
                "gpt-oss-20b|structured_json|greedy|effort_low|logprobs",
                "gpt-oss-20b|open_generation|greedy|effort_low|logprobs",
            },
        )

    def test_repeats_override_replaces_battery_n(self):
        items = build_schedule("study3-margins", box="cuda", repeats=2)
        measured = [
            it for it in items if it["meta"].get("control") != "warmup"
        ]
        self.assertEqual(len(measured), 2 * len(MARGINS_BATTERY["cuda"]))


class TestCompanionRunSettings(unittest.TestCase):
    def test_membership_and_single_flight(self):
        for mode in ("study3-churn-ab", "study3-margins"):
            self.assertIn(mode, COMPANION_MODES)
            self.assertIn(mode, FIXED_SCHEDULE_MODES)
            settings = study3_run_settings(mode)
            self.assertEqual(settings["concurrency"], 1, mode)
            self.assertFalse(settings["allow_same_cell_concurrency"], mode)


def _ok_payload(with_logprobs=False):
    payload = {
        "model": TAG,
        "message": {"role": "assistant", "content": "out"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 3,
        "eval_duration": 1000,
        "load_duration": 42,
    }
    if with_logprobs:
        payload["logprobs"] = [
            {
                "token": "o",
                "logprob": -0.01,
                "top_logprobs": [
                    {"token": "o", "logprob": -0.01},
                    {"token": "x", "logprob": -4.2},
                ],
            },
            {
                "token": "ut",
                "logprob": -0.5,
                "top_logprobs": [
                    {"token": "ut", "logprob": -0.5},
                    {"token": "n", "logprob": -0.9},
                ],
            },
        ]
    return payload


class TestEngineCompanionDispatch(unittest.TestCase):
    def _run(self, items, handler):
        opener = _ScriptedOpener(handler)
        plane = LocalPlane(opener=opener, name="local_metal")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "o.jsonl")
            engine = Engine(
                items, out, 1, 7,
                run_info={"schema": 3, "box": "metal", "mode": "test"},
                plane_clients={"local": plane},
            )
            summary = engine.run()
            with open(out, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh]
        return summary, records, opener

    def test_churn_item_unloads_confirms_and_records(self):
        state = {"resident": True}

        def handler(request):
            if request.full_url.endswith("/api/ps"):
                return {"models": [] if not state["resident"] else [{"name": TAG}]}
            body = json.loads(request.data)
            if body.get("messages") == []:
                state["resident"] = False
                return {"done": True, "done_reason": "unload"}
            state["resident"] = True
            return _ok_payload()

        items = build_schedule("study3-churn-ab", box="metal", repeats=1)
        # keep just the warmup, one blocked, one churn
        summary, records, opener = self._run(items, handler)
        self.assertEqual(summary["failures"], 0)
        by_arm = {r.get("meta_arm"): r for r in records if r.get("meta_arm")}
        churn = by_arm["churn"]
        self.assertTrue(churn["pre_unload_confirmed"])
        self.assertIsInstance(churn["unload_wait_ms"], int)
        blocked = by_arm["blocked"]
        self.assertNotIn("pre_unload_confirmed", blocked)
        # the unload side-call happened: an empty-messages chat POST exists
        unload_posts = [
            c for c in opener.calls
            if c.full_url.endswith("/api/chat")
            and json.loads(c.data).get("messages") == []
        ]
        self.assertEqual(len(unload_posts), 1)

    def test_margins_item_captures_compact_margins(self):
        def handler(request):
            if request.full_url.endswith("/api/ps"):
                return {"models": []}
            body = json.loads(request.data)
            return _ok_payload(with_logprobs=bool(body.get("logprobs")))

        items = build_schedule("study3-margins", box="cuda", repeats=1)
        summary, records, _ = self._run(items, handler)
        self.assertEqual(summary["failures"], 0)
        measured = [r for r in records if r.get("meta_control") != "warmup"]
        for record in measured:
            margins = record["logprob_margins"]
            self.assertEqual(margins["n_tokens"], 2)
            self.assertEqual(len(margins["tokens"]), 2)
            self.assertEqual(margins["chosen_not_top1"], 0)
        warmup = [r for r in records if r.get("meta_control") == "warmup"][0]
        self.assertNotIn("logprob_margins", warmup)


class TestCompanionDryRunManifest(unittest.TestCase):
    def _dry_run(self, mode, box):
        from harness.runner import main as runner_main

        with tempfile.TemporaryDirectory() as tmp:
            argv = sys.argv
            sys.argv = [
                "runner", "--mode", mode, "--window", "local",
                "--box", box, "--dry-run", "--out", tmp,
            ]
            try:
                rc = runner_main()
            finally:
                sys.argv = argv
            self.assertEqual(rc, 0)
            names = [n for n in os.listdir(tmp) if n.endswith("manifest.json")]
            with open(os.path.join(tmp, names[0]), encoding="utf-8") as fh:
                return json.load(fh)

    def test_churn_manifest_fixed_unshuffled_exploratory(self):
        manifest = self._dry_run("study3-churn-ab", "metal")
        self.assertTrue(manifest["schedule_fixed"])
        self.assertTrue(manifest["exploratory"])
        self.assertEqual(manifest["companion_plan"], "FOLLOWUP-COMPANIONS.md")
        self.assertEqual(manifest["schema"], 3)
        self.assertEqual(manifest["n_items"], 1 + 2 * CHURN_AB["n_per_arm"])
        self.assertEqual(manifest["warmup_items"], 1)
        # fixed = the manifest digest equals the unshuffled build's digest
        self.assertEqual(
            manifest["schedule_sha256"],
            schedule_digest(build_schedule("study3-churn-ab", box="metal")),
        )

    def test_margins_manifest_counts(self):
        manifest = self._dry_run("study3-margins", "metal")
        self.assertEqual(manifest["warmup_items"], 3)
        self.assertEqual(
            manifest["n_items"],
            3 + sum(c["n"] for c in MARGINS_BATTERY["metal"]),
        )
        self.assertEqual(
            manifest["schedule_sha256"],
            schedule_digest(build_schedule("study3-margins", box="metal")),
        )


if __name__ == "__main__":
    unittest.main()

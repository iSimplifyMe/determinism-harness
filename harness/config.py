"""Frozen experimental configuration.

Everything here is a pre-registered quantity once PREREGISTRATION.md is
frozen — change nothing between the freeze commit and the confirmatory runs
without recording a deviation.

Profile IDs were verified live on 2026-07-27 against account 024033896674
(us-east-1): `aws bedrock list-foundation-models` reports
inferenceTypesSupported=["INFERENCE_PROFILE"] for all three target models —
there is no single-region ON_DEMAND path — and `list-inference-profiles`
reports both the `us.` and `global.` system-defined profiles ACTIVE for all
three. Q2 is therefore US-bounded vs global routing, not profile vs bare ID.
See evidence/inference-profiles.json.

Thinking arms: adaptive vs disabled exists only on the Claude 5 family.
Haiku 4.5 predates adaptive thinking (its thinking form is budget_tokens,
a different manipulation) and rejects the effort parameter, so it runs a
single arm with both fields omitted — its roles are dated-version anchor
(the only model where a silent point-version roll is detectable from the
ID) and positive-control host (the only grid model that still accepts
sampling parameters).

Effort is pinned at "medium" on the 5-family arms because thinking:disabled
is illegal above effort "high" on Opus 5 — a fixed low-enough effort is the
only way to keep thinking a clean two-level factor.
"""

REGION = "us-east-1"

REPEATS_FULL = 100   # confirmatory, per cell per window; pilot may revise (prereg)
REPEATS_PILOT = 20   # single-window pilot, per spec section 5
REPEATS_Q4 = 25      # study-2 Q4 sparse ladder, reduced-n (prereg v2)

PROFILES = ("us", "global")

MODELS = {
    "opus-5": {
        "profiles": {
            "us": "us.anthropic.claude-opus-5",
            "global": "global.anthropic.claude-opus-5",
        },
        "family": "claude-5",
        "thinking_arms": ("adaptive", "disabled"),
        "effort": "medium",
        "max_tokens": 16000,
        "supports_sampling": False,
        "dated_id": False,
    },
    "sonnet-5": {
        "profiles": {
            "us": "us.anthropic.claude-sonnet-5",
            "global": "global.anthropic.claude-sonnet-5",
        },
        "family": "claude-5",
        "thinking_arms": ("adaptive", "disabled"),
        "effort": "medium",
        "max_tokens": 16000,
        "supports_sampling": False,
        "dated_id": False,
    },
    "haiku-4-5": {
        "profiles": {
            "us": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "global": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        },
        "family": "claude-4x",
        "thinking_arms": ("none",),
        "effort": None,
        "max_tokens": 8192,
        "supports_sampling": True,
        "dated_id": True,
    },
}

# --- Study 2 (cross-plane attribution, PREREGISTRATION-v2 DRAFT) ---------
# Plane is the routing-analog factor; Bedrock is pinned to the `us.` profile
# (v2 section 4). The Messages planes (Claude Platform on AWS, first-party
# API) take bare model IDs with no provider prefix. Haiku keeps a dated ID on
# every plane so the version-drift anchor survives the plane factor — on the
# Messages planes that is the published dated full ID, not a constructed one.

PLANES = ("bedrock", "p_aws", "anthropic_api")

MESSAGES_MODEL_IDS = {
    "opus-5": "claude-opus-5",
    "sonnet-5": "claude-sonnet-5",
    "haiku-4-5": "claude-haiku-4-5-20251001",
}


def plane_model_id(plane, model_key):
    """Model ID to send for a (plane, model) pair. Bedrock: `us.` inference
    profile (study-2 pin). Messages planes: bare / dated-full ID."""
    if plane == "bedrock":
        return MODELS[model_key]["profiles"]["us"]
    if plane in ("p_aws", "anthropic_api"):
        return MESSAGES_MODEL_IDS[model_key]
    raise ValueError(f"unknown plane: {plane}")


# Study 2 Q3 (exploratory): streamed delivery on the divergence-prone tasks,
# adaptive thinking pinned; the non-streamed comparators are the main grid's
# own cells. Runs in a control window.
Q3_STREAMING = {
    "models": ("opus-5", "sonnet-5"),
    "tasks": ("structured_json", "open_generation"),
    "thinking": "adaptive",
    "repeats": REPEATS_FULL,
}

# Study 2 Q4 (exploratory): sparse input-length ladder on extraction,
# adaptive thinking pinned, reduced repeats, cost-bounded. Control window.
Q4_LENGTHS = {
    "models": ("opus-5", "sonnet-5"),
    "labels": ("1k", "10k", "50k"),
    "thinking": "adaptive",
    "repeats": REPEATS_Q4,
}

# Instrument-validity control: sampling parameters are removed on the Claude 5
# family, so the divergence-detection check runs on Haiku 4.5 at temperature
# 0.7 — a cross-model control, disclosed as such in the prereg.
POSITIVE_CONTROL = {
    "model": "haiku-4-5",
    "task": "open_generation",
    "profile": "us",
    "thinking": "none",
    "extra": {"temperature": 0.7},
    "repeats": 100,
}

# Follow-on: does effort level stabilize or destabilize the answer? Runs with
# adaptive thinking only, so all five levels are legal on Opus 5.
EFFORT_SWEEP = {
    "model": "opus-5",
    "tasks": ("classification", "open_generation"),
    "efforts": ("low", "medium", "high", "xhigh", "max"),
    "profile": "us",
    "thinking": "adaptive",
    "repeats": 100,
}

# Load windows, defined in UTC against US traffic patterns. Runs should be
# compressed (hours, not days) to shrink the undetectable-version-roll window
# on the undated 5-family IDs.
WINDOWS = {
    "low": "07:00-10:00 UTC (US night)",
    "mid": "00:00-03:00 UTC (US evening)",
    "peak": "15:00-19:00 UTC (US business morning/midday)",
    "pilot": "any single window, recorded in the manifest",
    "control": "any, recorded in the manifest",
    "local": "owner-approved window on the target box, recorded in the manifest",
}


def grid_cells():
    """Yield the factorial grid: model x task x profile x thinking arm."""
    from harness.tasks import TASKS

    for model_key, cfg in MODELS.items():
        for task_key in TASKS:
            for profile in PROFILES:
                for arm in cfg["thinking_arms"]:
                    yield {
                        "model": model_key,
                        "task": task_key,
                        "profile": profile,
                        "thinking": arm,
                    }


def cell_key(cell):
    return f'{cell["model"]}|{cell["task"]}|{cell["profile"]}|{cell["thinking"]}'


def grid_cells_study2():
    """Study-2 factorial: model x task x plane x thinking. Plane takes the
    slot profile held in study 1 (Bedrock pinned to `us.`, prereg v2 s2/s4)."""
    from harness.tasks import TASKS

    for model_key, cfg in MODELS.items():
        for task_key in TASKS:
            for plane in PLANES:
                for arm in cfg["thinking_arms"]:
                    yield {
                        "model": model_key,
                        "task": task_key,
                        "plane": plane,
                        "thinking": arm,
                    }


def cell_key2(cell):
    return f'{cell["model"]}|{cell["task"]}|{cell["plane"]}|{cell["thinking"]}'


# --- Study 3 (local open-model baseline, PREREGISTRATION-v3 DRAFT) --------
# Open weights on owned hardware — the control-ceiling rung. Model identity
# includes the quantization tag; the weights digest recorded per run is the
# drift control. Thinking arms are per-family (Qwen hybrid = bool think,
# gpt-oss = reasoning-effort level), listed LOW/OFF first: the core grid
# pins each model's first arm and Q3 adds the second on structured JSON, so
# the thinking factor lives in Q3 cells only (design doc section 4).
# Per-model think-field acceptance is pilot-verified on the pinned engine
# before freeze (qwen3-vl may prove non-hybrid and be struck from Q3).

LOCAL_BOXES = ("metal", "cuda")  # hardware arms; base URL supplied at run time

REPEATS_STUDY3_PILOT = 10

LOCAL_SEED = 42
LOCAL_KEEP_ALIVE = "10m"
LOCAL_NUM_PREDICT = 4096

LOCAL_SAMPLING = {
    "greedy": {
        "temperature": 0, "seed": LOCAL_SEED, "num_predict": LOCAL_NUM_PREDICT,
    },
    # UNSEEDED by design: a seeded local engine reproduces temp-0.7 sampling
    # byte-for-byte (cuda pilot #1, 2026-07-29), which defeats this arm's
    # purpose as the divergence-detection positive control. With no seed
    # field the engine draws one per call, so byte-identical requests must
    # produce varied output — the analog of study 1-2's temp-0.7 control.
    "temp07": {"temperature": 0.7, "num_predict": LOCAL_NUM_PREDICT},
}

LOCAL_MODELS = {
    "qwen3.5-122b": {
        "tag": "qwen3.5:122b-a10b",
        "arch": "moe",
        "thinking_arms": ("think_off", "think_on"),
        "boxes": ("metal",),
    },
    "qwen3.6-35b": {
        "tag": "qwen3.6:35b-a3b-q8_0",
        "arch": "moe",
        "thinking_arms": ("think_off", "think_on"),
        "boxes": ("metal",),
    },
    "qwen3-vl-32b": {
        "tag": "qwen3-vl:32b-instruct-q8_0",
        "arch": "dense",
        "thinking_arms": ("think_off", "think_on"),
        "boxes": ("metal",),
    },
    "gpt-oss-20b": {
        "tag": "gpt-oss:20b",
        "arch": "moe",
        "thinking_arms": ("effort_low", "effort_high"),
        "boxes": ("metal", "cuda"),
    },
}

# Q2 (concurrency): the registered MoE-vs-dense contrast — the Qwen trio on
# the Metal box at concurrency 4; the single-flight comparators are the core
# grid's own cells.
Q2_LOCAL_CONCURRENCY = {
    "models": ("qwen3.5-122b", "qwen3.6-35b", "qwen3-vl-32b"),
    "level": 4,
    "box": "metal",
}


def local_pinned_arm(model_cfg):
    """Core-grid thinking arm: the model's LOW/OFF level (listed first)."""
    return model_cfg["thinking_arms"][0]


def local_on_arm(model_cfg):
    """Q3 thinking arm: the model's ON/HIGH level (listed second)."""
    return model_cfg["thinking_arms"][1]


def local_models_for_box(box):
    if box not in LOCAL_BOXES:
        raise ValueError(f"unknown box: {box}")
    return {key: cfg for key, cfg in LOCAL_MODELS.items() if box in cfg["boxes"]}


def grid_cells_study3(box):
    """Study-3 core factorial for one box: model x task x sampling, thinking
    pinned at the model's LOW/OFF arm. Single-flight (Q1's registered
    condition); hardware rides in the cell meta, not the cell key, so Q4
    compares identical cells across boxes."""
    from harness.tasks import TASKS

    for model_key, cfg in local_models_for_box(box).items():
        for task_key in TASKS:
            for sampling in sorted(LOCAL_SAMPLING):
                yield {
                    "model": model_key,
                    "task": task_key,
                    "sampling": sampling,
                    "thinking": local_pinned_arm(cfg),
                    "hardware": box,
                }


def cell_key3(cell):
    return f'{cell["model"]}|{cell["task"]}|{cell["sampling"]}|{cell["thinking"]}'


# gpt-oss:120b — REGISTERED as a single-dedicated-window arm (owner decision
# 2026-07-29; criterion met: active community use of the open-weights
# release). Deliberately NOT in LOCAL_MODELS: loading its 65 GB evicts the
# production residents, so its cells exist only in the study3-120b-window
# mode and never join the core-grid modes.
GPT_OSS_120B = {
    "key": "gpt-oss-120b",
    "tag": "gpt-oss:120b",
    "arch": "moe",
    "thinking_arms": ("effort_low", "effort_high"),
    "box": "metal",
}

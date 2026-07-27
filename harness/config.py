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

"""Study-5 call schedule: items x instruction templates x substrates.

The disagreement generators (design record 2026-08-31; prereg v5 will
restate what is registered):

- paraphrase  — every instruction template once per item, deterministic
  request settings; disagreement is read ACROSS templates within an
  (item, substrate). The registered primary generator.
- resample    — one fixed template repeated under sampling, on the
  substrates that accept it (Haiku 4.5 temperature 0.7; local unseeded
  temp 0.7). The semantic-entropy prior-art baseline.
- cross-door  — Sonnet 1P vs Sonnet Bedrock on identical template+item:
  derived in analysis from the paraphrase calls, no extra calls.
- cross-model — Haiku vs Sonnet on 1P, also derived in analysis.

Payloads come from the same request builders as studies 1-3, so the
factor encoding matches the published corpus. Prompts are always
template + "\n\n" + document. Scheduling follows the study-3 lessons:
per-substrate blocks, deterministic order, warmup heads on local
substrates (control="warmup", excluded from analysis), single-flight.

Stdlib only.
"""
from harness.request_builder import (
    canonical_body,
    canonical_bytes,
    canonical_local_body,
    canonical_messages_params,
    sha256_hex,
)
from harness.study5_fixtures import TEMPLATE_IDS, load_corpus

PROMPT_JOINER = "\n\n"

# Repeats of the resample template per item (pilot-scale default; the
# confirmatory count is set by the power calc at prereg).
RESAMPLE_N = 5
RESAMPLE_TEMPLATE = "t1"
RESAMPLE_TEMPERATURE = 0.7

# Thinking/effort pinned everywhere (standing discipline). Haiku has no
# thinking field ("none" omits it); Sonnet runs disabled — study 5 is not
# a thinking study, and disabled is the deterministic-lean arm.
MESSAGES_MAX_TOKENS = 512

STUDY5_SUBSTRATES = {
    "haiku_1p": {
        "kind": "messages",
        "plane": "anthropic_api",
        "model_id": "claude-haiku-4-5-20251001",
        "model_cfg": {"max_tokens": MESSAGES_MAX_TOKENS, "effort": None},
        "thinking": "none",
        "arms": ("paraphrase", "resample"),
    },
    "sonnet_1p": {
        "kind": "messages",
        "plane": "anthropic_api",
        "model_id": "claude-sonnet-5",
        "model_cfg": {"max_tokens": MESSAGES_MAX_TOKENS, "effort": "medium"},
        "thinking": "disabled",
        "arms": ("paraphrase",),
    },
    "sonnet_bedrock": {
        "kind": "bedrock",
        "plane": "bedrock",
        # The us. inference profile — same id studies 1-2 invoked (config
        # MODELS["sonnet-5"]["profiles"]["us"]); Bedrock has no bare-model
        # on-demand for the 5-family.
        "model_id": "us.anthropic.claude-sonnet-5",
        "model_cfg": {"max_tokens": MESSAGES_MAX_TOKENS, "effort": "medium"},
        "thinking": "disabled",
        "arms": ("paraphrase",),
    },
    "local_20b_cuda": {
        "kind": "local",
        "plane": "local_cuda",
        "model_tag": "gpt-oss:20b",
        "options": {"temperature": 0, "seed": 42, "num_predict": 512},
        "thinking": "effort_low",
        "arms": ("paraphrase", "resample"),
        "warmup": True,
    },
    "local_qwen_metal": {
        "kind": "local",
        "plane": "local_metal",
        "model_tag": "qwen3.6:35b-a3b-q8_0",
        "options": {"temperature": 0, "seed": 42, "num_predict": 512},
        "thinking": "think_off",
        "arms": ("paraphrase",),
        "warmup": True,
    },
}

# Substrate block order — fixed so the schedule digest is reproducible.
SUBSTRATE_ORDER = (
    "haiku_1p",
    "sonnet_1p",
    "sonnet_bedrock",
    "local_20b_cuda",
    "local_qwen_metal",
)

# Runner mode groupings: one run targets one credential family, like
# study 3's one-box-per-run.
STUDY5_API_SUBSTRATES = ("haiku_1p", "sonnet_1p", "sonnet_bedrock")
STUDY5_LOCAL_SUBSTRATES_BY_BOX = {
    "cuda": ("local_20b_cuda",),
    "metal": ("local_qwen_metal",),
}

# Pilot: stratified item subset — first PILOT_PER_CLASS ids per gradient
# class, so every gradient is represented (the first 20 corpus ids alone
# would carry zero ambiguous items).
PILOT_PER_CLASS = 7

WARMUP_PROMPT = "Reply with exactly: STUDY5-WARMUP-OK"


def pilot_corpus(corpus, per_class=PILOT_PER_CLASS):
    """Deterministic stratified pilot subset: first per_class items of
    each gradient class in id order, corpus meta carried through."""
    chosen = []
    taken = {}
    for item in sorted(corpus["items"], key=lambda i: i["id"]):
        gradient = item["gradient"]
        if taken.get(gradient, 0) < per_class:
            taken[gradient] = taken.get(gradient, 0) + 1
            chosen.append(item)
    return {"meta": corpus["meta"], "items": chosen}


def build_prompt(template_text, document):
    return f"{template_text}{PROMPT_JOINER}{document}"


def _arm_options(cfg, arm):
    """Local-plane decode options per arm. Local resample is UNSEEDED by
    design — a seeded local engine reproduces temp-0.7 byte-for-byte
    (study-3 pilot #1), which defeats the arm."""
    if arm != "resample":
        return dict(cfg["options"])
    options = {k: v for k, v in cfg["options"].items() if k != "seed"}
    options["temperature"] = RESAMPLE_TEMPERATURE
    return options


def _payload_and_sha(cfg, arm, prompt):
    extra = {"temperature": RESAMPLE_TEMPERATURE} if arm == "resample" else None
    if cfg["kind"] == "messages":
        params = canonical_messages_params(
            cfg["model_cfg"], cfg["model_id"], prompt, cfg["thinking"],
            extra=extra,
        )
        return params, sha256_hex(canonical_bytes(params))
    if cfg["kind"] == "bedrock":
        body = canonical_body(cfg["model_cfg"], prompt, cfg["thinking"], extra=extra)
        return body, sha256_hex(body)
    body = canonical_local_body(
        cfg["model_tag"], prompt, cfg["thinking"], options=_arm_options(cfg, arm)
    )
    return body, sha256_hex(body)


def _item_record(substrate, cfg, arm, corpus_item, template_id, prompt,
                 repeat, control=None):
    payload, sha = _payload_and_sha(cfg, arm, prompt)
    item_id = corpus_item["id"] if corpus_item else "warmup"
    meta = {
        "substrate": substrate,
        "arm": arm,
        "item_id": item_id,
        "template_id": template_id,
    }
    if corpus_item:
        meta["gradient"] = corpus_item["gradient"]
        meta["target_field"] = corpus_item["target_field"]
    if control:
        meta["control"] = control
    return {
        "cell": f"study5|{substrate}|{arm}|{item_id}|{template_id}",
        "meta": meta,
        "plane": cfg["plane"],
        "kind": cfg["kind"],
        "model_id": cfg.get("model_id") or cfg.get("model_tag"),
        "payload": payload,
        "sha": sha,
        "repeat": repeat,
    }


def build_study5_items(corpus=None, substrates=None, items_limit=None,
                       resample_n=RESAMPLE_N):
    """Deterministic full schedule: per-substrate blocks, warmup heads on
    local substrates, paraphrase (items x all templates x1) then resample
    (items x RESAMPLE_TEMPLATE x resample_n) within each block."""
    corpus = corpus or load_corpus()
    templates = corpus["meta"]["instruction_templates"]
    corpus_items = corpus["items"][:items_limit] if items_limit else corpus["items"]
    schedule = []
    for substrate in SUBSTRATE_ORDER:
        if substrates and substrate not in substrates:
            continue
        cfg = STUDY5_SUBSTRATES[substrate]
        if cfg.get("warmup"):
            schedule.append(
                _item_record(
                    substrate, cfg, "paraphrase", None, "warmup",
                    WARMUP_PROMPT, 0, control="warmup",
                )
            )
        if "paraphrase" in cfg["arms"]:
            for corpus_item in corpus_items:
                for template_id in TEMPLATE_IDS:
                    prompt = build_prompt(
                        templates[template_id], corpus_item["document"]
                    )
                    schedule.append(
                        _item_record(
                            substrate, cfg, "paraphrase", corpus_item,
                            template_id, prompt, 0,
                        )
                    )
        if "resample" in cfg["arms"]:
            for corpus_item in corpus_items:
                prompt = build_prompt(
                    templates[RESAMPLE_TEMPLATE], corpus_item["document"]
                )
                for repeat in range(resample_n):
                    schedule.append(
                        _item_record(
                            substrate, cfg, "resample", corpus_item,
                            RESAMPLE_TEMPLATE, prompt, repeat,
                        )
                    )
    return schedule


def schedule_digest(schedule):
    joined = "\n".join(f'{it["cell"]}#{it["repeat"]}' for it in schedule)
    return sha256_hex(joined.encode("utf-8"))

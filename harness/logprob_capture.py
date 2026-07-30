"""Compact per-token logprob margins from an Ollama /api/chat payload.

Companion-B capture (FOLLOWUP-COMPANIONS.md): the analysis object is the
per-token row [token, chosen_logprob, top1_logprob, top2_logprob]. The full
top-k arrays are deliberately not persisted — the top1-minus-top2 margin at
every generated position is what the margins endpoints consume, and the row
form keeps records bounded at long generation lengths.

Response shape (probe-verified on 0.30.5, evidence/logprobs-probe-metal.json):
top-level `logprobs` list; each entry carries `token`, `logprob`, and — when
`top_logprobs` was requested — a `top_logprobs` list of {token, logprob}.
"""


def compact_margins(payload):
    """None when the payload carries no logprobs (field not requested, or
    the engine returned none). `chosen_not_top1` counts positions where the
    generated token is not the top-1 candidate — an anomaly under greedy
    decoding, surfaced rather than assumed away."""
    entries = (payload or {}).get("logprobs")
    if not entries:
        return None
    rows = []
    chosen_not_top1 = 0
    min_margin = None
    argmin = None
    for index, entry in enumerate(entries):
        token = entry.get("token")
        chosen_lp = entry.get("logprob")
        tops = entry.get("top_logprobs") or []
        top1_lp = tops[0].get("logprob") if len(tops) >= 1 else None
        top2_lp = tops[1].get("logprob") if len(tops) >= 2 else None
        rows.append([token, chosen_lp, top1_lp, top2_lp])
        if top1_lp is not None and tops[0].get("token") != token:
            chosen_not_top1 += 1
        if top1_lp is not None and top2_lp is not None:
            margin = top1_lp - top2_lp
            if min_margin is None or margin < min_margin:
                min_margin = margin
                argmin = index
    return {
        "n_tokens": len(rows),
        "tokens": rows,
        "min_top2_margin": min_margin,
        "argmin_index": argmin,
        "chosen_not_top1": chosen_not_top1,
    }

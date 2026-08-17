#!/usr/bin/env python3
"""Pilot analysis: divergence + fence/format reads per door per task."""
import json, glob, collections, statistics, pathlib, re

SP = pathlib.Path("/private/tmp/claude-501/-Users-jelstner/4316d3c5-8bb4-4136-9f24-6a877a6bf1bf/scratchpad")

def door1_reps(task):
    reps = []
    for f in sorted(glob.glob(str(SP / f"pilot/door1/{task}/*.json")), key=lambda p: int(pathlib.Path(p).stem)):
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:
            continue
        text = "".join(c["text"] for c in d["output"]["message"]["content"] if "text" in c)
        reps.append({"text": text, "out_tok": d["usage"]["outputTokens"],
                     "in_tok": d["usage"]["inputTokens"], "stop": d["stopReason"],
                     "lat": d.get("metrics", {}).get("latencyMs")})
    return reps

def door4_reps(task):
    reps = []
    for f in sorted(glob.glob(str(SP / f"pilot/door4/{task}/*.jsonl")), key=lambda p: int(pathlib.Path(p).stem)):
        text, usage = None, None
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
                text = ev["item"].get("text")
            if "usage" in ev and isinstance(ev["usage"], dict):
                usage = ev["usage"]
        if text is None:
            continue
        reps.append({"text": text,
                     "out_tok": (usage or {}).get("output_tokens"),
                     "in_tok": (usage or {}).get("input_tokens"),
                     "cached": (usage or {}).get("cached_input_tokens"),
                     "reason_tok": (usage or {}).get("reasoning_output_tokens")})
    return reps

def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return len(a) if len(a) != len(b) else None

def try_json(t):
    s = t.strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        return None

def report(door, task, reps):
    if not reps:
        print(f"\n## {door} / {task}: NO DATA")
        return
    texts = [r["text"] for r in reps]
    n = len(texts)
    counter = collections.Counter(texts)
    modal, modal_n = counter.most_common(1)[0]
    divs = [first_div(modal, t) for t in texts if t != modal]
    divs = [d for d in divs if d is not None]
    outs = [r["out_tok"] for r in reps if r.get("out_tok") is not None]
    print(f"\n## {door} / {task}  (n={n})")
    print(f"  distinct outputs: {len(counter)}  | modal share: {modal_n}/{n}")
    if divs:
        print(f"  first-divergence char (vs modal): min={min(divs)} median={int(statistics.median(divs))}")
    if outs:
        print(f"  outputTokens: min={min(outs)} median={int(statistics.median(outs))} max={max(outs)}")
    stops = collections.Counter(r.get("stop") for r in reps if r.get("stop"))
    if stops and set(stops) != {"end_turn"}:
        print(f"  stopReasons: {dict(stops)}")
    if task == "extraction":
        ok = sum(1 for t in texts if t.strip() == "PO-83614-QN")
        print(f"  exact 'PO-83614-QN': {ok}/{n}")
        for t, c in counter.most_common(5):
            print(f"    {c}x {t[:70]!r}")
    if task == "structured_json":
        fenced = sum(1 for t in texts if "```" in t)
        parsed = [try_json(t) for t in texts]
        valid = sum(1 for p in parsed if p is not None)
        keysets = collections.Counter(tuple(sorted(p.keys())) for p in parsed if isinstance(p, dict))
        print(f"  FENCE RATE: {fenced}/{n}  | JSON-parseable (fences stripped): {valid}/{n}")
        print(f"  key sets: {dict(keysets)}")
        vals = collections.Counter(json.dumps(p, sort_keys=True) for p in parsed if p is not None)
        print(f"  distinct parsed payloads: {len(vals)} | modal payload share: {vals.most_common(1)[0][1] if vals else 0}/{valid}")
    if task == "open_generation":
        lens = [len(t.split()) for t in texts]
        print(f"  word count: min={min(lens)} median={int(statistics.median(lens))} max={max(lens)}")
    rt = [r.get("reason_tok") for r in reps if r.get("reason_tok") is not None]
    if rt:
        print(f"  reasoning_output_tokens: min={min(rt)} median={int(statistics.median(rt))} max={max(rt)}")

if __name__ == "__main__":
    for task in ("extraction", "structured_json", "open_generation"):
        reps = door1_reps(task)
        report("door1-converse", task, reps)
    for task in ("structured_json", "extraction"):
        reps = door4_reps(task)
        report("door4-codex-sub", task, reps)

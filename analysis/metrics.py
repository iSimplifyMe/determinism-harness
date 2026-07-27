"""Per-cell reproducibility metrics.

All measurement is deterministic code — no model in the loop, consistent
with the published stance in The Finite Chain. The measured artifact is the
concatenated text-block output of each call.
"""
import hashlib
from collections import Counter


def modal_response(texts):
    """Most frequent text in a cell. Ties break to the lexicographically
    smallest candidate so the choice is deterministic."""
    if not texts:
        raise ValueError("empty cell")
    counts = Counter(texts)
    top = max(counts.values())
    text = min(t for t, c in counts.items() if c == top)
    return text, top


def first_divergence_index(a, b):
    """Character index of the first difference, or None if equal."""
    if a == b:
        return None
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def token_divergence_index(a, b):
    """Whitespace-token index of the first difference, or None if the token
    streams are identical."""
    if a == b:
        return None
    ta, tb = a.split(), b.split()
    if ta == tb:
        return None
    limit = min(len(ta), len(tb))
    for i in range(limit):
        if ta[i] != tb[i]:
            return i
    return limit


def _banded_distance(a, b, band):
    """Levenshtein distance restricted to |i - j| <= band.

    Returns the exact distance when it is <= band, else None.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > band:
        return None
    inf = band + 1
    prev = {j: j for j in range(0, min(lb, band) + 1)}
    for i in range(1, la + 1):
        cur = {}
        jlo = max(0, i - band)
        jhi = min(lb, i + band)
        for j in range(jlo, jhi + 1):
            if j == 0:
                cur[0] = i
                continue
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = prev.get(j - 1, inf) + cost
            up = prev.get(j, inf) + 1
            if up < best:
                best = up
            left = cur.get(j - 1, inf) + 1
            if left < best:
                best = left
            cur[j] = best
        prev = cur
        if not prev or min(prev.values()) > band:
            return None
    d = prev.get(lb, inf)
    return d if d <= band else None


def levenshtein_banded(a, b, cap=512):
    """Exact Levenshtein distance below `cap`, else (cap, True).

    Shared prefix/suffix are stripped first (divergent model outputs usually
    share a long prefix), then Ukkonen band-doubling keeps the common case
    fast. Distances at or above `cap` are reported as the cap with a capped
    flag — 'massively divergent' needs no finer resolution and keeps the
    analysis pass bounded.
    """
    if a == b:
        return 0, False
    i = 0
    la, lb = len(a), len(b)
    limit = min(la, lb)
    while i < limit and a[i] == b[i]:
        i += 1
    a, b = a[i:], b[i:]
    j = 0
    limit = min(len(a), len(b))
    while j < limit and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    if j:
        a, b = a[: len(a) - j], b[: len(b) - j]
    la, lb = len(a), len(b)
    if abs(la - lb) >= cap:
        return cap, True
    if la == 0:
        return lb, False
    if lb == 0:
        return la, False
    band = 1
    while band < cap:
        d = _banded_distance(a, b, band)
        if d is not None and d <= band:
            return d, False
        band *= 2
    d = _banded_distance(a, b, cap - 1)
    if d is not None and d <= cap - 1:
        return d, False
    return cap, True


def normalized_distance(a, b, cap=512):
    """Levenshtein distance divided by max(len(a), len(b)) — lengths taken
    before affix stripping. Returns (value, capped)."""
    if a == b:
        return 0.0, False
    d, capped = levenshtein_banded(a, b, cap)
    denom = max(len(a), len(b))
    return (d / denom if denom else 0.0), capped


def pop_variance(xs):
    """Population variance."""
    xs = list(xs)
    if not xs:
        raise ValueError("empty sequence")
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def cell_metrics(records):
    """Derived metrics for one cell.

    `records` is a list of dicts carrying at least `text`; `output_tokens`
    is used when present. Records must already be filtered to valid calls
    (no truncation, no errors) — exclusion is the analyzer's job.
    """
    if not records:
        raise ValueError("empty cell")
    texts = [r["text"] for r in records]
    n = len(texts)
    modal_text, modal_count = modal_response(texts)
    counts = Counter(texts)

    same_pairs = sum(c * (c - 1) for c in counts.values())
    total_pairs = n * (n - 1)
    pairwise = same_pairs / total_pairs if total_pairs else 1.0

    divergence_indices = []
    norm_all = []
    norm_max = 0.0
    any_capped = False
    for t in texts:
        if t == modal_text:
            norm_all.append(0.0)
            continue
        divergence_indices.append(first_divergence_index(modal_text, t))
        nd, capped = normalized_distance(modal_text, t)
        norm_all.append(nd)
        if nd > norm_max:
            norm_max = nd
        any_capped = any_capped or capped

    token_counts = [r.get("output_tokens") for r in records]
    known = [t for t in token_counts if t is not None]
    token_variance = pop_variance(known) if known else None

    return {
        "n": n,
        "modal_count": modal_count,
        "modal_share": modal_count / n,
        "modal_sha256": hashlib.sha256(modal_text.encode("utf-8")).hexdigest(),
        "distinct_count": len(counts),
        "all_identical": len(counts) == 1,
        "pairwise_agreement": pairwise,
        "divergence_char_indices": sorted(divergence_indices),
        "norm_distance_mean_all": sum(norm_all) / n,
        "norm_distance_max": norm_max,
        "any_distance_capped": any_capped,
        "output_token_variance": token_variance,
    }

"""Statistics for the determinism study.

Stdlib only, deliberately: every number in the paper must be reproducible
with zero third-party dependencies, and no model is ever in the measurement
loop. Normal CDF comes from math.erf; binomial terms from math.lgamma.
"""
import math

Z95 = 1.959963984540054  # 97.5th percentile of the standard normal
Z90 = 1.6448536269514722  # 95th percentile (used for the TOST-consistent 90% CI)


def normal_cdf(x):
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson_interval(k, n, z=Z95):
    """Wilson score interval for a binomial proportion.

    Same method as the Blind Panel paper — methodological continuity across
    the library is deliberate.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be in [0, n]")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_pmf(k, n, p):
    """Exact binomial pmf, computed in log space for stability."""
    if not 0 <= k <= n:
        return 0.0
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    log_coeff = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.exp(log_coeff + k * math.log(p) + (n - k) * math.log(1.0 - p))


def binom_test(k, n, p=0.5, alternative="two-sided"):
    """Exact binomial test.

    Two-sided uses the standard small-p-value definition: the sum of all
    outcome probabilities no larger than the observed outcome's probability
    (with a 1+1e-7 tolerance for floating-point ties).
    """
    if alternative == "greater":
        return min(1.0, sum(binom_pmf(i, n, p) for i in range(k, n + 1)))
    if alternative == "less":
        return min(1.0, sum(binom_pmf(i, n, p) for i in range(0, k + 1)))
    if alternative == "two-sided":
        observed = binom_pmf(k, n, p)
        threshold = observed * (1.0 + 1e-7)
        total = 0.0
        for i in range(0, n + 1):
            pm = binom_pmf(i, n, p)
            if pm <= threshold:
                total += pm
        return min(1.0, total)
    raise ValueError(f"unknown alternative: {alternative}")


def _anscombe(x, n):
    """Anscombe-adjusted proportion, used only to rescue a zero SE."""
    return (x + 0.5) / (n + 1.0)


def two_prop_tost(x1, n1, x2, n2, delta, alpha=0.05):
    """Two one-sided tests for equivalence of two proportions.

    H0a: p1 - p2 <= -delta   vs   Ha: p1 - p2 > -delta
    H0b: p1 - p2 >= +delta   vs   Ha: p1 - p2 < +delta

    Equivalence is claimed when both one-sided tests reject at alpha, i.e.
    max(p_lower, p_upper) < alpha. Wald z with unpooled SE; when both
    observed proportions sit exactly on 0 or 1 (SE of zero), the SE is
    recomputed from Anscombe-adjusted proportions so the test remains
    defined — the observed difference itself is never adjusted.
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("arm sizes must be positive")
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    if se == 0.0:
        q1, q2 = _anscombe(x1, n1), _anscombe(x2, n2)
        se = math.sqrt(q1 * (1.0 - q1) / n1 + q2 * (1.0 - q2) / n2)
    p_lower = 1.0 - normal_cdf((diff + delta) / se)
    p_upper = normal_cdf((diff - delta) / se)
    p = max(p_lower, p_upper)
    return {
        "diff": diff,
        "se": se,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p": p,
        "equivalent": p < alpha,
        "ci90": (diff - Z90 * se, diff + Z90 * se),
        "delta": delta,
        "alpha": alpha,
    }


def diff_ci(x1, n1, x2, n2, z=Z95):
    """Wald confidence interval on the difference of two proportions."""
    if n1 <= 0 or n2 <= 0:
        raise ValueError("arm sizes must be positive")
    p1, p2 = x1 / n1, x2 / n2
    diff = p1 - p2
    se = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)
    return diff - z * se, diff + z * se

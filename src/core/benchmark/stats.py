"""Statistical summaries for fixed-start policy benchmarks."""
import math

def wilson_interval(successes, episodes):
    if not 0 <= successes <= episodes:
        raise ValueError('Require 0 <= successes <= episodes')
    if episodes == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / episodes
    denom = 1 + z * z / episodes
    centre = (p + z * z / (2 * episodes)) / denom
    half = z * math.sqrt(p * (1 - p) / episodes + z * z / (4 * episodes**2)) / denom
    return [max(0., centre - half), min(1., centre + half)]


def mcnemar(first, second):
    """Exact two-sided paired binomial McNemar, one Boolean per shared start."""
    if len(first) != len(second):
        raise ValueError('Paired observations must have equal length')
    b = sum(bool(a) and not bool(c) for a, c in zip(first, second))
    c = sum(not bool(a) and bool(d) for a, d in zip(first, second))
    n = b + c
    p = min(1., 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)) if n else 1.
    return dict(shared_starts=len(first), first_only=b, second_only=c, discordant=n,
                p_value=p, method='exact two-sided McNemar')


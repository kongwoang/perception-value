"""Tie-robust measures of whether two downstream targets want compute on the same inputs.

Two lessons are baked in here, both learned the hard way.

**Quota-based transfer is not tie-robust when a target is sparse.** The braking controller responds
on 219 of 2,655 frames, so any allocation of 20% of the frames is mostly free choice: its optimum
is under-determined, and the reported cross-eta depends on how those free slots are filled. Under
a favourable filling, brake-optimum -> plan reaches 0.839, while plan-optimum -> brake stays at
0.037. The symmetric "indistinguishable from random in both directions" reading was an artefact of
one tie-break.

**Pairwise disagreement is tie-free.** Restricting to pairs that *both* targets rank strictly
removes the free choice entirely, so the disagreement rate and Goodman-Kruskal gamma say what the
quota measures cannot.

`best_compatible` therefore reports the transfer that is most favourable to the source target --
an upper bound -- so a low value is evidence and a high value is not.
"""
from __future__ import annotations

import numpy as np


def best_compatible_set(v_from: np.ndarray, v_to: np.ndarray, k: int) -> np.ndarray:
    """The |A| <= k allocation that maximises sum(v_from), breaking ties to favour v_to.

    Among all optima for the source target, this is the one best for the target target:
      * every frame with v_from > 0 is taken, since each strictly increases the source sum;
      * if more than k of those exist the quota binds, so the top k by v_from are taken and any
        tie at the cut value is broken by v_to descending;
      * if fewer than k exist the leftover slots are free -- adding a v_from == 0 frame leaves the
        source sum unchanged -- so they are filled by the highest v_to among those;
      * frames with v_from < 0 are never taken.
    """
    v_from = np.asarray(v_from, float)
    v_to = np.asarray(v_to, float)
    n = len(v_from)
    sel = np.zeros(n, bool)
    if k <= 0:
        return sel
    pos = np.flatnonzero(v_from > 0)
    if len(pos) >= k:
        # order by source value, then by target value, so ties at the cut favour the target
        order = pos[np.lexsort((-v_to[pos], -v_from[pos]))]
        sel[order[:k]] = True
        return sel
    sel[pos] = True
    free = np.flatnonzero((v_from == 0) & ~sel)
    take = free[np.argsort(-v_to[free], kind="stable")][: k - len(pos)]
    sel[take] = True
    return sel


def best_compatible_eta(v_from: np.ndarray, v_to: np.ndarray, quota: float) -> float:
    """Share of the target's achievable benefit captured by the source's *most favourable* optimum."""
    v_to = np.asarray(v_to, float)
    k = max(int(round(quota * len(v_to))), 1)
    own = best_compatible_set(v_to, v_to, k)
    best = float(v_to[own].sum())
    got = float(v_to[best_compatible_set(v_from, v_to, k)].sum())
    return got / best if abs(best) > 1e-12 else np.nan


def strict_pairs(a: np.ndarray, b: np.ndarray, rng=None, npairs: int = 600_000,
                 exhaustive_max: int = 8000, decimals: int = 9) -> dict:
    """Disagreement rate and Goodman-Kruskal gamma over pairs both targets rank strictly.

    A pair is usable only when neither target is indifferent about it; those are exactly the pairs
    where a quota-based measure would have been free to choose either way.

    Every pair is enumerated when there are at most `exhaustive_max` items, so the pair count is
    exact and comparable across analyses; random pair sampling is only the fallback beyond that.
    """
    # Values are rounded before comparison so floating-point noise is a tie, not a ranking.
    # With exact comparison the brake-vs-plan analysis counted 464,858 strict pairs; 620 of those
    # differed only around 1e-12, and at 1e-9 the count is 464,238, matching an independent
    # calculation. Disagreement moves by 0.0001, so no conclusion depends on it.
    a, b = np.round(np.asarray(a, float), decimals), np.round(np.asarray(b, float), decimals)
    n = len(a)
    conc = disc = 0
    if n <= exhaustive_max:
        for i in range(n - 1):
            sa = np.sign(a[i] - a[i + 1:])
            sb = np.sign(b[i] - b[i + 1:])
            u = (sa != 0) & (sb != 0)
            if u.any():
                same = sa[u] == sb[u]
                conc += int(same.sum())
                disc += int((~same).sum())
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        i = rng.integers(0, n, npairs)
        j = rng.integers(0, n, npairs)
        m = i != j
        sa, sb = np.sign(a[i[m]] - a[j[m]]), np.sign(b[i[m]] - b[j[m]])
        u = (sa != 0) & (sb != 0)
        conc, disc = int((sa[u] == sb[u]).sum()), int((sa[u] != sb[u]).sum())
    tot = conc + disc
    if tot == 0:
        return {"n_pairs_strict": 0, "disagreement": np.nan, "gamma": np.nan}
    return {"n_pairs_strict": tot, "disagreement": disc / tot, "gamma": (conc - disc) / tot}

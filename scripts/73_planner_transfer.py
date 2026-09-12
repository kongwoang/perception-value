#!/usr/bin/env python
"""Phase 0G: does the identity of the downstream planner change WHICH inputs are valuable?

The Phase 0F/0G eta tables answer "how well does signal s allocate for planner q".  This
script answers the question the paper's claim actually rests on: given the *oracle* ranking
for planner q1, how much of planner q2's achievable benefit does it capture?

    eta_{q1 -> q2}(k) = sum over the top-k frames by V^q1 of V^q2
                        / sum over the top-k frames by V^q2 of V^q2

That is 1.0 when the two planners agree about which inputs deserve compute and near k/N when
they are unrelated.  Ordering agreement is also reported directly: Spearman, Kendall, pairwise
inversion rate, and top-10/20% oracle-set overlap.

Costs are never compared across datasets; only these normalised quantities are.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import runmeta                                                        # noqa: E402
from rap.paths import CACHE, RESULTS                                           # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]


def topk(v: np.ndarray, q: float) -> np.ndarray:
    k = int(round(q * len(v)))
    sel = np.zeros(len(v), bool)
    if k > 0:
        sel[np.argsort(-np.asarray(v, float), kind="stable")[:k]] = True
    return sel


def cross_eta(v_from: np.ndarray, v_to: np.ndarray, q: float) -> float:
    """Share of q2's achievable benefit captured by q1's oracle ranking."""
    best = float(np.asarray(v_to, float)[topk(v_to, q)].sum())
    got = float(np.asarray(v_to, float)[topk(v_from, q)].sum())
    return got / best if abs(best) > 1e-12 else np.nan


def inversion_rate(a, b, rng, npairs=200_000) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = rng.integers(0, len(a), npairs), rng.integers(0, len(a), npairs)
    k = i != j
    i, j = i[k], j[k]
    sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    ok = (sa != 0) & (sb != 0)
    return float((sa[ok] != sb[ok]).mean()) if ok.any() else np.nan


def boot_ci(fn, scenes, nboot, rng, *arrays):
    uniq = np.unique(scenes)
    idx = {u: np.flatnonzero(scenes == u) for u in uniq}
    vals = []
    for _ in range(nboot):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        v = fn(*[a[take] for a in arrays])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < nboot // 4:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", required=True,
                    help="CSV with sample_token, scene and one V_<planner> column per planner")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--out", default=str(Path(RESULTS) / "final" /
                                        "phase0g_planner_d_transfer.csv"))
    ap.add_argument("--tag", default="planner_transfer")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    rng = np.random.default_rng(0)

    d = pd.read_csv(args.values)
    cols = [c for c in d.columns if c.startswith("V_")]
    if len(cols) < 2:
        raise SystemExit(f"need at least two V_ columns, found {cols}")
    scenes = d["scene"].to_numpy()
    print(f"  {len(d)} frames, planners: {[c[2:] for c in cols]}")

    rows = []
    for a, b in itertools.combinations(cols, 2):
        va, vb = d[a].to_numpy(float), d[b].to_numpy(float)
        r = {"pair": f"{a[2:]}-{b[2:]}", "q1": a[2:], "q2": b[2:], "n_frames": len(d),
             "n_scenes": int(len(np.unique(scenes))),
             "spearman": float(stats.spearmanr(va, vb).correlation),
             "kendall": float(stats.kendalltau(va, vb).correlation),
             "inversion_rate": inversion_rate(va, vb, rng)}
        for q in QUOTAS:
            k = int(q * 100)
            sa, sb = topk(va, q), topk(vb, q)
            r[f"top{k}_overlap"] = float((sa & sb).sum() / max(sa.sum(), 1))
            r[f"eta_q1_to_q2_{k}"] = cross_eta(va, vb, q)
            r[f"eta_q2_to_q1_{k}"] = cross_eta(vb, va, q)
            r[f"regret_q1_to_q2_{k}"] = 1.0 - r[f"eta_q1_to_q2_{k}"]
            r[f"regret_q2_to_q1_{k}"] = 1.0 - r[f"eta_q2_to_q1_{k}"]
        lo, hi = boot_ci(lambda x, y: cross_eta(x, y, 0.20), scenes, args.nboot, rng, va, vb)
        r["eta_q1_to_q2_20_lo"], r["eta_q1_to_q2_20_hi"] = lo, hi
        lo, hi = boot_ci(lambda x, y: cross_eta(y, x, 0.20), scenes, args.nboot, rng, va, vb)
        r["eta_q2_to_q1_20_lo"], r["eta_q2_to_q1_20_hi"] = lo, hi
        lo, hi = boot_ci(lambda x, y: float((topk(x, 0.20) & topk(y, 0.20)).sum()
                                            / max(topk(y, 0.20).sum(), 1)),
                         scenes, args.nboot, rng, va, vb)
        r["top20_overlap_lo"], r["top20_overlap_hi"] = lo, hi
        rows.append(r)

    m = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    m.to_csv(run / Path(args.out).name, index=False)
    pd.set_option("display.width", 200)
    print("\n=== planner-to-planner transfer of the valuable-input ranking ===")
    print(m[["pair", "spearman", "kendall", "inversion_rate", "top10_overlap",
             "top20_overlap", "eta_q1_to_q2_20", "eta_q2_to_q1_20"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # falsifier D-F2 / B-F3: high overlap AND high cross-eta in both directions
    fired = m[(m.top20_overlap >= 0.80) & (m.eta_q1_to_q2_20 >= 0.80)
              & (m.eta_q2_to_q1_20 >= 0.80)]
    verdict = {"pairs_with_transferable_ranking": fired.pair.tolist(),
               "threshold": {"top20_overlap": 0.80, "cross_eta_20": 0.80},
               "n_pairs": len(m)}
    (run / "transfer_falsifier.json").write_text(json.dumps(verdict, indent=2))
    if len(fired):
        print(f"\n  FALSIFIER FIRED for: {fired.pair.tolist()} "
              f"-- ranking transfers, planner conditionality not supported for these pairs")
    else:
        print("\n  no pair reaches top-20 overlap >= 0.80 with cross-eta >= 0.80 both ways")
    print("  wrote", args.out)


if __name__ == "__main__":
    main()

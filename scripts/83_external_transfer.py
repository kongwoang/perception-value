#!/usr/bin/env python
"""Phase 0G Track B, B7 + B8: two published planners with the SAME objective, same states.

The strongest remaining objection to the planner-conditionality claim is that a safety objective
and an imitation objective would *obviously* want compute on different inputs.  PDM-Closed and
IDMPlanner remove that: both are published rule-based planners we did not write, both are scored
here by the SAME external cost against the SAME real tracked objects, on the SAME logged states.
If they still disagree about which states deserve compute, the disagreement is not a tautology.

Everything uses the tie-robust measures in `rap.transfer`, because both decision values are zero on
most states and quota-based transfer then depends on how the free slots are filled:
  * disagreement and Goodman-Kruskal gamma over pairs BOTH planners rank strictly;
  * best-compatible transfer, the source optimum most favourable to the target (an upper bound);
  * responsive fraction, the share of each top-q set a target actually distinguishes.

Costs are reported per component and as two scalars, so no result hinges on one weighting:
`safety` (collision and clearance only) and `scalar_J` (safety plus log deviation).  The weights
are Planner B's, chosen after the 112-state pilot and before this 1,440-state run.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import RESULTS                                                  # noqa: E402
from rap.transfer import best_compatible_eta, strict_pairs                     # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]
PLANNERS = ("idm", "pdm_closed")


def costs(d: pd.DataFrame, mode: str) -> dict:
    coll = d[f"collision_{mode}"].to_numpy(float)
    short = np.clip(1.0 - d[f"min_clearance_{mode}"].to_numpy(float), 0, 2) ** 2
    dev = d[f"log_deviation_mean_{mode}"].to_numpy(float)
    return {"collision": coll, "clearance_shortfall": short, "log_deviation": dev,
            "safety": 10 * coll + 1.5 * short, "scalar_J": 10 * coll + 1.5 * short + 0.1 * dev}


def topk_rand(v, q, seed):
    v = np.asarray(v, float)
    k = max(int(round(q * len(v))), 1)
    sel = np.zeros(len(v), bool)
    sel[np.lexsort((np.random.default_rng(seed).random(len(v)), -v))[:k]] = True
    return sel


def scene_boot(groups, nboot, rng):
    uniq = np.unique(groups)
    idx = {u: np.flatnonzero(groups == u) for u in uniq}
    for _ in range(nboot):
        yield np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])


def ci(vals):
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
        if len(vals) >= 20 else (np.nan, np.nan)


def per_planner(cc, cf, name, planner):
    V = cc - cf
    nz = np.abs(V) > 1e-9
    allc, allf = float(cc.sum()), float(cf.sum())
    r = {"planner": planner, "cost": name, "n_states": len(V),
         "v_pos": int((V > 1e-9).sum()), "v_zero": int((~nz).sum()), "v_neg": int((V < -1e-9).sum()),
         "harmful_among_affected": float((V < -1e-9).sum() / max(nz.sum(), 1)),
         "all_cheap_cost": allc, "all_full_cost": allf,
         "all_full_reduction": (allc - allf) / allc if allc > 1e-12 else np.nan}
    for q in QUOTAS:
        k = max(int(round(q * len(V))), 1)
        top = np.sort(V)[::-1][:k]
        gain = float(top[top > 0].sum())
        r[f"oracle{int(q*100)}_reduction"] = gain / allc if allc > 1e-12 else np.nan
    o20, af = r["oracle20_reduction"], r["all_full_reduction"]
    r["oracle20_beats_all_full"] = bool(np.isfinite(o20) and np.isfinite(af) and o20 > af)
    # B-F2: share of the oracle's achievable improvement that selectivity adds over all-FULL
    r["selective_extra_share"] = (o20 - af) / o20 if (np.isfinite(o20) and o20 > 1e-12) else np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--out", default=str(Path(RESULTS) / "final"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(args.out)

    raw = {p: pd.read_csv(out / f"phase0g_external_{p}_raw.csv") for p in PLANNERS}
    keys = ["scenario", "iteration"]
    j = raw["idm"].merge(raw["pdm_closed"], on=keys, suffixes=("_idm", "_pdm"))
    for m in ("reference", "cheap", "full"):
        for p, s in (("idm", "_idm"), ("pdm_closed", "_pdm")):
            j = j[j[f"ok_{m}{s}"] == 1]
    j = j.reset_index(drop=True)
    groups = j["scenario"].to_numpy()
    print(f"  common states {len(j)} across {j.scenario.nunique()} scenarios")

    def planner_view(sfx):
        return j.rename(columns=lambda c: c[:-len(sfx)] if c.endswith(sfx) else c)

    views = {"idm": planner_view("_idm"), "pdm_closed": planner_view("_pdm")}
    summary, transfer = [], []
    for name in ("collision", "clearance_shortfall", "log_deviation", "safety", "scalar_J"):
        V = {}
        for p in PLANNERS:
            cc, cf = costs(views[p], "cheap")[name], costs(views[p], "full")[name]
            summary.append(per_planner(cc, cf, name, p))
            V[p] = cc - cf
        a, b = V["idm"], V["pdm_closed"]
        row = {"cost": name, "n_states": len(a),
               "idm_responsive": float((np.abs(a) > 1e-9).mean()),
               "pdm_responsive": float((np.abs(b) > 1e-9).mean()),
               "both_responsive": int(((np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)).sum())}

        sp = strict_pairs(a, b, rng)
        row.update({f"strict_{k}": v for k, v in sp.items()})
        boots = [strict_pairs(a[t], b[t], rng, npairs=200_000)
                 for t in scene_boot(groups, min(args.nboot, 200), rng)]
        for k in ("disagreement", "gamma"):
            row[f"strict_{k}_lo"], row[f"strict_{k}_hi"] = ci([x[k] for x in boots])

        both = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
        row["spearman_both_responsive"] = (float(stats.spearmanr(a[both], b[both]).correlation)
                                           if both.sum() > 5 else np.nan)
        for q in QUOTAS:
            qq = int(q * 100)
            ov = np.mean([(topk_rand(a, q, s) & topk_rand(b, q, s + 10_000)).sum()
                          / max(topk_rand(a, q, s).sum(), 1) for s in range(8)])
            row[f"overlap{qq}"], row[f"overlap{qq}_chance"] = float(ov), q
            row[f"bc_idm_to_pdm_{qq}"] = best_compatible_eta(a, b, q)
            row[f"bc_pdm_to_idm_{qq}"] = best_compatible_eta(b, a, q)
            row[f"resp_idm_top{qq}"] = float((np.abs(a[topk_rand(a, q, 0)]) > 1e-9).mean())
            row[f"resp_pdm_top{qq}"] = float((np.abs(b[topk_rand(b, q, 0)]) > 1e-9).mean())
        transfer.append(row)

    S, T = pd.DataFrame(summary), pd.DataFrame(transfer)
    S.to_csv(out / "phase0g_external_planner_summary.csv", index=False)
    T.to_csv(out / "phase0g_external_planner_transfer.csv", index=False)

    # falsifiers, on the pre-declared scalar and on safety alone
    fals = {}
    for name in ("scalar_J", "safety"):
        s = S[S.cost == name].set_index("planner")
        t = T[T.cost == name].iloc[0]
        fals[name] = {
            "B-F1_fires": bool((s.harmful_among_affected < 0.05).all()),
            "B-F2_fires": bool((s.selective_extra_share < 0.10).all()),
            "B-F3_fires": bool(t.overlap20 >= 0.80 and t.bc_idm_to_pdm_20 >= 0.80
                               and t.bc_pdm_to_idm_20 >= 0.80),
            "harmful": s.harmful_among_affected.round(3).to_dict(),
            "selective_extra_share": s.selective_extra_share.round(3).to_dict(),
            "overlap20": round(float(t.overlap20), 3),
            "best_compatible_20": {"idm->pdm": round(float(t.bc_idm_to_pdm_20), 3),
                                   "pdm->idm": round(float(t.bc_pdm_to_idm_20), 3)},
            "strict_pairs": {"n": int(t.strict_n_pairs_strict),
                             "disagreement": round(float(t.strict_disagreement), 4),
                             "gamma": round(float(t.strict_gamma), 4)}}
    (out / "phase0g_external_planner_stats.json").write_text(json.dumps(fals, indent=2))

    pd.set_option("display.width", 220)
    print("\n=== B7 per planner ===")
    print(S[["planner", "cost", "v_pos", "v_zero", "v_neg", "harmful_among_affected",
             "all_full_reduction", "oracle20_reduction", "selective_extra_share"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n=== B8 PDM-Closed vs IDMPlanner, same states, same cost ===")
    print(T[["cost", "idm_responsive", "pdm_responsive", "both_responsive", "strict_n_pairs_strict",
             "strict_disagreement", "strict_gamma", "overlap20", "bc_idm_to_pdm_20",
             "bc_pdm_to_idm_20"]].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n=== falsifiers ===")
    print(json.dumps(fals, indent=2))


if __name__ == "__main__":
    main()

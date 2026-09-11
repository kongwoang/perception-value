#!/usr/bin/env python
"""Render every figure in the Phase-0 report from a finished analysis run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import runmeta, viz                       # noqa: E402
from rap.paths import FIGURES, PROCESSED           # noqa: E402

MAIN_POLICIES = ["random", "lowest_confidence", "highest_uncertainty",
                 "criticality_heuristic", "learned_uncertainty", "learned_unc_crit", "oracle"]
LEARNED_POLICIES = ["random", "learned_uncertainty", "learned_all_visual",
                    "learned_unc_crit", "learned_all", "oracle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="", help="run dir; default = latest analysis run")
    ap.add_argument("--profile", default="", help="run dir; default = latest profile run")
    ap.add_argument("--table", default=str(PROCESSED / "frames_composite.pkl"))
    ap.add_argument("--out", default=str(FIGURES))
    args = ap.parse_args()

    run = Path(args.analysis) if args.analysis else runmeta.latest("analysis")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_pickle(args.table).reset_index(drop=True)
    made = []

    viz.value_heterogeneity(df, out / "fig1_value_heterogeneity.png")
    made.append("fig1_value_heterogeneity.png")

    try:
        prof = Path(args.profile) if args.profile else runmeta.latest("profile")
        summary = json.loads((prof / "profile_summary.json").read_text())
        order = [k for k in ["cheap_320", "cheap_384", "cheap_512", "full_640", "full_960"]
                 if k in summary]
        viz.compute_profile(summary, out / "fig2_compute_profile.png", order)
        made.append("fig2_compute_profile.png")
    except (FileNotFoundError, KeyError) as e:
        print("skipping compute profile figure:", e)

    pred = pd.read_csv(run / "predictability.csv")
    viz.predictability(pred, out / "fig3_predictability.png", model="gbm",
                       target="value_task", metric="spearman")
    made.append("fig3_predictability.png")
    if (pred.target == "value_task_pos").any():
        viz.predictability(pred, out / "fig3b_predictability_auc.png", model="gbm",
                           target="value_task_pos", metric="auc",
                           ylabel="out-of-fold AUC (does FULL help this frame?)")
        made.append("fig3b_predictability_auc.png")

    bud = pd.read_csv(run / "budget.csv")
    viz.budget_curves(bud, out / "fig4_budget.png",
                      [p for p in MAIN_POLICIES if p in set(bud.policy)])
    made.append("fig4_budget.png")
    viz.budget_curves(bud, out / "fig4b_budget_learned.png",
                      [p for p in LEARNED_POLICIES if p in set(bud.policy)])
    made.append("fig4b_budget_learned.png")
    if np.isfinite(bud.get("risk_per_1000_extra_ms", pd.Series([np.nan]))).any():
        viz.risk_per_ms(bud, out / "fig5_risk_per_ms.png",
                        [p for p in MAIN_POLICIES if p in set(bud.policy) and p != "random"])
        made.append("fig5_risk_per_ms.png")

    mp = run / "matched_pairs.pkl"
    if mp.exists():
        viz.matched_pairs(pd.read_pickle(mp), out / "fig6_matched_pairs.png")
        made.append("fig6_matched_pairs.png")

    print("wrote", len(made), "figures to", out)
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()

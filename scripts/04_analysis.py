#!/usr/bin/env python
"""Phase-0 analysis: heterogeneity, predictability, budgeted selection, matched pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, pairs, predict, runmeta, tables   # noqa: E402
from rap import features as F                             # noqa: E402
from rap.paths import PROCESSED                           # noqa: E402

QUOTAS = [0.10, 0.20, 0.30, 0.50]
ARMS = list(F.ARMS)


# ----------------------------------------------------------------------------- A. heterogeneity

def heterogeneity(df: pd.DataFrame) -> dict:
    v = df["value_task"].to_numpy()
    vv = df["value_visual"].to_numpy()
    order = np.argsort(-v)
    cum = np.cumsum(v[order])
    total = cum[-1]
    n = len(v)

    def frac_of_gain(p):
        return float(cum[max(int(round(p * n)) - 1, 0)] / total) if total > 1e-12 else np.nan

    gini_num = np.sum((2 * np.arange(1, n + 1) - n - 1) * np.sort(np.clip(v, 0, None)))
    gini = float(gini_num / (n * np.sum(np.clip(v, 0, None)))) if np.sum(np.clip(v, 0, None)) > 0 else np.nan

    return {
        "n_frames": int(n),
        "risk_cheap_total": float(df.risk_cheap.sum()),
        "risk_full_total": float(df.risk_full.sum()),
        "risk_reduction_total": float(total),
        "risk_reduction_rel": float(total / df.risk_cheap.sum()) if df.risk_cheap.sum() > 0 else np.nan,
        "frac_frames_value_gt0": float(np.mean(v > 1e-9)),
        "frac_frames_value_lt0": float(np.mean(v < -1e-9)),
        "frac_frames_value_zero": float(np.mean(np.abs(v) <= 1e-9)),
        "gain_in_top_5pct": frac_of_gain(0.05),
        "gain_in_top_10pct": frac_of_gain(0.10),
        "gain_in_top_20pct": frac_of_gain(0.20),
        "gain_in_top_30pct": frac_of_gain(0.30),
        "gini_value_task": gini,
        "spearman_value_task_vs_visual": float(stats.spearmanr(v, vv).correlation),
        "std_err_cheap_total": float(df.err_std_cheap.sum()),
        "std_err_full_total": float(df.err_std_full.sum()),
        "mean_n_gt": float(df.n_gt.mean()),
        "mean_crit_sum_gt": float(df.crit_sum_gt.mean()),
    }


# ----------------------------------------------------------------------------- B/C. predictability

def predictability(df: pd.DataFrame, models=("linear", "tree", "gbm"),
                   targets=(("value_task", "reg"), ("value_task_pos", "clf"),
                            ("value_visual", "reg"))) -> tuple:
    df = df.copy()
    df["value_task_pos"] = (df["value_task"] > 1e-9).astype(float)
    rows, oof = [], {}
    for target, task in targets:
        for arm in ARMS:
            cols = F.columns_for(arm)
            for mname in models:
                res = predict.loso(df, cols, mname, target, task, arm=arm)
                s = predict.score(res)
                oof[(arm, mname, target)] = res
                rows.append(s)
                key = "spearman" if task == "reg" else "auc"
                print(f"  {target:15s} {arm:16s} {mname:7s} "
                      f"{key}={s.get(key, float('nan')):+.3f} "
                      f"per-seq med={s['per_seq_median']:+.3f} "
                      f"({s['per_seq_pos']}/{s['per_seq_n']} better than chance)")
    return pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]), rows, oof


def arm_comparisons(rows: list, model: str, target: str) -> pd.DataFrame:
    by_arm = {r["arm"]: r for r in rows if r["model"] == model and r["target"] == target}
    tests = [("A_conf", "B_uncertainty"), ("B_uncertainty", "E_unc_crit"),
             ("C_complexity", "E_unc_crit"), ("F_all_visual", "G_all"),
             ("B_uncertainty", "D_criticality"), ("F_all_visual", "D_criticality"),
             ("F_all_visual", "H_visual_plus_stakes"), ("H_visual_plus_stakes", "G_all")]
    out = []
    for a, b in tests:
        if a in by_arm and b in by_arm:
            t = predict.paired_test(by_arm[a], by_arm[b])
            t.update({"baseline": a, "candidate": b, "model": model, "target": target,
                      "baseline_score": by_arm[a]["per_seq_median"],
                      "candidate_score": by_arm[b]["per_seq_median"]})
            out.append(t)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- D. budgeted selection

def specificity(rows: list, model: str) -> pd.DataFrame:
    """Is the criticality gain about *task risk*, or just about how much is in the scene?

    Value_visual is the same quantity with every criticality set to 1. A feature set
    that merely senses "busy scene" should help both targets equally; one that senses
    downstream consequence should help Value_task specifically.
    """
    out = []
    for a, b in [("B_uncertainty", "E_unc_crit"), ("F_all_visual", "G_all"),
                 ("C_complexity", "D_criticality"),
                 ("F_all_visual", "H_visual_plus_stakes"),
                 ("H_visual_plus_stakes", "G_all")]:
        r = {"baseline": a, "candidate": b, "model": model}
        for target in ("value_task", "value_visual"):
            by = {x["arm"]: x for x in rows if x["model"] == model and x["target"] == target}
            if a not in by or b not in by:
                break
            t = predict.paired_test(by[a], by[b])
            r[f"delta_{target}"] = t["delta_median"]
            r[f"p_{target}"] = t["wilcoxon_p"]
            r[f"nbetter_{target}"] = t["n_better"]
            r[f"n_{target}"] = t["n"]
        else:
            r["specificity"] = r["delta_value_task"] - r["delta_value_visual"]
            out.append(r)
    return pd.DataFrame(out)


def policy_scores(df: pd.DataFrame, oof: dict, model: str, target: str) -> dict:
    s = {
        "lowest_confidence": -df["feat_conf_mean"].to_numpy(),
        "lowest_min_confidence": -df["feat_conf_min"].to_numpy(),
        "highest_uncertainty": df["feat_binent_sum"].to_numpy(),
        "highest_entropy": df["feat_ent_sum"].to_numpy(),
        "scene_complexity": df["feat_n_det"].to_numpy(),
        "criticality_heuristic": df["feat_crit_sum"].to_numpy(),
        "riskweighted_uncertainty": df["feat_riskw_unc_sum"].to_numpy(),
    }
    for arm, label in [("B_uncertainty", "learned_uncertainty"),
                       ("D_criticality", "learned_criticality"),
                       ("E_unc_crit", "learned_unc_crit"),
                       ("F_all_visual", "learned_all_visual"),
                       ("G_all", "learned_all")]:
        key = (arm, model, target)
        if key in oof:
            s[label] = oof[key].pred
    return s


def budget_per_sequence(df: pd.DataFrame, scores: dict, quota: float) -> pd.DataFrame:
    """eta per held-out sequence, so consistency can be checked rather than assumed."""
    pos = {seq: np.flatnonzero(df["seq"].to_numpy() == seq) for seq in df["seq"].unique()}
    rows = []
    for seq, idx in pos.items():
        sub = df.iloc[idx].reset_index(drop=True)
        sc = {k: np.asarray(v)[idx] for k, v in scores.items()}
        r = budget.evaluate(sub, sc, [quota], seeds=16, mode="pooled")
        r["seq"] = seq
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(PROCESSED / "frames_composite.pkl"))
    ap.add_argument("--model", default="gbm")
    ap.add_argument("--lat_cheap", type=float, default=np.nan)
    ap.add_argument("--lat_full", type=float, default=np.nan)
    ap.add_argument("--energy_cheap", type=float, default=np.nan)
    ap.add_argument("--energy_full", type=float, default=np.nan)
    ap.add_argument("--tag", default="analysis")
    args = ap.parse_args()

    df = pd.read_pickle(args.table).reset_index(drop=True)
    tables.feature_columns(df)          # leakage guard on the real table
    run = runmeta.new_run(args.tag, vars(args))
    out = {}

    print("== A. heterogeneity of value-of-compute ==")
    het = heterogeneity(df)
    out["heterogeneity"] = het
    for k in ("frac_frames_value_gt0", "gain_in_top_10pct", "gain_in_top_20pct",
              "gini_value_task", "spearman_value_task_vs_visual", "risk_reduction_rel"):
        print(f"  {k:34s} {het[k]:.4f}")

    print("\n== B/C. predictability of Value_task from cheap-only features ==")
    pred_df, rows, oof = predictability(df)
    pred_df.to_csv(run / "predictability.csv", index=False)
    pd.DataFrame([{"arm": r["arm"], "model": r["model"], "target": r["target"],
                   "seq": k, "score": v}
                  for r in rows for k, v in r["_per_seq"].items()]
                 ).to_csv(run / "predictability_per_seq.csv", index=False)

    cmp_frames = []
    for target in ("value_task", "value_task_pos"):
        for m in ("linear", "tree", "gbm"):
            cmp_frames.append(arm_comparisons(rows, m, target))
    cmp_df = pd.concat(cmp_frames, ignore_index=True)
    cmp_df.to_csv(run / "arm_comparisons.csv", index=False)
    spec = pd.concat([specificity(rows, m) for m in ("linear", "gbm")], ignore_index=True)
    spec.to_csv(run / "specificity.csv", index=False)
    print("\n  specificity — criticality gain on task risk vs on plain detection value:")
    for _, r in spec.iterrows():
        print(f"    [{r.model}] {r.baseline:16s}->{r.candidate:14s} "
              f"dValue_task={r.delta_value_task:+.3f} (p={r.p_value_task:.3g})  "
              f"dValue_visual={r.delta_value_visual:+.3f} (p={r.p_value_visual:.3g})  "
              f"specificity={r.specificity:+.3f}")
    print("\n  paired tests over held-out sequences (gbm, value_task):")
    for _, r in cmp_df[(cmp_df.model == "gbm") & (cmp_df.target == "value_task")].iterrows():
        print(f"    {r.baseline:16s} -> {r.candidate:16s} d={r.delta_median:+.3f} "
              f"better in {r.n_better}/{r.n} seqs  p={r.wilcoxon_p:.4g}")

    print("\n== D. budgeted selection ==")
    scores = policy_scores(df, oof, args.model, "value_task")
    bud = []
    for mode in ("pooled", "per_sequence"):
        b = budget.evaluate(df, scores, QUOTAS, mode=mode)
        b = budget.add_compute_columns(b, args.lat_cheap, args.lat_full,
                                       args.energy_cheap, args.energy_full)
        bud.append(b)
    bud_df = pd.concat(bud, ignore_index=True)
    bud_df.to_csv(run / "budget.csv", index=False)
    piv = bud_df[bud_df["mode"] == "pooled"].pivot(index="policy", columns="quota", values="eta")
    print("  eta (share of the oracle's risk reduction captured):")
    print(piv.round(3).to_string())

    std_all_cheap = float(df.err_std_cheap.sum())
    std_all_full = float(df.err_std_full.sum())
    pooled = bud_df[bud_df["mode"] == "pooled"].copy()
    pooled["std_err_captured"] = (std_all_cheap - pooled["std_err_total"]) / \
        max(std_all_cheap - std_all_full, 1e-12)
    pooled.to_csv(run / "budget_pooled.csv", index=False)
    print("\n  same selections scored on the STANDARD detection metric "
          "(share of full-compute detection gain captured):")
    print(pooled.pivot(index="policy", columns="quota",
                       values="std_err_captured").round(3).to_string())
    out["standard_metric"] = {"std_err_all_cheap": std_all_cheap,
                              "std_err_all_full": std_all_full}

    print("\n== E. per-sequence consistency at 20% quota ==")
    per_seq = budget_per_sequence(df, scores, 0.20)
    per_seq.to_csv(run / "budget_per_sequence.csv", index=False)
    wide = per_seq.pivot(index="seq", columns="policy", values="eta")
    key_pairs = [("learned_uncertainty", "learned_unc_crit"),
                 ("highest_uncertainty", "criticality_heuristic"),
                 ("learned_all_visual", "learned_all")]
    consist = []
    for a, b in key_pairs:
        if a in wide and b in wide:
            d = (wide[b] - wide[a]).dropna()
            w = stats.wilcoxon(d, alternative="greater") if len(d) >= 5 and d.abs().sum() > 0 else None
            consist.append({"baseline": a, "candidate": b, "n_seq": int(len(d)),
                            "n_better": int((d > 0).sum()), "median_delta": float(d.median()),
                            "wilcoxon_p": float(w.pvalue) if w else np.nan})
            print(f"  {a:22s} -> {b:22s} better in {int((d>0).sum())}/{len(d)} seqs, "
                  f"median deta={d.median():+.3f}")
    out["consistency"] = consist

    print("\n== F. matched pairs ==")
    out["matched_pairs"] = {}
    for label, arm, caliper in [("uncertainty", "B_uncertainty", 0.25),
                                ("uncertainty_and_complexity", "F_all_visual", 0.25),
                                ("uncertainty_tight", "B_uncertainty", 0.12)]:
        pr = pairs.matched_pairs(df, F.columns_for(arm), caliper=caliper)
        summ = pairs.summarise(pr)
        out["matched_pairs"][label] = summ
        print(f"  matched on {label}: " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}))
        if label == "uncertainty":
            pairs.exemplars(df, pr, agree=True).to_csv(run / "pair_exemplars.csv", index=False)
            pairs.exemplars(df, pr, agree=False).to_csv(
                run / "pair_counterexamples.csv", index=False)
            pr.sample(min(len(pr), 200000), random_state=0).to_pickle(
                run / "matched_pairs.pkl")
        # negative control: the same test with a criticality-free scene-scale variable
        neg = pairs.matched_pairs(df, F.columns_for(arm), crit_col="feat_n_det",
                                  caliper=caliper)
        out["matched_pairs"][label + "__control_n_det"] = pairs.summarise(neg)

    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", run)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 0F Stage 1 -- do published planning-aware perception metrics already solve
compute allocation?

Joins per-sample PKL (CVPR 2020) and TIP (ICML 2023) gains onto the nuScenes decision
table from Phase 0E and scores every allocation signal by the same eta as the rest of the
project: the share of the decision-value oracle's achievable cost reduction that a signal
captures at a compute quota.

Reads the PKL/TIP CSVs written by 61_run_planning_metric.py (chunked runs are
concatenated) and writes results/final/phase0f_planning_metric_eta.csv.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rap import budget, percep_metrics as PM, predict, runmeta                  # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

from importlib import import_module                                             # noqa: E402
_pm = import_module("50_percep_metrics")

QUOTAS = [0.10, 0.20, 0.30, 0.50]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}
NOCHK = lambda c: None


def eta(df, score, col, quota=0.20):
    """Share of the decision oracle's achievable cost reduction captured at `quota`."""
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def eta_boot(df, score, col, quota, nboot=300, seed=0):
    """Scene-level bootstrap CI for eta.  Frames within a scene are strongly correlated,
    so scenes, not frames, are the resampling unit."""
    rng = np.random.default_rng(seed)
    scenes = df.seq.to_numpy()
    uniq = np.unique(scenes)
    idx_of = {u: np.flatnonzero(scenes == u) for u in uniq}
    sc = None if score is None else np.asarray(score, float)
    vals = []
    for b in range(nboot):
        take = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        db = df.iloc[take]
        s_ = (np.random.default_rng(1000 + b).random(len(db)) if sc is None else sc[take])
        v = eta(db, s_, col, quota)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < nboot // 4:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def topk_overlap(a, b, frac):
    k = int(round(frac * len(a)))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / max(k, 1)


def inversion_rate(a, b, rng, npairs=200_000):
    """Fraction of frame pairs the two signals order differently (ties excluded)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), npairs)
    j = rng.integers(0, len(a), npairs)
    k = i != j
    i, j = i[k], j[k]
    sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    ok = (sa != 0) & (sb != 0)
    return float((sa[ok] != sb[ok]).mean()) if ok.any() else np.nan


def load_metric(subs: Path, metric: str, variant: str) -> pd.DataFrame | None:
    """One CSV per chunk, or a single unchunked CSV."""
    parts = sorted(subs.glob(f"{metric}_{variant}_c*.csv"))
    single = subs / f"{metric}_{variant}.csv"
    if not parts and single.exists():
        parts = [single]
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("sample_token", keep="first")
    return df


def token_map(seqs, dataroot, version) -> pd.DataFrame:
    """(scene name, frame) -> sample_token, in the same temporal order NuScenesDB uses.

    Reads scene.json and sample.json directly rather than instantiating NuScenesDB: the
    full table set costs ~3 GB, which the concurrent PKL/TIP runs cannot spare.
    """
    cache = Path(CACHE) / "nusc_token_map.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if set(seqs) <= set(df.seq.unique()):
            return df[df.seq.isin(set(seqs))].reset_index(drop=True)
    t = Path(dataroot) / version
    scenes = json.loads((t / "scene.json").read_text())
    samples = {r["token"]: r for r in json.loads((t / "sample.json").read_text())}
    want = set(seqs)
    rows = []
    for sc in scenes:
        if sc["name"] not in want:
            continue
        tok, fr = sc["first_sample_token"], 0
        while tok:
            rows.append((sc["name"], fr, tok))
            tok, fr = samples[tok]["next"], fr + 1
    df = pd.DataFrame(rows, columns=["seq", "frame", "sample_token"])
    df.to_csv(cache, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "results/raw/20260912_071140_core_matrix"),
                    help="core_matrix run holding the pickled nuScenes decision tables")
    ap.add_argument("--subs", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--variant", default="oracle", help="submission geometry variant")
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--tag", default="phase0f_eta")
    ap.add_argument("--skip_multimetric", action="store_true")
    ap.add_argument("--nboot", type=int, default=300)
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    subs = Path(args.subs)
    rng = np.random.default_rng(0)

    # Phase 0E's nuScenes table.  The geometry variant of the decision table and of the
    # submission are kept the same so PKL/TIP see the boxes the planner saw.
    tbl = Path(args.tables) / \
        f"nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{args.variant}.pkl"
    d = pd.read_pickle(tbl)
    print(f"decision table {tbl.name}: {len(d)} frames, {d.seq.nunique()} scenes")

    d = d.merge(token_map(sorted(d.seq.unique()), args.dataroot, args.version),
                on=["seq", "frame"],
                how="left", validate="one_to_one")
    assert d.sample_token.notna().all(), "unmapped frames"

    have = {}
    for m in ("pkl", "tip"):
        g = load_metric(subs, m, args.variant)
        if g is None:
            print(f"  {m.upper()}: no CSV yet -- skipped")
            continue
        col = f"G_{m.upper()}"
        d = d.merge(g[["sample_token", col]], on="sample_token", how="left")
        cov = float(d[col].notna().mean())
        print(f"  {m.upper()}: {len(g)} samples, coverage {cov:.3f}")
        have[m] = col

    # A frame with no published-metric score cannot be ranked by it; restricting every
    # signal to the covered subset keeps the comparison like-for-like.
    if have:
        keep = np.ones(len(d), bool)
        for col in have.values():
            keep &= d[col].notna().to_numpy()
        print(f"  common covered subset: {int(keep.sum())}/{len(d)} frames")
        d = d[keep].reset_index(drop=True)

    for tname, col in TASKS.items():
        d[f"_dJ_{tname}"] = (d[col[0]] - d[col[1]]).to_numpy()

    # Wiring validation.  Before believing a low eta, check the metrics are connected the
    # way their papers define them: the per-frame *level* must rise with that frame's error
    # count.  If it does and the cheap-minus-full *difference* still fails to rank frames by
    # dJ, the failure is in the signal, not in the plumbing.
    wire = []
    for m, gcol in have.items():
        for a, b, why in [(f"{m}_cheap", "cheap_fn", "more missed objects -> worse score"),
                          (f"{m}_full", "full_fn", "more missed objects -> worse score"),
                          (f"{m}_cheap", "n_gt", "more objects present -> more to get wrong"),
                          (f"{m}_cheap", "cheap_fp", "more false positives -> worse score"),
                          (gcol, "dE", "does the gain track the exact perception gain"),
                          (gcol, "crit_sum", "does the gain track downstream criticality")]:
            if a in d.columns and b in d.columns:
                wire.append({"metric": m, "x": a, "y": b, "expect": why,
                             "spearman": float(stats.spearmanr(d[a], d[b]).correlation)})
    if wire:
        w = pd.DataFrame(wire)
        w.to_csv(run / "wiring_validation.csv", index=False)
        print("\n=== wiring validation (levels must track error counts) ===")
        print(w[["x", "y", "spearman", "expect"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))

    # How much signal there is to recover at all: dJ is zero wherever the cheap and the
    # expensive mode lead to the same action and the same cost.
    print("\n=== decision-signal density ===")
    for tname, col in TASKS.items():
        dj = (d[col[0]] - d[col[1]]).to_numpy()
        act = "same_action" if tname == "longitudinal" else "lat_same_action"
        print(f"  {tname:13s} |dJ|>0 on {int((np.abs(dj) > 1e-9).sum())}/{len(d)} frames"
              f"  ({float((np.abs(dj) > 1e-9).mean()):.3f})"
              f"   action changes {int((d[act] == 0).sum())}")

    rows, curves = [], []
    for tname, col in TASKS.items():
        dj = d[f"_dJ_{tname}"].to_numpy()
        best_metric = max(PM.METRICS, key=lambda m: eta(d, d[f"dE_{m}"], col))

        signals = {
            "random": None,                                   # averaged over seeds below
            "uncertainty": d.unc_sum.to_numpy(),
            "criticality": d.crit_sum.to_numpy(),
            "perception_oracle_dE": d["dE"].to_numpy(),
            f"best_dE_metric ({best_metric})": d[f"dE_{best_metric}"].to_numpy(),
        }
        if not args.skip_multimetric:
            feats = _pm.oracle_features(d)
            dd2 = d.copy()
            for c in feats:
                dd2[c] = pd.to_numeric(dd2[c], errors="coerce").fillna(0.0)
            dd2["_dJ"] = dj
            signals["multimetric_dE_oracle"] = predict.loso(
                dd2, feats, "gbm", "_dJ", "reg", checker=NOCHK).pred
        for m, gcol in have.items():
            signals[f"{m.upper()}_gain"] = d[gcol].to_numpy()
        signals["decision_oracle_dJ"] = dj

        for name, sc in signals.items():
            r = {"task": tname, "variant": args.variant, "signal": name,
                 "n_frames": len(d), "n_scenes": int(d.seq.nunique())}
            for q in QUOTAS:
                if sc is None:
                    v = float(np.mean([eta(d, np.random.default_rng(k).random(len(d)), col, q)
                                       for k in range(16)]))
                else:
                    v = eta(d, sc, col, q)
                r[f"eta_{int(q * 100)}"] = v
                lo, hi = eta_boot(d, sc, col, q, nboot=args.nboot)
                r[f"eta_{int(q * 100)}_lo"], r[f"eta_{int(q * 100)}_hi"] = lo, hi
                curves.append({"task": tname, "variant": args.variant, "signal": name,
                               "quota": q, "eta": v, "eta_lo": lo, "eta_hi": hi})
            if sc is not None:
                r["spearman_vs_dJ"] = float(stats.spearmanr(sc, dj).correlation)
                r["kendall_vs_dJ"] = float(stats.kendalltau(sc, dj).correlation)
                r["inversion_rate_vs_dJ"] = inversion_rate(sc, dj, rng)
                r["top10_overlap_vs_dJ"] = topk_overlap(sc, dj, 0.10)
                r["top20_overlap_vs_dJ"] = topk_overlap(sc, dj, 0.20)
            rows.append(r)

    out = Path(RESULTS) / "final"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(out / "phase0f_planning_metric_eta.csv", index=False)
    m.to_csv(run / "phase0f_planning_metric_eta.csv", index=False)
    pd.DataFrame(curves).to_csv(run / "phase0f_regret_curves.csv", index=False)
    d.to_pickle(run / "joined_frames.pkl")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    for tname in TASKS:
        s = m[m.task == tname]
        print(f"\n=== {tname} ({args.variant}) ===")
        s = s.copy()
        s["eta_20_ci"] = [f"[{a:+.2f},{b:+.2f}]" if np.isfinite(a) else "--"
                          for a, b in zip(s.eta_20_lo, s.eta_20_hi)]
        print(s[["signal", "eta_10", "eta_20", "eta_20_ci", "eta_30", "eta_50",
                 "spearman_vs_dJ", "kendall_vs_dJ", "inversion_rate_vs_dJ",
                 "top20_overlap_vs_dJ"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # kill test 1: a published planning-aware metric recovering >=0.8 at 20% quota
    kill = m[m.signal.str.contains("PKL|TIP")][["task", "signal", "eta_20"]]
    if len(kill):
        kill = m[m.signal.str.contains("PKL|TIP")][
            ["task", "signal", "eta_10", "eta_20", "eta_20_lo", "eta_20_hi",
             "eta_30", "eta_50"]]
        worst = kill.eta_20.max()
        print(f"\nKILL TEST 1  max eta@20 over published planning-aware metrics: {worst:+.3f}"
              f"  -> {'STOP (metrics already solve it)' if worst >= 0.8 else 'gap survives'}")
        (run / "kill_test_1.json").write_text(json.dumps(
            {"max_eta20_published": float(worst), "threshold": 0.8,
             "verdict": "STOP" if worst >= 0.8 else "PROCEED",
             "rows": kill.to_dict("records")}, indent=2))
    print("\nwrote", run)


if __name__ == "__main__":
    main()

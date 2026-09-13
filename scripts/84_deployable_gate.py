#!/usr/bin/env python
"""Phase 0G, experiment 2: is decision value learnable BEFORE escalating?

Every signal in the Phase 0F/0G tables that clears random is a diagnostic: exact dE, the dE
variants, the multi-metric oracle, PKL and TIP all need the expensive output to compute, and
`crit_sum` needs ground-truth geometry.  None could decide whether to run the expensive detector.
This script closes that gap with the minimum that makes the benchmark idea concrete, and claims no
method: predict V^q from cheap-side information only, out of fold by scene, and report how much of
the oracle's achievable benefit that captures.

Information policy, enforced in code rather than stated:
  * features come only from `rap.features.frame_features` on the CHEAP detections and the
    geometry predicted from them -- the same builder the KITTI phases used;
  * `rap.tables.feature_columns` runs `assert_no_leakage`, which refuses any column not registered
    with a cheap-side source; it is not bypassed here, unlike the diagnostic multi-metric oracle;
  * folds are leave-one-scene-out, so no frame of a test scene is ever seen in training.

Features do not depend on the geometry variant (the variant changes how *decisions* lift boxes,
not what the cheap detector saw), so they are built once; the targets differ per variant.
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import budget, features as F, geometry as G, predict, runmeta         # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.tables import feature_columns                                          # noqa: E402

QUOTAS = [0.10, 0.20, 0.30]
TIE_SEEDS = 8


def build_features(det_dir: Path, mode: str, cfg: RiskConfig) -> pd.DataFrame:
    model = G.PRIMARY
    rows = []
    for f in sorted((det_dir / mode).glob("*.npz")):
        c = DetCache(f)
        for i, fr in enumerate(c.frames):
            det, geo = c.det(i), c.geo(i)
            try:
                sc = c.scalars(i)
            except Exception:                                   # noqa: BLE001
                sc = {}
            img_area = sc.get("img_w", 1600.0) * sc.get("img_h", 900.0)
            img = {k: v for k, v in sc.items()
                   if k.startswith(("img_bright", "img_dark", "img_lap", "img_edge",
                                    "img_contrast", "motion"))}
            rec = {"seq": f.stem, "frame": int(fr)}
            rec.update(F.frame_features(det, geo, img, img_area, model, cfg.op_conf))
            rows.append(rec)
    return pd.DataFrame(rows)


def eta_parts(d, score, ccol, fcol, q):
    col = (ccol, fcol)
    dj = (d[ccol] - d[fcol]).to_numpy()
    sc = np.asarray(score, float)
    allc = float(d[ccol].sum())
    orc = np.mean([budget.total_risk(d, budget.select_pooled(dj, q, seed=s), col)
                   for s in range(TIE_SEEDS)])
    tot = np.mean([budget.total_risk(d, budget.select_pooled(sc, q, seed=s), col)
                   for s in range(TIE_SEEDS)])
    return allc - tot, allc - orc, allc


def eta(d, score, ccol, fcol, q):
    got, prize, _ = eta_parts(d, score, ccol, fcol, q)
    return got / prize if prize > 1e-12 else np.nan


def boot_ci(d, score, ccol, fcol, q, nboot, rng):
    scenes = d.seq.to_numpy()
    uniq = np.unique(scenes)
    idx = {u: np.flatnonzero(scenes == u) for u in uniq}
    _, prize0, _ = eta_parts(d, np.zeros(len(d)), ccol, fcol, q)
    sc = np.asarray(score, float)
    vals, dropped = [], 0
    for _ in range(nboot):
        t = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        got, prize, _ = eta_parts(d.iloc[t], sc[t], ccol, fcol, q)
        if prize <= max(1e-12, 0.25 * prize0):
            dropped += 1; continue
        vals.append(got / prize)
    if len(vals) < nboot // 4:
        return np.nan, np.nan, dropped
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_tv"))
    ap.add_argument("--mode", default="ns_cheap_320")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--tag", default="deployable_gate")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    rng = np.random.default_rng(0)
    cfg = RiskConfig()

    feats = build_features(Path(args.det), args.mode, cfg)
    fcols = feature_columns(feats)                      # assert_no_leakage runs here
    print(f"  {len(feats)} frames, {feats.seq.nunique()} scenes, {len(fcols)} cheap-side features "
          f"(leakage guard passed)")

    tm = pd.read_csv(Path(CACHE) / "nusc_token_map.csv")
    cm = sorted(glob.glob(str(ROOT / "results/raw/*core_matrix_postreview")))
    cm = Path(cm[-1]) if cm else ROOT / "results/raw/20260912_071140_core_matrix"
    print(f"  decision tables from {cm.name}")

    rows = []
    for variant in ("oracle", "mono"):
        b = pd.read_pickle(cm / f"nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{variant}.pkl")
        sfx = "" if variant == "oracle" else f"_{variant}"
        p = pd.read_csv(Path(CACHE) / "planner_d" / f"planC_vs_truth{sfx}.csv")
        base = b[["seq", "frame", "J_cheap", "J_full"]].merge(feats, on=["seq", "frame"],
                                                               how="inner", validate="one_to_one")
        base = base.merge(tm, on=["seq", "frame"], how="left")
        targets = {
            "brake": (base, "J_cheap", "J_full"),
            "plan": (base.merge(p[["sample_token", "JC_ade_cheap", "JC_ade_full"]],
                                on="sample_token", how="inner"), "JC_ade_cheap", "JC_ade_full"),
        }
        for tname, (d, ccol, fcol) in targets.items():
            d = d.reset_index(drop=True).copy()
            d["_V"] = d[ccol] - d[fcol]
            for c in fcols:
                d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0.0)
            signals = {"random": None,
                       "uncertainty (cheap)": d["feat_ent_mean"].to_numpy()
                       if "feat_ent_mean" in d else None}
            for mdl in ("linear", "gbm"):
                signals[f"{mdl} (OOF by scene)"] = predict.loso(
                    d, fcols, mdl, "_V", "reg").pred                 # guard ON: default checker
            signals["decision oracle"] = d["_V"].to_numpy()
            for sname, sc in signals.items():
                for q in QUOTAS:
                    if sc is None and sname == "random":
                        v = float(np.mean([eta(d, np.random.default_rng(s).random(len(d)),
                                               ccol, fcol, q) for s in range(16)]))
                        lo = hi = np.nan; dr = 0
                    elif sc is None:
                        continue
                    else:
                        v = eta(d, sc, ccol, fcol, q)
                        lo, hi, dr = boot_ci(d, sc, ccol, fcol, q, args.nboot, rng)
                    rows.append({"geometry": variant, "target": tname, "signal": sname,
                                 "deployable": sname not in ("decision oracle",),
                                 "quota": q, "eta": v, "lo": lo, "hi": hi,
                                 "boot_dropped": dr, "n_frames": len(d),
                                 "n_scenes": int(d.seq.nunique())})
                print(f"  {variant:6s} {tname:5s} {sname:22s} " + "  ".join(
                    f"@{int(r['quota']*100)} {r['eta']:+.3f}" for r in rows[-len(QUOTAS):]))

    m = pd.DataFrame(rows)
    out = Path(RESULTS) / "final" / "phase0g_deployable_gate.csv"
    m.to_csv(out, index=False); m.to_csv(run / out.name, index=False)
    print("  wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""End-to-end plumbing check on real GT labels with simulated detections.

The simulator is deliberately built so that FULL's advantage depends on object
*size* (small objects are dropped by CHEAP), which is correlated with distance but
not with criticality — so a correct pipeline should find some criticality signal
but not a fabricated one. It exercises every stage without needing the images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import budget, cache, features as F, geometry as G, kitti, mono, predict, tables
from rap.cache import DetCache
from rap.risk import RiskConfig


def simulate(seq: str, drop_scale: float, jitter: float, fp_rate: float,
             seed: int, calib) -> tuple:
    """Detections whose recall falls off with apparent box height."""
    rng = np.random.default_rng(seed)
    lab = kitti.load_labels(seq)
    lab = lab[np.isin(lab["type"], kitti.EVAL_TYPES)]
    frames = sorted(set(int(f) for f in lab["frame"]))
    dets, geos, scal = [], [], {}
    prev = None
    for fr in frames:
        rows = lab[lab["frame"] == fr]
        h = rows["y2"] - rows["y1"]
        keep = rng.random(len(rows)) < 1.0 / (1.0 + np.exp(-(h - drop_scale) / 12.0))
        r = rows[keep]
        n = len(r)
        xyxy = np.stack([r["x1"], r["y1"], r["x2"], r["y2"]], 1).astype(np.float32)
        if n:
            xyxy += rng.normal(0, jitter, xyxy.shape).astype(np.float32)
        nf = rng.poisson(fp_rate)
        if nf:
            fx = rng.uniform(0, 1100, nf); fy = rng.uniform(140, 300, nf)
            fw = rng.uniform(20, 90, nf); fh = rng.uniform(20, 70, nf)
            xyxy = np.concatenate([xyxy, np.stack([fx, fy, fx + fw, fy + fh], 1).astype(np.float32)])
        m = len(xyxy)
        conf = rng.uniform(0.12, 0.98, m).astype(np.float32)
        d = {"xyxy": xyxy, "conf": conf,
             "coarse": np.array(["vehicle"] * m),
             "entropy": rng.uniform(0, 1.2, m).astype(np.float32),
             "margin": rng.uniform(0, 1, m).astype(np.float32),
             "binent": (-conf * np.log(conf) - (1 - conf) * np.log(1 - conf)).astype(np.float32),
             "n_cand": float(m * 3), "n_cand_raw": float(m * 4), "n_post": float(m)}
        dets.append(d)
        geos.append(mono.predicted_geometry(d, prev, calib))
        prev = d
        for k in ("n_cand", "n_cand_raw", "n_post"):
            scal.setdefault(k, []).append(d[k])
    return frames, dets, geos, scal


def main():
    out = Path("/tmp/rap_smoke/det")
    seqs = kitti.sequences()[:8]
    for seq in seqs:
        calib = kitti.load_calib(seq)
        for mode, ds, jit, fp, seed in [("cheap_320", 46.0, 3.0, 1.2, 1),
                                        ("full_640", 22.0, 1.6, 0.7, 2)]:
            frames, dets, geos, scal = simulate(seq, ds, jit, fp, seed + hash(seq) % 1000, calib)
            if mode == "cheap_320":
                rng = np.random.default_rng(7)
                for k, v in {"img_bright_mean": 110, "img_bright_std": 50, "img_dark_frac": .1,
                             "img_bright_frac": .02, "img_lap_var": 300, "img_edge_density": .1,
                             "img_contrast_p95_p5": 150, "motion_mean": 8, "motion_p95": 30,
                             "motion_first": 0, "img_h": 375, "img_w": 1242}.items():
                    scal[k] = list(rng.normal(v, abs(v) * .15 + 1e-6, len(frames)))
            cache.save(out / mode / f"{seq}.npz", frames, dets, geos, scal)

    cfg = RiskConfig()
    df = pd.concat([tables.build_sequence(
        s, DetCache(out / "cheap_320" / f"{s}.npz"), DetCache(out / "full_640" / f"{s}.npz"),
        G.PRIMARY, cfg) for s in seqs], ignore_index=True)
    print(f"frames={len(df)}  value_task>0 {(df.value_task>1e-9).mean():.1%}  "
          f"risk_cheap={df.risk_cheap.mean():.3f} risk_full={df.risk_full.mean():.3f}")
    tables.feature_columns(df)
    Path("/tmp/rap_smoke").mkdir(parents=True, exist_ok=True)
    df.to_pickle("/tmp/rap_smoke/frames_synthetic.pkl")

    oof = {}
    for arm in ["B_uncertainty", "D_criticality", "E_unc_crit", "G_all"]:
        r = predict.loso(df, F.columns_for(arm), "gbm", "value_task", "reg", arm=arm)
        oof[arm] = r
        s = predict.score(r)
        print(f"  {arm:16s} rho={s['spearman']:+.3f} per-seq median={s['per_seq_median']:+.3f}")
    print("  paired B->E:", predict.paired_test(predict.score(oof["B_uncertainty"]),
                                                predict.score(oof["E_unc_crit"])))

    scores = {"highest_uncertainty": df.feat_binent_sum.to_numpy(),
              "criticality_heuristic": df.feat_crit_sum.to_numpy(),
              "learned_uncertainty": oof["B_uncertainty"].pred,
              "learned_unc_crit": oof["E_unc_crit"].pred}
    b = budget.evaluate(df, scores, [0.1, 0.2, 0.3, 0.5])
    b = budget.add_compute_columns(b, 6.8, 12.0)
    print(b.pivot(index="policy", columns="quota", values="eta").round(3).to_string())

    from rap import pairs, viz
    pr = pairs.matched_pairs(df, F.columns_for("B_uncertainty"))
    print("  matched pairs:", {k: (round(v, 4) if isinstance(v, float) else v)
                               for k, v in pairs.summarise(pr).items()})
    figdir = Path("/tmp/rap_smoke/fig"); figdir.mkdir(parents=True, exist_ok=True)
    viz.value_heterogeneity(df, figdir / "het.png")
    viz.budget_curves(b, figdir / "budget.png",
                      ["random", "highest_uncertainty", "criticality_heuristic",
                       "learned_uncertainty", "learned_unc_crit", "oracle"])
    if len(pr) > 100:
        viz.matched_pairs(pr, figdir / "pairs.png")
    print("figures ->", sorted(p.name for p in figdir.glob("*.png")))


if __name__ == "__main__":
    main()

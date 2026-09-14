#!/usr/bin/env python
"""Task 2, R1: a detection-list router (ORIC-style) on the frozen benchmark split.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 2 pre-registration").

Input per frame: every cached CHEAP detection (conf >= 0.10, the engine floor), top 25 by
confidence, each as [conf, x1/W, y1/H, x2/W, y2/H, one-hot(vehicle, person, cyclist), area/(W*H)],
zero-padded to 225 dims and built with vectorised numpy.  nuPlan has no confidence: the CHEAP-kept
tracks at the decision iteration, top 25 by distance, each as [presence, x/80, y/40, l/10, w/5,
one-hot(vehicle, pedestrian, bicycle, static), area/50] (250 dims, `104_nuplan_track_lists.py`).

Models: MLP (scikit-learn, 2 x 64 ReLU, StandardScaler, alpha 1e-4, 300 iterations, seed 0) and the
fixed GBM of `predict.make_model`; each for V (regression) and 1[V>0] (classification), so four
score rows per cell.  Fit on train + val units, scored once on test with the benchmark's exact tie
expectation and unit bootstrap paired against random (`92_benchmark_table.evaluate`).
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta                                                # noqa: E402
from rap.cache import COARSE_ID, DetCache                                       # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

_s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
t92 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t92)

TOPK = 25
DET_DIM = 9
IMG_WH = {"nuScenes": (1600.0, 900.0), "KITTI": (1242.0, 375.0)}


def det_list_features(det, img_w, img_h):
    """One frame's CHEAP detection list as a fixed 225-dim vector (vectorised)."""
    conf = det["conf"].astype(np.float64)
    o = np.argsort(-conf, kind="stable")[:TOPK]
    n = len(o)
    x = np.zeros((TOPK, DET_DIM))
    if n:
        b = det["xyxy"][o].astype(np.float64)
        x[:n, 0] = conf[o]
        x[:n, 1:5] = b / np.array([img_w, img_h, img_w, img_h])
        cid = np.array([COARSE_ID[str(c)] for c in det["coarse"][o]])
        x[np.arange(n), 5 + cid] = 1.0
        x[:n, 8] = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / (img_w * img_h)
    return x.reshape(-1)


def frame_matrix(det_dir, mode, dataset):
    rows, keys = [], []
    for f in sorted((Path(det_dir) / mode).glob("*.npz")):
        c = DetCache(f)
        for i, fr in enumerate(c.frames):
            sc = c.scalars(i)
            w, h = sc.get("img_w", IMG_WH[dataset][0]), sc.get("img_h", IMG_WH[dataset][1])
            rows.append(det_list_features(c.det(i), w, h))
            keys.append((f.stem, int(fr)))
    k = pd.DataFrame(keys, columns=["seq", "frame"])
    return k, np.asarray(rows)


def models():
    return {"R1_mlp_reg": ("reg", lambda: Pipeline([("s", StandardScaler()), ("m", MLPRegressor(
                hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=300, random_state=0))])),
            "R1_mlp_clf": ("clf", lambda: Pipeline([("s", StandardScaler()), ("m", MLPClassifier(
                hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=300, random_state=0))])),
            "R1_gbm_reg": ("reg", lambda: predict.make_model("gbm", "reg", 0)),
            "R1_gbm_clf": ("clf", lambda: predict.make_model("gbm", "clf", 0))}


def fit_score(X, v, fit):
    out = {}
    for name, (task, make) in models().items():
        y = v if task == "reg" else (v > t92.EPS).astype(int)
        if task == "clf" and len(np.unique(y[fit])) < 2:
            out[name] = np.zeros(len(v))                  # one class in training: no ranking information
            continue
        m = make().fit(X[fit], y[fit])
        out[name] = m.predict(X) if task == "reg" else m.predict_proba(X)[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--tag", default="routers_r1")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    rng = np.random.default_rng(0)

    mats = {"nuScenes": frame_matrix(CACHE / "nusc_det_tv", "ns_cheap_320", "nuScenes"),
            "KITTI": frame_matrix(CACHE / "det", "cheap_320", "KITTI")}
    trk = sorted(glob.glob(str(ROOT / "results" / "raw" / "*_nuplan_track_lists" / "nuplan_track_features.npz")))
    rows = []
    for gen in (t92.nuscenes_cells, t92.kitti_cells, t92.nuplan_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            if c["track"] == "nuPlan":
                if not trk:
                    print("  nuPlan track lists missing -- run 104 first"); continue
                z = np.load(trk[-1], allow_pickle=False)
                kk = pd.DataFrame({"scenario": z["scenario"].astype(str), "iteration": z["iteration"].astype(int),
                                   "_row": np.arange(len(z["scenario"]))})
                j = d[["scenario", "iteration"]].merge(kk, on=["scenario", "iteration"], how="left", validate="one_to_one")
                assert j._row.notna().all()
                X = z["X"][j._row.to_numpy(int)]
            else:
                keys, M = mats[c["track"]]
                kk = keys.assign(_row=np.arange(len(keys)))
                j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(kk, on=["seq", "frame"], how="left",
                                                                                 validate="one_to_one")
                assert j._row.notna().all()
                X = M[j._row.to_numpy(int)]
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            fit = d.split.isin(["train", "val"]).to_numpy()
            test = (d.split == "test").to_numpy()
            t0 = time.time()
            scores = fit_score(X, v, fit)
            ks, prize0, point, draws, dropped = t92.evaluate(v[test], d.unit.to_numpy()[test],
                                                             {"random": None, **{k: s[test] for k, s in scores.items()}},
                                                             args.nboot, rng)
            key = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"])
            for s in scores:
                for qi, q in enumerate(t92.QUOTAS):
                    dr = draws[s][:, qi] - draws["random"][:, qi]
                    lo, hi = t92.ci(draws[s][:, qi])
                    dlo, dhi = t92.ci(dr)
                    rows.append({**key, "split": "test", "signal": s, "deployable": True, "input_dim": X.shape[1],
                                 "quota": q, "k": int(ks[qi]), "eta": float(point[s]["eta"][qi]), "eta_lo": lo, "eta_hi": hi,
                                 "minus_random": float(np.nanmean(dr)), "minus_random_lo": dlo, "minus_random_hi": dhi,
                                 "p_le_random": float(np.mean(dr[np.isfinite(dr)] <= 0)),
                                 "tie_frac": float(point[s]["tie"][qi]), "responsive_frac": float(point[s]["resp"][qi]),
                                 "boot_dropped": int(dropped[qi]), "n_units": int(len(np.unique(d.unit[test]))),
                                 "n_frames": int(test.sum())})
            r20 = [r for r in rows[-len(scores) * len(t92.QUOTAS):] if r["quota"] == 0.2]
            print(f"  {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:8s} @20 " +
                  "  ".join(f"{r['signal'][3:]}={r['eta']:+.2f}({r['minus_random_lo']:+.2f})" for r in r20) +
                  f"  [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    out = Path(RESULTS) / "final" / "benchmark_table_routers.csv"
    if out.exists():                                        # R2 rows written by 107 are kept
        old = pd.read_csv(out)
        df = pd.concat([old[~old.signal.str.startswith("R1_")], df], ignore_index=True)
    df.to_csv(out, index=False)
    df.to_csv(run / out.name, index=False)
    print("  wrote", out)


if __name__ == "__main__":
    main()

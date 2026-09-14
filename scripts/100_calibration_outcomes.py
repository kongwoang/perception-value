#!/usr/bin/env python
"""Task 1: per-mode operating points -- precision/recall curves, S1-S3 thresholds, per-mode outcomes.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 1 pre-registration").  Question: does harm
survive when each fidelity has its own operating point instead of one shared op_conf?

  Stage A  pooled precision / recall / F1 / detections per frame for every mode on a 0.01 grid
           (0.10-0.70), for train+val, test and all units.  Caches floor at 0.10, so nothing lower.
  Stage B  S1 (count-matched), S2 (F1-optimal on the 0.05 grid) and S3 (precision-matched) from
           train+val only.  S4 needs downstream losses and is chosen in 102.
  Stage C  the unchanged decision-value pipeline -- decision.build, add_perception_gain,
           50.primitives and, for KITTI, 65.build_b -- run for every pair and geometry at each
           threshold of the 0.05 grid and at the S1/S3 FULL thresholds, both modes at that
           threshold.  No quantity couples the two modes, so a cell at (t_c, t_f) is composed from
           the CHEAP columns of the t_c run and the FULL columns of the t_f run.
  Checks   1: the 0.25 runs reproduce the recorded core-matrix and Planner B tables exactly.
           2: direct asymmetric runs at (0.15, 0.45) and (0.45, 0.15) equal the composition.
           Both must pass; 102 refuses to score anything otherwise.

nuScenes q_plan is 101 (it needs the devkit and the GPU); cells, sweep and S4 are 102.
"""
from __future__ import annotations

import argparse, json, sys, time
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import percep_metrics as PM, runmeta                                   # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.tables import match_detection_counts                                   # noqa: E402

RAW = ROOT / "results" / "raw"
CORE = RAW / "20260913_133004_core_matrix_postreview"
PLANB = RAW / "20260912_111225_planner_b_static_fixed"
NUSC_ROOT, NUSC_VER = "/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval"
GRID05 = [round(0.10 + 0.05 * i, 2) for i in range(13)]
GRID01 = [round(0.10 + 0.01 * i, 2) for i in range(61)]
CHECK_PAIRS = [(0.15, 0.45), (0.45, 0.15)]
CHECK_SPECS = ("nusc_mono", "kitti_y8_384_mono")

# key: (dataset, det_dir, cheap, full, geometry, recorded core-matrix table, recorded Planner B table)
SPECS = {
    "nusc_oracle": ("nuScenes", "nusc_det_tv", "ns_cheap_320", "ns_full_640", "oracle",
                    "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__oracle.pkl", None),
    "nusc_mono": ("nuScenes", "nusc_det_tv", "ns_cheap_320", "ns_full_640", "mono",
                  "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__mono.pkl", None),
    "kitti_y8_320_mono": ("KITTI", "det", "cheap_320", "full_640", "mono",
                          "KITTI__YOLOv8s__cheap_320tofull_640__mono.pkl",
                          "planB__KITTI__YOLOv8s__cheap_320tofull_640__mono.pkl"),
    "kitti_y8_320_oracle": ("KITTI", "det", "cheap_320", "full_640", "oracle",
                            "KITTI__YOLOv8s__cheap_320tofull_640__oracle.pkl",
                            "planB__KITTI__YOLOv8s__cheap_320tofull_640__oracle.pkl"),
    "kitti_y8_384_mono": ("KITTI", "det", "cheap_384", "full_640", "mono",
                          "KITTI__YOLOv8s__cheap_384tofull_640__mono.pkl",
                          "planB__KITTI__YOLOv8s__cheap_384tofull_640__mono.pkl"),
    "kitti_y8_512_mono": ("KITTI", "det512", "cheap_512", "full_640", "mono",
                          "KITTI__YOLOv8s__cheap_512tofull_640__mono.pkl",
                          "planB__KITTI__YOLOv8s__cheap_512tofull_640__mono.pkl"),
    "kitti_rt_480_mono": ("KITTI", "rtdetr_kitti_mid", "rt_mid_480", "rt_full_640", "mono",
                          "KITTI__RT-DETR-l__rt_mid_480tort_full_640__mono.pkl",
                          "planB__KITTI__RTDETRl__rt_mid_480tort_full_640__mono.pkl"),
}
# every distinct (dataset, det_dir, mode) whose precision/recall curve is needed
MODES = {("nuScenes", "nusc_det_tv", "ns_cheap_320"), ("nuScenes", "nusc_det_tv", "ns_full_640"),
         ("KITTI", "det", "cheap_320"), ("KITTI", "det", "cheap_384"), ("KITTI", "det", "full_640"),
         ("KITTI", "det512", "cheap_512"), ("KITTI", "rtdetr_kitti_mid", "rt_mid_480"),
         ("KITTI", "rtdetr_kitti_mid", "rt_full_640")}
PER_MODE = ["J", "Jlat", "n", "JB"]                       # {col}_{tag}
PER_MODE_PRIM = ["fn", "fp", "loc", "cls", "crit_fn", "n_det", "n_gt"]   # {tag}_{col}
TAGS = ("cheap", "full")

_W = {}                                                   # per-worker state


def _init(dataset):
    m52 = import_module("52_core_matrix")
    m65 = import_module("65_planner_b_decision")
    _W.update(m52=m52, m65=m65, dataset=dataset)
    if dataset == "nuScenes":
        _W["adapter"] = m52.make_adapter(m52.NuScenesDB(NUSC_ROOT, NUSC_VER))
    else:
        _W["adapter"] = m52.decision.KittiAdapter
    seqdir = (CACHE / "nusc_det_tv" / "ns_cheap_320") if dataset == "nuScenes" else (CACHE / "det" / "cheap_320")
    _W["seqs"] = [p.stem for p in sorted(seqdir.glob("*.npz"))]


def curve_task(det_dir, mode):
    """Per-frame detections, matched detections, misses and GT count at every 0.01-grid threshold."""
    from rap.kitti import TYPE_TO_COARSE
    ad, cfg = _W["adapter"], RiskConfig()
    seq_col, frame_col, arr = [], [], []
    for s in _W["seqs"]:
        geom = ad.geometry(s)
        c = DetCache(CACHE / det_dir / mode / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                  if len(g) else np.zeros((0, 4)))
            gcls = np.array([TYPE_TO_COARSE.get(t, "vehicle") for t in g["type"]])
            det = c.det(i)
            row = []
            for t in GRID01:
                L = PM.frame_losses(gt, gcls, np.zeros(len(gt)), det, cfg, op_conf=t)
                row.append((L["n_det"], L["n_det"] - L["fp"], L["fn"], L["n_gt"]))
            arr.append(row); seq_col.append(s); frame_col.append(fr)
    return mode, det_dir, np.array(seq_col), np.array(frame_col), np.asarray(arr, np.float64)


def run_task(key, t_cheap, t_full, outdir):
    """The unchanged pipeline for one pair and geometry at (t_cheap, t_full); keeps per-mode columns."""
    t0 = time.time()
    m52, m65, ad, seqs = _W["m52"], _W["m65"], _W["adapter"], _W["seqs"]
    dataset, det_dir, cm, fm, geo, _, _ = SPECS[key]
    dd = CACHE / det_dir
    cfg = RiskConfig(op_conf=t_cheap, op_conf_full=t_full)
    d = m52.decision.build(dd, cm, fm, seqs, cfg, m52.P.PlannerParams(), m52.P.CostParams(),
                           m52.G.PRIMARY, range_source=geo, adapter=ad)
    d = m52.decision.add_perception_gain(d, dd, cm, fm, seqs, cfg, adapter=ad)
    prim = m52._pm.primitives(dd, cm, fm, seqs, cfg, ad, cfg.min_gt_height)
    d = d.merge(prim, on=["seq", "frame"], how="left", validate="one_to_one")
    if dataset == "KITTI":
        pb, cb = m65.B.PARAMS_B["default"], m65.B.COSTS_B["default"]
        b = m65.build_b(dd, cm, fm, seqs, cfg, ad, geo, pb, cb)
        d = d.merge(b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
    else:
        d["JB_cheap"] = d["JB_full"] = np.nan
    keep = (["seq", "frame", "dE"] + [f"{c}_{t}" for c in PER_MODE for t in TAGS]
            + [f"{t}_{c}" for t in TAGS for c in PER_MODE_PRIM])
    name = f"{key}__c{t_cheap:.6f}__f{t_full:.6f}.npz"
    # string columns as fixed-width unicode: an object array would need allow_pickle to read back
    np.savez_compressed(outdir / name, **{c: (d[c].astype(str).to_numpy().astype("U32") if c == "seq"
                                             else d[c].to_numpy()) for c in keep})
    return key, t_cheap, t_full, name, time.time() - t0


def pooled(arr, rows):
    """Pooled detections/frame, precision, recall and F1 over the selected frames, per grid point."""
    a = arr[rows].sum(0)                                   # (n_t, 4): det, tp, fn, n_gt
    det, tp, fn, ngt = a.T
    prec = np.where(det > 0, tp / np.maximum(det, 1), np.nan)
    rec = np.where(ngt > 0, (ngt - fn) / np.maximum(ngt, 1), np.nan)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return det / max(rows.sum(), 1), prec, rec, f1


def compare(a: pd.DataFrame, b: pd.DataFrame, cols, label):
    m = a.merge(b, on=["seq", "frame"], suffixes=("_new", "_rec"), validate="one_to_one")
    worst = {}
    for c in cols:
        x, y = m[f"{c}_new"].to_numpy(float), m[f"{c}_rec"].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        worst[c] = float(np.nanmax(np.abs(np.where(both_nan, 0.0, x - y)))) if len(m) else np.nan
    ok = len(m) == len(a) == len(b) and all(v <= 1e-9 for v in worst.values())
    bad = {k: v for k, v in worst.items() if not v <= 1e-9}
    print(f"  {'PASS' if ok else 'FAIL'} {label}: {len(m)} frames" + (f"  differing: {bad}" if bad else ""))
    return {"label": label, "pass": bool(ok), "n": len(m), "max_abs_diff": worst}


def load_run(outdir, name):
    z = np.load(outdir / name, allow_pickle=False)
    return pd.DataFrame({k: z[k] for k in z.files})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="calibration_outcomes")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    outdir = run / "outcomes"
    outdir.mkdir()
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    trval = {"nuScenes": set(splits["nuscenes"]["train"] + splits["nuscenes"]["val"]),
             "KITTI": set(splits["kitti"]["train"] + splits["kitti"]["val"])}
    test = {"nuScenes": set(splits["nuscenes"]["test"]), "KITTI": set(splits["kitti"]["test"])}
    ctx = get_context("spawn")
    pools = {"nuScenes": ctx.Pool(1, initializer=_init, initargs=("nuScenes",)),
             "KITTI": ctx.Pool(args.workers, initializer=_init, initargs=("KITTI",))}

    # ---- Stage A
    t0 = time.time()
    jobs = [(ds, pools[ds].apply_async(curve_task, (dd, mode))) for ds, dd, mode in sorted(MODES)]
    curves, rows = {}, []
    for ds, j in jobs:
        mode, dd, seqs, frames, arr = j.get()
        curves[mode] = (ds, seqs, frames, arr)
        np.savez_compressed(run / f"curve__{mode}.npz", seq=seqs, frame=frames, arr=arr)
        for split, units in (("trainval", trval[ds]), ("test", test[ds]), ("all", None)):
            sel = np.ones(len(seqs), bool) if units is None else np.isin(seqs, list(units))
            dpf, prec, rec, f1 = pooled(arr, sel)
            for i, t in enumerate(GRID01):
                rows.append({"dataset": ds, "mode": mode, "split": split, "threshold": t,
                             "det_per_frame": dpf[i], "precision": prec[i], "recall": rec[i], "f1": f1[i]})
        print(f"  curve {ds:8s} {mode:12s} {len(seqs)} frames  ({time.time() - t0:.0f}s)", flush=True)
    pr = pd.DataFrame(rows)
    pr.to_csv(Path(RESULTS) / "final" / "calibration_pr_curves.csv", index=False)
    pr.to_csv(run / "calibration_pr_curves.csv", index=False)

    # ---- Stage B: S1, S2, S3 from train+val only
    tv = pr[pr.split == "trainval"]
    s2 = {}
    for mode in curves:
        g = tv[(tv["mode"] == mode) & tv.threshold.isin(GRID05)]
        s2[mode] = float(g.loc[g.f1.idxmax(), "threshold"])
    thresholds = {}
    for key, (ds, dd, cm, fm, geo, _, _) in SPECS.items():
        units = sorted(trval[ds])
        s1 = match_detection_counts([DetCache(CACHE / dd / cm / f"{s}.npz") for s in units],
                                    [DetCache(CACHE / dd / fm / f"{s}.npz") for s in units], 0.25)
        pc = float(tv[(tv["mode"] == cm) & (tv.threshold == 0.25)].precision.iloc[0])
        gf = tv[tv["mode"] == fm].sort_values("threshold")
        ok = gf[gf.precision >= pc]
        s3, s3_flag = (float(ok.threshold.iloc[0]), False) if len(ok) else (0.70, True)
        thresholds[key] = {"S0": [0.25, 0.25], "S1": [0.25, float(s1)], "S2": [s2[cm], s2[fm]],
                           "S3": [0.25, s3], "S3_none_reached_target": s3_flag,
                           "S3_target_precision": pc,
                           "S2_at_grid_floor": {cm: s2[cm] <= 0.10, fm: s2[fm] <= 0.10}}
        print(f"  {key:20s} S1 full {s1:.4f}  S2 {s2[cm]:.2f}/{s2[fm]:.2f}  S3 full {s3:.2f}"
              f"{' (target not reached)' if s3_flag else ''}  (CHEAP precision@0.25 {pc:.3f})")
    (run / "thresholds_S0_S3.json").write_text(json.dumps(thresholds, indent=1))

    # ---- Stage C
    tasks = []
    for key, (ds, *_rest) in SPECS.items():
        ts = sorted(set(GRID05) | {thresholds[key]["S1"][1], thresholds[key]["S3"][1]})
        tasks += [(ds, key, t, t) for t in ts]
        if key in CHECK_SPECS:
            tasks += [(ds, key, a, b) for a, b in CHECK_PAIRS]
    print(f"\n  stage C: {len(tasks)} pipeline runs", flush=True)
    t0 = time.time()
    jobs = [pools[ds].apply_async(run_task, (key, a, b, outdir)) for ds, key, a, b in tasks]
    index = []
    for n, j in enumerate(jobs, 1):
        key, a, b, name, dt = j.get()
        index.append({"spec": key, "t_cheap": a, "t_full": b, "file": name, "seconds": dt})
        print(f"  [{n}/{len(jobs)}] {key:20s} ({a:.4f}, {b:.4f})  {dt:.0f}s  elapsed {time.time() - t0:.0f}s", flush=True)
    for p in pools.values():
        p.close(); p.join()
    idx = pd.DataFrame(index)
    idx.to_csv(run / "outcome_index.csv", index=False)

    # ---- Checks
    checks = []
    derived = [f"dE_{m}" for m in PM.METRICS]
    for key, (ds, dd, cm, fm, geo, core_name, planb_name) in SPECS.items():
        new = load_run(outdir, idx[(idx.spec == key) & (idx.t_cheap == 0.25) & (idx.t_full == 0.25)].file.iloc[0])
        for m, w in PM.METRICS.items():
            new[f"dE_{m}"] = (sum(wt * new[f"cheap_{k}"] for k, wt in w.items())
                              - sum(wt * new[f"full_{k}"] for k, wt in w.items()))
        rec = pd.read_pickle(CORE / core_name)
        rec["seq"] = rec.seq.astype(str)
        new["seq"] = new.seq.astype(str)
        cols = (["J_cheap", "J_full", "Jlat_cheap", "Jlat_full", "n_cheap", "n_full", "dE"] + derived
                + [f"{t}_{c}" for t in TAGS for c in ("fn", "fp", "n_det")])
        checks.append(compare(new, rec, cols, f"check1 {key} vs {core_name}"))
        if planb_name:
            rb = pd.read_pickle(PLANB / planb_name)
            rb["seq"] = rb.seq.astype(str)
            checks.append(compare(new, rb, ["JB_cheap", "JB_full"], f"check1 {key} vs {planb_name}"))
    for key in CHECK_SPECS:
        for a, b in CHECK_PAIRS:
            direct = load_run(outdir, idx[(idx.spec == key) & (idx.t_cheap == a) & (idx.t_full == b)].file.iloc[0])
            ca = load_run(outdir, idx[(idx.spec == key) & (idx.t_cheap == a) & (idx.t_full == a)].file.iloc[0])
            cb = load_run(outdir, idx[(idx.spec == key) & (idx.t_cheap == b) & (idx.t_full == b)].file.iloc[0])
            comp = ca[["seq", "frame"] + [c for c in ca.columns if c.endswith("_cheap") or c.startswith("cheap_")]].merge(
                cb[["seq", "frame"] + [c for c in cb.columns if c.endswith("_full") or c.startswith("full_")]],
                on=["seq", "frame"], validate="one_to_one")
            cols = [c for c in comp.columns if c not in ("seq", "frame")]
            checks.append(compare(direct, comp, cols, f"check2 {key} direct ({a}, {b}) vs composition"))
    allpass = all(c["pass"] for c in checks)
    (run / "checks.json").write_text(json.dumps({"all_pass": allpass, "checks": checks}, indent=1))
    print(f"\n  checks 1-2: {'ALL PASS' if allpass else 'FAILED -- no scheme may be scored'}")
    print("  wrote", run)
    sys.exit(0 if allpass else 3)


if __name__ == "__main__":
    main()

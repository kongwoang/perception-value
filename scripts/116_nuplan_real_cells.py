#!/usr/bin/env python
"""Task 5 B4 + C2/C3: recall checks, real-perception cells next to the transported profile, the reading.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44).

  --stage checks  before any CHEAP/FULL branch is scored: data, detection, projection and reference
                  checks merged, match IoU, per-class and per-distance recall at 320 / 640 against the
                  KITTI measurement and against the transported model's prediction on the same objects.
  --stage cells   per planner x loss x split x variant: affected states, harm rate, rho, all-FULL and
                  oracle@20 reductions, log-level bootstrap CIs (1,000 draws), and the reading.

Costs are 83's `costs` (collision, clearance shortfall, log deviation, safety, scalar_J), unchanged.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.detector_model import KITTI_TO_COARSE, NUPLAN_TO_COARSE, DetectorMissModel    # noqa: E402
from rap.paths import CACHE, RESULTS                                                  # noqa: E402

_s = importlib.util.spec_from_file_location("x83", ROOT / "scripts" / "83_external_transfer.py")
x83 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(x83)

WORK = ROOT / "results" / "raw" / "nuplan_task5"
FINAL = Path(RESULTS) / "final"
OUTC = Path(CACHE) / "nuplan_real"
PLANNERS = ("pdm_closed", "idm")
LOSSES = ("collision", "clearance_shortfall", "log_deviation", "safety", "scalar_J")
VARIANTS = {"transported": ("cheap", "full"), "primary": ("cheap", "full"), "nofp": ("cheap_nofp", "full_nofp"),
            "s1": ("cheap", "full_s1"), "iou50": ("cheap_iou50", "full_iou50")}
BANDS = [0, 10, 20, 30, 40, 60, np.inf]
NBOOT = 1000


def point_stats(cc, cf):
    V = cc - cf
    nz = np.abs(V) > 1e-9
    pos, neg = V > 1e-9, V < -1e-9
    allc, allf = float(cc.sum()), float(cf.sum())
    ben, harm_mass = float(np.clip(V, 0, None).sum()), float(np.clip(-V, 0, None).sum())
    k = max(int(round(0.2 * len(V))), 1)
    top = np.sort(V)[::-1][:k]
    o20 = float(top[top > 0].sum()) / allc if allc > 1e-12 else np.nan
    af = (allc - allf) / allc if allc > 1e-12 else np.nan
    return {"states": len(V), "affected": int(nz.sum()), "v_pos": int(pos.sum()), "v_neg": int(neg.sum()),
            "harm_rate": float(neg.sum() / nz.sum()) if nz.any() else 0.0,
            "rho": harm_mass / ben if ben > 1e-12 else (np.inf if harm_mass > 1e-12 else np.nan),
            "all_full_reduction": af, "oracle20_reduction": o20,
            "selective_extra_share": (o20 - af) / o20 if (np.isfinite(o20) and o20 > 1e-12) else np.nan,
            "all_cheap_cost": allc, "all_full_cost": allf}


def boot_ci(cc, cf, logs, rng):
    uniq = np.unique(logs)
    rows = {u: np.flatnonzero(logs == u) for u in uniq}
    draws = {k: [] for k in ("harm_rate", "rho", "all_full_reduction", "oracle20_reduction")}
    for _ in range(NBOOT):
        ix = np.concatenate([rows[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        s = point_stats(cc[ix], cf[ix])
        for k in draws:
            draws[k].append(s[k])
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        out[f"{k}_lo"], out[f"{k}_hi"] = ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                                          if len(v) >= 20 else (np.nan, np.nan))
        out[f"{k}_finite_draws"] = int(len(v))
    return out


def stage_checks():
    ck = {}
    dirs = json.loads((WORK / "dirs.json").read_text())
    ck["data"] = {
        "shards": {k: {x: v.get(x) for x in ("url", "source", "size", "head_status", "content_type", "accept_ranges",
                                             "members", "cam_f0_members", "logs", "logs_equal_file_group", "cd_size")}
                   for k, v in dirs.items()},
        "all_shards_equal_file_group": all(v.get("logs_equal_file_group") for v in dirs.values()),
        "plan": json.loads((WORK / "plan.json").read_text())}
    man = pd.read_csv(WORK / "manifest.csv")
    led = {}
    for f in glob.glob(str(WORK / "requests_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            led.setdefault(r["kind"], {"requests": 0, "bytes": 0})
            led[r["kind"]]["requests"] += 1
            led[r["kind"]]["bytes"] += r["body_bytes"]
    ck["data"].update(images_verified=int((man.crc_ok & man.decode_ok).sum()), crc_failures=int((~man.crc_ok).sum()),
                      decode_failures=int((man.crc_ok & ~man.decode_ok).sum()),
                      member_bytes_manifest=int(man.fetched_bytes.sum()), overread_bytes=int(man.overread_bytes.sum()),
                      extra_requests=int(man.extra_requests.sum()), ledger=led,
                      caps={"metadata_bytes": 400_000_000, "image_bytes": 3_500_000_000})
    ck["detection"] = json.loads((WORK / "detect_summary.json").read_text())
    ck["projection"] = json.loads((WORK / "checks_projection.json").read_text())
    for p in PLANNERS:
        f = WORK / f"checks_reference_{p}.json"
        ck[f"reference_{p}"] = json.loads(f.read_text()) if f.exists() else None

    # recall on the decision iterations' eligible in-camera tracks
    o = pd.read_csv(OUTC / "objects_decision.csv.gz")
    z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
    model = DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"])
    o["p_cheap_transported"], o["p_full_transported"] = model.p_detect(
        o.dist.to_numpy(), o.lat.to_numpy(), np.array([NUPLAN_TO_COARSE[c] for c in o.category]))
    o["band"] = pd.cut(o.dist, BANDS, right=False).astype(str)
    kitti = pd.read_pickle(ROOT / "results/raw/20260911_143838_mechanism/objects.pkl")
    kitti["category"] = [KITTI_TO_COARSE.get(str(t), "other") for t in kitti["type"]]
    kitti["band"] = pd.cut(kitti.dist_m, BANDS, right=False).astype(str)
    rows = []
    for tall in (False, True):
        oo = o[o.height_px >= 10] if tall else o
        for cat in ("all", "vehicle", "pedestrian", "bicycle"):
            oc = oo if cat == "all" else oo[oo.category == cat]
            kc = kitti if cat == "all" else kitti[kitti.category == cat]
            for band in ["all"] + sorted(o.band.unique(), key=lambda b: float(b.split(",")[0][1:])):
                ob = oc if band == "all" else oc[oc.band == band]
                kb = kc if band == "all" else kc[kc.band == band]
                if not len(ob):
                    continue
                hc, hf = ob.hit_cheap_025_iou30, ob.hit_full_025_iou30
                rows.append({"min_height_10px": tall, "category": cat, "band_m": band, "n_nuplan": len(ob),
                             "recall_320_iou30": hc.mean(), "recall_640_iou30": hf.mean(),
                             "recall_320_iou50": ob.hit_cheap_025_iou50.mean(),
                             "recall_640_iou50": ob.hit_full_025_iou50.mean(),
                             "p_640_given_320_iou30": hf[hc].mean() if hc.any() else np.nan,
                             "share_320_hit_640_miss_iou30": (hc & ~hf).mean(),
                             "transported_pred_320": ob.p_cheap_transported.mean(),
                             "transported_pred_640": ob.p_full_transported.mean(),
                             "n_kitti": len(kb), "kitti_recall_320": kb.hit_cheap_320.mean() if len(kb) else np.nan,
                             "kitti_recall_640": kb.hit_full_640.mean() if len(kb) else np.nan})
    rec = pd.DataFrame(rows)
    rec.to_csv(FINAL / "nuplan_real_perception_recall.csv", index=False)
    ck["recall_overall"] = rec[(rec.category == "all") & (rec.band_m == "all")].round(4).to_dict("records")
    ck["recall_note"] = ("nuPlan: Hungarian, class-compatible, projected lidar boxes, distance = centre range from "
                         "the rear axle. KITTI: greedy class-agnostic IoU 0.5 on 2D labels, distance = near-edge "
                         "longitudinal range, boxes >= 10 px. Different matching: a caveat, not a pass/fail.")
    (FINAL / "nuplan_real_perception_checks.json").write_text(json.dumps(ck, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(rec[(rec.band_m == "all")].round(3).to_string(index=False))
    print(rec[(rec.category == "all") & ~rec.min_height_10px].round(3).to_string(index=False))


def stage_cells():
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuplan"]
    split_of = {l: k for k in ("train", "val", "test") for l in splits[k]}
    rows = []
    for p in PLANNERS:
        tb = pd.read_csv(FINAL / f"phase0g_external_{p}_raw.csv")
        rl = pd.read_csv(FINAL / f"nuplan_real_perception_{p}_raw.csv")
        assert len(tb) == len(rl) == 1440
        for variant, (cm, fm) in VARIANTS.items():
            d = tb if variant == "transported" else rl
            ok = (d[f"ok_{cm}"] == 1) & (d[f"ok_{fm}"] == 1)
            d = d[ok].reset_index(drop=True)
            cc_all, cf_all = x83.costs(d, cm), x83.costs(d, fm)
            logs = d.log.to_numpy()
            for split in ("all", "test"):
                m = np.ones(len(d), bool) if split == "all" else (d.log.map(split_of) == "test").to_numpy()
                for loss in LOSSES:
                    cc, cf = cc_all[loss][m], cf_all[loss][m]
                    r = {"planner": p, "loss": loss, "split": split, "variant": variant,
                         "logs": int(len(np.unique(logs[m]))), "states_failed": int((~ok).sum()) if split == "all" else
                         int((~ok & (tb.log.map(split_of) == "test") if variant == "transported" else
                              ~ok & (rl.log.map(split_of) == "test")).sum()),
                         **point_stats(cc, cf)}
                    r.update(boot_ci(cc, cf, logs[m], np.random.default_rng(0)))
                    rows.append(r)
            print(f"  {p} {variant}: {int(ok.sum())} states", flush=True)
    cells = pd.DataFrame(rows)
    cells.to_csv(FINAL / "nuplan_real_perception_cells.csv", index=False)

    def reading(variant, split):
        s = cells[(cells.variant == variant) & (cells.split == split)].set_index(["loss", "planner"])
        out = {}
        for agg in ("safety", "scalar_J"):
            h = {p: float(s.loc[(agg, p), "harm_rate"]) for p in PLANNERS}
            rho = {p: float(s.loc[(agg, p), "rho"]) for p in PLANNERS}
            out[agg] = {"harm_rate": h, "rho": rho, "B-F1_fires": all(v < 0.05 for v in h.values()),
                        "harm_ge_20_and_rho_ge_20_both": all(h[p] >= 0.20 and rho[p] >= 0.20 for p in PLANNERS)}
        consistent = all(out[a]["harm_ge_20_and_rho_ge_20_both"] for a in out)
        fires = {a: out[a]["B-F1_fires"] for a in out}
        out["reading"] = ("falsifier fires" if any(fires.values()) and not consistent else
                          "consistent with the benchmark" if consistent else "intermediate")
        return out

    ck_path = FINAL / "nuplan_real_perception_checks.json"
    ck = json.loads(ck_path.read_text())
    ck["reading_primary_all"] = reading("primary", "all")
    ck["reading_other"] = {f"{v}_{s}": reading(v, s) for v in VARIANTS for s in ("all", "test")
                           if not (v == "primary" and s == "all")}
    desc = pd.read_csv(WORK / "branch_state_summary.csv.gz")
    ck["branch_descriptives_per_state"] = desc.drop(columns=["scenario", "log", "split", "image_token"]).mean().round(3).to_dict()
    ck_path.write_text(json.dumps(ck, indent=1, default=float))
    show = cells[cells.loss.isin(["safety", "scalar_J"])][
        ["planner", "loss", "split", "variant", "affected", "v_pos", "v_neg", "harm_rate", "harm_rate_lo", "harm_rate_hi",
         "rho", "rho_lo", "rho_hi", "all_full_reduction", "oracle20_reduction"]]
    pd.set_option("display.width", 250)
    print(show.round(3).to_string(index=False))
    print(json.dumps(ck["reading_primary_all"], indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["checks", "cells"])
    args = ap.parse_args()
    {"checks": stage_checks, "cells": stage_cells}[args.stage]()


if __name__ == "__main__":
    main()

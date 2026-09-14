#!/usr/bin/env python
"""Checks and before/after comparison for the IDM route-fix rerun (pre-registered in the research log, 2026-09-15).

  --stage trackb   after the corrected Track B IDM run: every branch computed a trajectory; the perception filter is
                   unchanged (CHEAP/FULL/reference track counts equal the archived run on every state); IDM's
                   reference plan is sane -- median mean log deviation <= 3 m and at most 2% of states above 20 m.
                   Exit code 3 on failure, which stops the chain before anything is scored.
  --stage compare  after the chain: every PDM-Closed, nuScenes and KITTI row of the regenerated tables equals the
                   archived one (row order ignored); a before/after table of the IDM rows.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
ARCH = ROOT / "results" / "archive" / "pre_idm_route_fix" / "final"


def stage_trackb():
    new = pd.read_csv(FINAL / "phase0g_external_idm_raw.csv")
    old = pd.read_csv(ARCH / "phase0g_external_idm_raw.csv")
    pdm = pd.read_csv(FINAL / "phase0g_external_pdm_closed_raw.csv")
    m = new.merge(old, on=["scenario", "log", "iteration"], suffixes=("", "_old"), validate="one_to_one")
    counts_equal = all((m[f"n_tracks_{b}"] == m[f"n_tracks_{b}_old"]).all() for b in ("reference", "cheap", "full"))
    dev = new.log_deviation_mean_reference
    c = {"states": len(new), "all_branches_ok": bool((new[["ok_reference", "ok_cheap", "ok_full"]] == 1).all().all()),
         "track_counts_equal_archive": bool(counts_equal and len(m) == 1440),
         "idm_ref_logdev_median_m": float(dev.median()), "idm_ref_share_logdev_over_20m": float((dev > 20).mean()),
         "idm_ref_collision_rate": float(new.collision_reference.mean()),
         "archived_idm_ref_logdev_median_m": float(old.log_deviation_mean_reference.median()),
         "archived_idm_ref_share_logdev_over_20m": float((old.log_deviation_mean_reference > 20).mean()),
         "archived_idm_ref_collision_rate": float(old.collision_reference.mean()),
         "pdm_ref_logdev_median_m": float(pdm.log_deviation_mean_reference.median()),
         "pdm_ref_share_logdev_over_20m": float((pdm.log_deviation_mean_reference > 20).mean()),
         "pdm_ref_collision_rate": float(pdm.collision_reference.mean())}
    c["pass"] = bool(c["all_branches_ok"] and c["track_counts_equal_archive"] and c["idm_ref_logdev_median_m"] <= 3.0
                     and c["idm_ref_share_logdev_over_20m"] <= 0.02)
    (ROOT / "results" / "raw" / "idm_route_diagnostics" / "check_trackb_idm.json").write_text(json.dumps(c, indent=1))
    print(json.dumps(c, indent=1))
    if not c["pass"]:
        print("CHECK FAILED: IDM reference plans are not sane after the route fix; stopping before scoring")
        sys.exit(3)


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, f"shape {a.shape} vs {b.shape}"
    keys = [c for c in a.columns if not pd.api.types.is_numeric_dtype(a[c])] + [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    a = a.sort_values(keys, na_position="first", kind="mergesort").reset_index(drop=True)
    b = b.sort_values(keys, na_position="first", kind="mergesort").reset_index(drop=True)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]):
            if not np.isclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True).all():
                return False, f"column {c}"
        elif not a[c].astype(str).equals(b[c].astype(str)):
            return False, f"column {c} (text)"
    return True, ""


def stage_compare():
    report = {"unchanged_rows": {}, "idm_before_after": {}}
    # 1. rows that must not move
    for f, sel in [("benchmark_table.csv", lambda d: d.system != "idm"), ("benchmark_cells.csv", lambda d: d.system != "idm"),
                   ("benchmark_table_routers.csv", lambda d: d.system != "idm"),
                   ("benchmark_budget_routers.csv", lambda d: d.system != "idm"),
                   ("nuplan_real_perception_cells.csv", lambda d: d.planner != "idm"),
                   ("benchmark_table_nuplan_real.csv", lambda d: d.system != "idm"),
                   ("benchmark_budget_nuplan_real.csv", lambda d: d.system != "idm"),
                   ("phase0g_external_pdm_closed_raw.csv", lambda d: d.index >= 0),
                   ("nuplan_real_perception_pdm_closed_raw.csv", lambda d: d.index >= 0)]:
        a, b = pd.read_csv(FINAL / f), pd.read_csv(ARCH / f)
        ok, why = same_rows(a[sel(a)].reset_index(drop=True), b[sel(b)].reset_index(drop=True))
        report["unchanged_rows"][f] = "identical" if ok else f"DIFFERS: {why}"
    # 2. IDM before / after
    def cells(root):
        c = pd.read_csv(root / "nuplan_real_perception_cells.csv")
        c = c[(c.planner == "idm") & c.loss.isin(["safety", "scalar_J"])]
        return c.set_index(["variant", "split", "loss"])[["affected", "v_pos", "v_neg", "harm_rate", "rho", "all_full_reduction", "oracle20_reduction"]]
    report["idm_before_after"]["nuplan_real_perception_cells"] = {
        "before": cells(ARCH).round(4).reset_index().to_dict("records"), "after": cells(FINAL).round(4).reset_index().to_dict("records")}
    for f in ("benchmark_table.csv", "benchmark_table_nuplan_real.csv", "benchmark_table_routers.csv"):
        rows = {}
        for tag, root in (("before", ARCH), ("after", FINAL)):
            d = pd.read_csv(root / f)
            d = d[(d.system == "idm") & (d.split == "test") & (d.get("quota", pd.Series(0.2, index=d.index)) == 0.2)]
            cols = [c for c in ("target", "signal", "eta", "eta_lo", "eta_hi", "minus_random_lo", "n_affected", "undefined_reason") if c in d.columns]
            rows[tag] = d[cols].round(4).to_dict("records")
        report["idm_before_after"][f] = rows
    for tag, root in (("before", ARCH), ("after", FINAL)):
        ck = json.loads((root / "nuplan_real_perception_checks.json").read_text())
        report["idm_before_after"].setdefault("reading_primary_all", {})[tag] = ck.get("reading_primary_all")
        s = json.loads((root / "phase0g_external_planner_stats.json").read_text())
        report["idm_before_after"].setdefault("trackb_falsifiers", {})[tag] = s
    report["all_unchanged_rows_identical"] = all(v == "identical" for v in report["unchanged_rows"].values())
    (FINAL / "idm_route_fix_comparison.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report["unchanged_rows"], indent=1))
    print("reading before:", json.dumps(report["idm_before_after"]["reading_primary_all"]["before"])[:600])
    print("reading after: ", json.dumps(report["idm_before_after"]["reading_primary_all"]["after"])[:600])
    if not report["all_unchanged_rows_identical"]:
        print("WARNING: rows that should not change moved; see idm_route_fix_comparison.json")
        sys.exit(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["trackb", "compare"])
    args = ap.parse_args()
    {"trackb": stage_trackb, "compare": stage_compare}[args.stage]()


if __name__ == "__main__":
    main()

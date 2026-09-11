#!/usr/bin/env python
"""Pick the CHEAP/FULL pair on evidence rather than convention.

A pair is only useful for Phase 0 if it buys a real compute gap on this board AND a
real accuracy gap on this data — and if CHEAP is not so degraded that every frame
benefits, which would destroy the heterogeneity the study is about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import geometry as G, runmeta, tables      # noqa: E402
from rap.cache import DetCache                      # noqa: E402
from rap.paths import CACHE                         # noqa: E402
from rap.risk import RiskConfig                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "modesel"))
    ap.add_argument("--modes", nargs="+",
                    default=["cheap_320", "cheap_384", "cheap_512", "full_640", "full_960"])
    ap.add_argument("--profile", default="")
    args = ap.parse_args()

    det = Path(args.det)
    modes = [m for m in args.modes if (det / m).exists()]
    seqs = [p.stem for p in sorted((det / modes[0]).glob("*.npz"))]
    cfg = RiskConfig()
    geom = {s: G.sequence_geometry(s) for s in seqs}
    run = runmeta.new_run("modesel", vars(args))

    prof = {}
    if args.profile:
        prof = json.loads((Path(args.profile) / "profile_summary.json").read_text())

    # every mode scored against itself as both arms gives its own absolute risk
    rows = []
    for m in modes:
        df = pd.concat([tables.build_sequence(
            s, DetCache(det / m / f"{s}.npz"), DetCache(det / m / f"{s}.npz"),
            G.PRIMARY, cfg, geom[s]) for s in seqs], ignore_index=True)
        r = {"mode": m, "risk": df.risk_cheap.sum(), "miss_rate": df.n_miss_cheap.sum() / df.n_gt.sum(),
             "n_fp": df.n_fp_cheap.sum(), "n_det": df.n_det_cheap.sum(), "n_gt": df.n_gt.sum()}
        if m in prof:
            r["lat_ms"] = prof[m]["lat_e2e_ms_median"]
            r["lat_p95_ms"] = prof[m]["lat_e2e_ms_p95"]
            r["gpu_mw"] = prof[m].get("GPU_mw_over_idle", np.nan)
        rows.append(r)
    solo = pd.DataFrame(rows)
    print("\nper-mode absolute quality and cost:")
    print(solo.round(3).to_string(index=False))

    pairs = []
    for i, c in enumerate(modes):
        for f in modes[i + 1:]:
            df = pd.concat([tables.build_sequence(
                s, DetCache(det / c / f"{s}.npz"), DetCache(det / f / f"{s}.npz"),
                G.PRIMARY, cfg, geom[s]) for s in seqs], ignore_index=True)
            v = df.value_task.to_numpy()
            order = np.argsort(-v)
            cum = np.cumsum(v[order])
            tot = cum[-1]
            row = {
                "cheap": c, "full": f, "n_frames": len(df),
                "risk_cheap": df.risk_cheap.sum(), "risk_full": df.risk_full.sum(),
                "risk_reduction_rel": tot / df.risk_cheap.sum() if df.risk_cheap.sum() else np.nan,
                "frac_value_gt0": float(np.mean(v > 1e-9)),
                "frac_value_zero": float(np.mean(np.abs(v) <= 1e-9)),
                "gain_top20": float(cum[int(0.2 * len(v)) - 1] / tot) if tot > 1e-12 else np.nan,
            }
            if c in prof and f in prof:
                row["lat_gap_ms"] = prof[f]["lat_e2e_ms_median"] - prof[c]["lat_e2e_ms_median"]
                row["lat_ratio"] = prof[f]["lat_e2e_ms_median"] / prof[c]["lat_e2e_ms_median"]
            pairs.append(row)
    pdf = pd.DataFrame(pairs)
    print("\ncandidate CHEAP/FULL pairs:")
    print(pdf.round(3).to_string(index=False))
    solo.to_csv(run / "modes.csv", index=False)
    pdf.to_csv(run / "pairs.csv", index=False)
    print("\nwrote", run)


if __name__ == "__main__":
    main()

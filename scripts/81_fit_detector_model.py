#!/usr/bin/env python
"""Fit the measured detector miss model and write it where the nuPlan intervention reads it.

Input is the per-object table from Phase 0's mechanism analysis: one row per ground-truth object
with its range, lateral offset and class, and whether YOLOv8s detected it at 320 and at 640.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.detector_model import fit_from_objects                                # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", default=str(ROOT / "results/raw/20260911_143838_mechanism/objects.pkl"))
    ap.add_argument("--out", default=str(Path(CACHE) / "detector_miss_model.npz"))
    args = ap.parse_args()

    d = pd.read_pickle(args.objects)
    model, diag = fit_from_objects(d)
    np.savez(args.out, w_cheap=model.w_cheap, w_rescue=model.w_rescue,
             w_lose=model.w_lose, diag=json.dumps(diag))
    keys = ("n_objects", "measured_classes", "recall_cheap_observed", "recall_cheap_predicted",
            "recall_full_observed", "recall_full_predicted", "brier_cheap_val",
            "brier_cheap_baserate", "brier_full_implied_val", "brier_full_baserate",
            "p_rescue_observed", "p_lose_observed", "monotone_violation_observed")
    for k in keys:
        v = diag[k]
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    (Path(args.out).with_suffix(".json")).write_text(json.dumps(diag, indent=2))
    print("  wrote", args.out)


if __name__ == "__main__":
    main()

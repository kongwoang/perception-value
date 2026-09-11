#!/usr/bin/env python
"""Assemble the per-frame table (oracle targets + deployable features) per criticality model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import geometry as G                       # noqa: E402
from rap import kitti, runmeta, tables              # noqa: E402
from rap.cache import DetCache                      # noqa: E402
from rap.paths import CACHE, PROCESSED              # noqa: E402
from rap.risk import RiskConfig                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "det"))
    ap.add_argument("--cheap", default="cheap_320")
    ap.add_argument("--full", default="full_640")
    ap.add_argument("--crit_models", nargs="+", default=["composite"])
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--op_conf", type=float, default=0.25)
    ap.add_argument("--error", default="miss", choices=["miss", "soft_iou"])
    ap.add_argument("--fp_lambda", type=float, default=0.0)
    ap.add_argument("--class_aware", type=int, default=0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--out", default=str(PROCESSED))
    args = ap.parse_args()

    run = runmeta.new_run("tables", vars(args))
    cfg = RiskConfig(iou_thr=args.iou_thr, op_conf=args.op_conf, error=args.error,
                     fp_lambda=args.fp_lambda, class_aware=bool(args.class_aware))
    det = Path(args.det)
    seqs = [p.stem for p in sorted((det / args.cheap).glob("*.npz"))]
    assert seqs, f"no detection cache under {det/args.cheap}"

    geom_cache = {s: G.sequence_geometry(s) for s in seqs}
    for mname in args.crit_models:
        model = G.CRITICALITY_MODELS[mname]
        frames = []
        for s in seqs:
            cheap = DetCache(det / args.cheap / f"{s}.npz")
            full = DetCache(det / args.full / f"{s}.npz")
            frames.append(tables.build_sequence(s, cheap, full, model, cfg, geom_cache[s]))
        df = pd.concat(frames, ignore_index=True)
        name = f"frames_{mname}{args.suffix}.pkl"
        df.to_pickle(Path(args.out) / name)
        print(f"[{mname}] {len(df)} frames  value_task>0 in {(df.value_task>1e-9).mean():.1%}  "
              f"mean risk_cheap={df.risk_cheap.mean():.3f} risk_full={df.risk_full.mean():.3f}  -> {name}")
    print("wrote", run)


if __name__ == "__main__":
    main()

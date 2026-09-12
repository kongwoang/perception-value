#!/usr/bin/env python
"""Phase 0F: write nuScenes detection submissions for our CHEAP/FULL modes.

Kept separate from the PKL/TIP runners because this needs our own nuScenes reader while
they need the devkit's, and loading both sets of JSON tables in one process exhausts this
board's 15 GB of unified memory.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import nusc_submission as NS, runmeta          # noqa: E402
from rap.cache import DetCache                          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter           # noqa: E402
from rap.paths import CACHE                             # noqa: E402
from rap.risk import RiskConfig                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--det", default=str(CACHE / "nusc_det_tv"))
    ap.add_argument("--modes", nargs="+", default=["ns_cheap_320", "ns_full_640"])
    ap.add_argument("--variants", nargs="+", default=["oracle", "mono"])
    ap.add_argument("--scenes", type=int, default=0)
    ap.add_argument("--out", default=str(CACHE / "nusc_submissions"))
    args = ap.parse_args()

    run = runmeta.new_run("submissions", vars(args))
    cfg = RiskConfig()
    db = NuScenesDB(args.dataroot, args.version)
    adapter = make_adapter(db)
    det = Path(args.det)
    seqs = [p.stem for p in sorted((det / args.modes[0]).glob("*.npz"))]
    if args.scenes:
        seqs = seqs[: args.scenes]
    for s in seqs:
        adapter.geometry(s)          # populates the track-id map annotations_for needs

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"scenes": seqs, "files": {}}
    for variant in args.variants:
        for mode in args.modes:
            caches = {s: DetCache(det / mode / f"{s}.npz") for s in seqs}
            sub = NS.build_submission(db, adapter, caches, seqs, cfg, variant)
            p = NS.write_submission(sub, out / f"{variant}__{mode}.json")
            n = sum(len(v) for v in sub["results"].values())
            manifest["files"][f"{variant}/{mode}"] = str(p)
            print(f"  {variant:7s} {mode:14s} {len(sub['results'])} samples, {n} boxes")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

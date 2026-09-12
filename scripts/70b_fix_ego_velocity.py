#!/usr/bin/env python
"""Recompute the cached ego velocity in place, after the backward-difference fix.

`ego_velocity` originally used a central difference, which reads the pose half a second into
the future.  That leaked the target twice over: it inflated the constant-velocity baseline, and
once ego velocity became an input to Planner D it fed the network part of the answer.

Only the `ego_v` array depends on it, so the rasters -- three hours of rendering -- are reused
and just this field is rewritten.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nuscenes.nuscenes import NuScenes                                         # noqa: E402
from pyquaternion import Quaternion                                            # noqa: E402
from rap.ego_traj import ego_velocity                                          # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(CACHE / "planner_d"))
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    args = ap.parse_args()

    files = sorted(Path(args.data).glob("*/chunk_*.npz"))
    if not files:
        raise SystemExit("no cached chunks")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # every scene's pose track, built once
    poses = {}
    for sc in nusc.scene:
        ts, xy, yaw, toks = [], [], [], []
        tok = sc["first_sample_token"]
        while tok:
            samp = nusc.get("sample", tok)
            sd = nusc.get("sample_data", samp["data"]["LIDAR_TOP"])
            pose = nusc.get("ego_pose", sd["ego_pose_token"])
            r = Quaternion(pose["rotation"]).rotation_matrix
            ts.append(samp["timestamp"] / 1e6); xy.append(pose["translation"][:2])
            yaw.append(np.arctan2(r[1, 0], r[0, 0])); toks.append(tok)
            tok = samp["next"]
        poses[sc["name"]] = (np.array(ts), np.array(xy), np.array(yaw),
                             {t: i for i, t in enumerate(toks)})

    for f in files:
        z = dict(np.load(f, allow_pickle=False))
        if len(z["sample_token"]) == 0:
            continue
        old = z["ego_v"].copy()
        new = np.zeros_like(old)
        for k, (tok, scene) in enumerate(zip(z["sample_token"], z["scene"])):
            ts, xy, yaw, idx = poses[str(scene)]
            new[k] = ego_velocity(ts, xy, yaw, idx[str(tok)])
        z["ego_v"] = new.astype(np.float32)
        np.savez_compressed(f, **z)
        d = np.abs(new - old)
        print(f"  {f.parent.name}/{f.name}: {len(new)} samples, "
              f"mean |change| {d.mean():.4f} m/s, max {d.max():.3f} m/s", flush=True)
    print("done")


if __name__ == "__main__":
    main()

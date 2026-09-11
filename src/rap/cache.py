"""Ragged per-frame detection cache: flat arrays plus offsets, one file per (seq, mode)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

COARSE_ID = {"vehicle": 0, "person": 1, "cyclist": 2}
ID_COARSE = np.array(["vehicle", "person", "cyclist"])

DET_ARRAYS = ("xyxy", "conf", "entropy", "margin", "binent")
GEO_ARRAYS = ("z", "z_ground", "z_height", "lat_min", "lat_max", "ttc", "box_h")


def save(path: Path, frames: list[int], dets: list[dict], geos: list[dict] | None,
         frame_scalars: dict[str, list]) -> None:
    counts = np.array([len(d["conf"]) for d in dets], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out = {"frames": np.asarray(frames, np.int32), "offsets": offsets}
    for k in DET_ARRAYS:
        stacked = [np.asarray(d[k]) for d in dets]
        out[k] = (np.concatenate(stacked, axis=0) if any(len(s) for s in stacked)
                  else np.zeros((0, 4) if k == "xyxy" else 0, np.float32)).astype(np.float32)
    out["coarse_id"] = np.concatenate(
        [np.array([COARSE_ID[c] for c in d["coarse"]], np.int8) for d in dets]
    ) if len(dets) else np.zeros(0, np.int8)
    if geos is not None:
        for k in GEO_ARRAYS:
            out[f"geo_{k}"] = np.concatenate([np.asarray(g[k], np.float32) for g in geos]) \
                if len(geos) else np.zeros(0, np.float32)
    for k, v in frame_scalars.items():
        out[f"fs_{k}"] = np.asarray(v, np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


class DetCache:
    """Per-frame views over one cached sequence.

    The arrays are materialised on construction rather than read through the lazy
    `NpzFile`: indexing a lazy npz re-inflates the whole array on every access, which
    made per-frame reads dominate table building by a factor of ~15.
    """

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as z:
            self.z = {k: z[k] for k in z.files}
        self.frames = self.z["frames"]
        self.offsets = self.z["offsets"]
        self.index = {int(f): i for i, f in enumerate(self.frames)}
        self.has_geo = "geo_z" in self.z

    def __len__(self) -> int:
        return len(self.frames)

    def _slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def det(self, i: int) -> dict:
        s = self._slice(i)
        d = {k: self.z[k][s] for k in DET_ARRAYS}
        d["coarse"] = ID_COARSE[self.z["coarse_id"][s]]
        for k in ("n_cand", "n_cand_raw", "n_post"):
            d[k] = float(self.z[f"fs_{k}"][i])
        return d

    def geo(self, i: int) -> dict:
        s = self._slice(i)
        return {k: self.z[f"geo_{k}"][s] for k in GEO_ARRAYS}

    def scalars(self, i: int) -> dict:
        return {k[3:]: float(v[i]) for k, v in self.z.items() if k.startswith("fs_")}

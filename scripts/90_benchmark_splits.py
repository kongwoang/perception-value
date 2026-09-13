#!/usr/bin/env python
"""Benchmark splits, frozen before any baseline is re-run on them.

One rule for every track, no RNG: sort the track's split units by (stratum, name) and assign the
unit at position i by i mod 10 -- 0-4 train, 5-6 val, 7-9 test.  Systematic assignment over a
stratum-sorted list keeps every stratum represented roughly in proportion without drawing anything,
and the list can be rebuilt from this docstring alone.

Units
  nuScenes  scene; stratum = map location (scene.json -> log.json).  The pool is the 85 scenes with
            cheap and full detections on disk.  NOT the Planner D split in
            configs/phase0g_scene_split.json, which held these same 85 scenes out as its test set.
  KITTI     tracking sequence 0000-0020; no stratum.
  nuPlan    LOG; stratum = map.  Splitting by scenario token was the request, but 3 of the 34
            same-log scenario pairs overlap in time by 5.2-9.1 s (configs/benchmark_nuplan_scenarios.csv),
            so a token split would put near-identical states on both sides.  Every scenario of a log
            follows the log, and the resulting scenario-token list per split is written out.

Frame counts are read from the decision tables only to be recorded; nothing about outcomes enters
the assignment.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import CACHE                                                     # noqa: E402

NUSC_META = Path("/home/kongwoang/datasets/nuscenes/trainval/v1.0-trainval")
CORE = ROOT / "results/raw/20260913_133004_core_matrix_postreview"
JOINED = ROOT / "results/raw/20260913_211441_phase0g_eta_fde_oracle/joined_frames.pkl"
SLOT = {0: "train", 1: "train", 2: "train", 3: "train", 4: "train", 5: "val", 6: "val",
        7: "test", 8: "test", 9: "test"}


def assign(units: list[tuple[str, str]]) -> dict[str, list[str]]:
    """units: (stratum, name).  Returns split -> names, in sorted order."""
    out = {"train": [], "val": [], "test": []}
    for i, (_, name) in enumerate(sorted(units)):
        out[SLOT[i % 10]].append(name)
    return out


def main():
    # nuScenes
    scenes = {s["name"]: s for s in json.load(open(NUSC_META / "scene.json"))}
    logs = {l["token"]: l for l in json.load(open(NUSC_META / "log.json"))}
    pool = sorted(p.stem for p in (Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))
    loc = {n: logs[scenes[n]["log_token"]]["location"] for n in pool}
    ns = assign([(loc[n], n) for n in pool])
    tm = pd.read_csv(Path(CACHE) / "nusc_token_map.csv")
    j = pd.read_pickle(JOINED)
    plan = j[j.JC_ade_cheap.notna()]
    nusc = {"unit": "scene", "stratum": "location", **ns,
            "units": {k: len(v) for k, v in ns.items()},
            "frames": {k: int(tm.seq.isin(v).sum()) for k, v in ns.items()},
            "plan_frames": {k: int(plan.seq.isin(v).sum()) for k, v in ns.items()},
            "locations": {k: pd.Series([loc[n] for n in v]).value_counts().to_dict()
                          for k, v in ns.items()}}

    # KITTI
    kt = pd.read_pickle(CORE / "KITTI__YOLOv8s__cheap_320tofull_640__oracle.pkl")
    seqs = sorted(kt.seq.astype(str).unique())
    ks = assign([("", s) for s in seqs])
    kitti = {"unit": "sequence", "stratum": None, **ks,
             "units": {k: len(v) for k, v in ks.items()},
             "frames": {k: int(kt.seq.astype(str).isin(v).sum()) for k, v in ks.items()}}

    # nuPlan
    meta = pd.read_csv(ROOT / "configs" / "benchmark_nuplan_scenarios.csv")
    raw = pd.read_csv(ROOT / "results/final/phase0g_external_idm_raw.csv")
    assert set(meta.scenario) == set(raw.scenario), "scenario metadata does not match Track B"
    lmap = meta.groupby("log")["map"].first().to_dict()
    npl = assign([(lmap[lg], lg) for lg in sorted(lmap)])
    scen = {k: sorted(meta[meta.log.isin(v)].scenario) for k, v in npl.items()}
    nuplan = {"unit": "log", "stratum": "map", **npl, "scenarios": scen,
              "units": {k: len(v) for k, v in npl.items()},
              "n_scenarios": {k: len(v) for k, v in scen.items()},
              "states": {k: int(raw.scenario.isin(v).sum()) for k, v in scen.items()},
              "maps": {k: pd.Series([lmap[x] for x in v]).value_counts().to_dict()
                       for k, v in npl.items()},
              "overlapping_same_log_pairs": 3, "same_log_pairs": 34}

    out = {"frozen": "2026-09-13, before any baseline was re-run on these splits",
           "rule": "sort units by (stratum, name); position i -> i mod 10: 0-4 train, 5-6 val, "
                   "7-9 test. No RNG.",
           "nuscenes": nusc, "kitti": kitti, "nuplan": nuplan}
    path = ROOT / "configs" / "benchmark_splits.json"
    path.write_text(json.dumps(out, indent=1))
    for name, t in (("nuScenes", nusc), ("KITTI", kitti), ("nuPlan", nuplan)):
        extra = t.get("frames") or t.get("states")
        print(f"  {name:8s} units {t['units']}  frames/states {extra}")
    print(f"  nuScenes locations {nusc['locations']}")
    print(f"  nuScenes plan frames {nusc['plan_frames']}")
    print(f"  nuPlan scenarios {nuplan['n_scenarios']}  maps {nuplan['maps']}")
    print("  wrote", path)


if __name__ == "__main__":
    main()

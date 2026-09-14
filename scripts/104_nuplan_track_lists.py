#!/usr/bin/env python
"""Task 2, R1 on nuPlan: the CHEAP-kept track list at each benchmark state, as a fixed vector.

Pre-registered in RESEARCH_LOG.md (2026-09-14, "Task 2 pre-registration").  nuPlan tracks carry no
detection confidence, so the list is ordered by distance: the 25 nearest tracks the CHEAP branch
kept at the decision iteration, each [presence, x/80, y/40, length/10, width/5,
one-hot(vehicle, pedestrian, bicycle, static), area/50] in the ego frame -> 250 dims.  The kept set
is regenerated with the same deterministic intervention and the same helpers as
`91_nuplan_cheap_features.py`, and its size is checked against Track B's `n_tracks_cheap` on every
state.  Run with scripts/pynuplan.
"""
from __future__ import annotations

import importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_s = importlib.util.spec_from_file_location("m91", ROOT / "scripts" / "91_nuplan_cheap_features.py")
m91 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m91)

from rap import runmeta                                                         # noqa: E402
from rap.detector_model import DetectorMissModel                                # noqa: E402
from rap.nuplan_perception import PerceptionFilter                              # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

TOPK = 25
CLASSES = ("vehicle", "pedestrian", "bicycle", "static")


def track_vector(objs, ego, keep, filt):
    x = np.zeros((TOPK, 5 + len(CLASSES)))
    if not len(objs) or not keep.any():
        return x.reshape(-1)
    g = m91.geometry(filt, objs, ego)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(g["dist"][idx], kind="stable")][:TOPK]
    length = np.array([objs[i].box.length for i in idx], float)
    width = np.array([objs[i].box.width for i in idx], float)
    n = len(idx)
    x[:n, 0] = 1.0
    x[:n, 1] = g["fwd"][idx] / 80.0
    x[:n, 2] = g["lat"][idx] / 40.0
    x[:n, 3] = length / 10.0
    x[:n, 4] = width / 5.0
    for j, i in enumerate(idx):
        cls = str(g["coarse"][i])
        x[j, 5 + (CLASSES.index(cls) if cls in CLASSES else 0)] = 1.0
    x[:n, 9] = length * width / 50.0
    return x.reshape(-1)


def main():
    run = runmeta.new_run("nuplan_track_lists", {})
    z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
    filt = PerceptionFilter(DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"]))
    raw = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_idm_raw.csv")
    states = raw[["scenario", "iteration", "n_tracks_cheap"]].drop_duplicates(["scenario", "iteration"])
    builder = m91.NuPlanScenarioBuilder(
        data_root="/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini",
        map_root="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0",
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = m91.ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                             num_scenarios_per_type=None, limit_total_scenarios=60, timestamp_threshold_s=None,
                             ego_displacement_minimum_m=None, expand_scenarios=False,
                             remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, m91.Sequential())}
    X, keys, bad = [], [], 0
    for _, r in states.sort_values(["scenario", "iteration"]).iterrows():
        sc, it = scen[r.scenario], int(r.iteration)
        objs = list(sc.get_tracked_objects_at_iteration(it).tracked_objects)
        ego = sc.get_ego_state_at_iteration(it)
        keep = filt.keep_mask(objs, ego, sc.token, it, "cheap") if objs else np.zeros(0, bool)
        bad += int(int(keep.sum()) != int(r.n_tracks_cheap))
        X.append(track_vector(objs, ego, keep, filt))
        keys.append((r.scenario, it))
    assert bad == 0, f"regenerated CHEAP track count differs from Track B on {bad} states"
    k = pd.DataFrame(keys, columns=["scenario", "iteration"])
    np.savez_compressed(run / "nuplan_track_features.npz", scenario=k.scenario.to_numpy().astype("U32"),
                        iteration=k.iteration.to_numpy(), X=np.asarray(X))
    print(f"  {len(k)} states, {np.asarray(X).shape[1]} dims, CHEAP counts match Track B on all; wrote {run}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Task 7: pre-escalation inputs for the nuPlan allocation track, rebuilt from the REAL CHEAP branch.

Pre-registered in RESEARCH_LOG.md (Task 7 pre-registration).  The CHEAP-branch observation is exactly what
the planner received in Task 5: 115's `RealPerceptionFilter` applied to the logged tracks, i.e. the tracks
the real 320 detections matched plus their false-positive agents.  On that observation:

  * 91's 18 gate features, with 91's definitions (`d_n_cheap_fov` against the CHEAP branch at it - 1);
  * 104's R1 track vector: the 25 nearest branch objects, 250 dims;
  * diagnostics, never offered to a gate: 91's reference criticality `crit_sum_gt` on the logged tracks,
    dE_E1_fn_only = |in-camera tracks removed by CHEAP| - |removed by FULL|, and dE_E6_risk_weighted =
    sum criticality x (1[removed by CHEAP] - 1[removed by FULL]).

Checks before anything is written: the branch's object count equals Task 5's `n_tracks_cheap` on every state,
and the recomputed `crit_sum_gt` equals the benchmark's existing column.  Run with scripts/pynuplan.
"""
from __future__ import annotations

import importlib.util, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m91 = _load("m91", "91_nuplan_cheap_features.py")
m104 = _load("m104", "104_nuplan_track_lists.py")
m115 = _load("m115", "115_nuplan_real_counterfactual.py")

from rap import runmeta                                                         # noqa: E402
from rap.detector_model import DetectorMissModel                                # noqa: E402
from rap.nuplan_perception import PerceptionFilter                              # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

DIAGNOSTIC = ["crit_sum_gt", "dE_E1_fn_only", "dE_E6_risk_weighted"]


def branch(filt_real, sc, it, mode):
    ego = sc.get_ego_state_at_iteration(it)
    obs = filt_real.apply(sc.get_tracked_objects_at_iteration(it), ego, sc.token, it, mode)
    return list(obs.tracked_objects.tracked_objects), ego


def n_fov_measured(filt, objs, ego):
    return int(m91.geometry(filt, objs, ego)["measured"].sum()) if objs else 0


def gate_row(filt, objs, ego, sc, it, prev):
    """91's `state_row` gate block, with every branch object counted as kept."""
    g = m91.geometry(filt, objs, ego)
    kc = np.ones(len(objs), bool)
    corridor = (np.abs(g["lat"]) < m91.CORRIDOR_HALF_W) & (g["fwd"] > 0)
    seen, seen_m, mc = kc & g["fov"], kc & g["measured"], kc & corridor
    r = {"ego_speed": ego.dynamic_car_state.rear_axle_velocity_2d.x,
         "ego_accel": ego.dynamic_car_state.rear_axle_acceleration_2d.x,
         "n_cheap": int(kc.sum()), "n_cheap_fov": int(seen_m.sum())}
    for c in m91.CLASSES:
        r[f"n_cheap_fov_{c}"] = int((seen_m & (g["coarse"] == c)).sum())
    r["n_static_fov"] = int((seen & (g["coarse"] == "static")).sum())
    r["min_range_fov"] = float(g["dist"][seen].min()) if seen.any() else m91.FAR
    r["min_gap_corridor"] = float(g["long_near"][mc].min()) if mc.any() else m91.FAR
    r["n_corridor_20"] = int((mc & (g["long_near"] < 20)).sum())
    r["n_corridor_40"] = int((mc & (g["long_near"] < 40)).sum())
    r["min_ttc_corridor"] = float(min(g["ttc"][mc].min(), m91.TTC_CAP)) if mc.any() else m91.TTC_CAP
    r["n_fov_band_30_40"] = int((seen_m & (g["dist"] >= 30) & (g["dist"] < 40)).sum())
    r["crit_cheap_sum"] = float(g["crit"][seen].sum())
    r["crit_cheap_max"] = float(g["crit"][seen].max()) if seen.any() else 0.0
    r["n_red_lights"] = int(sum(1 for t in sc.get_traffic_light_status_at_iteration(it) if t.status.name == "RED"))
    r["d_n_cheap_fov"] = r["n_cheap_fov"] - (n_fov_measured(filt, *prev) if prev is not None else r["n_cheap_fov"])
    return r


def main():
    run = runmeta.new_run("nuplan_real_features", {})
    z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
    filt = PerceptionFilter(DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"]))   # geometry helpers only
    with open(Path(CACHE) / "nuplan_real" / "branches.pkl", "rb") as f:
        spec = pickle.load(f)["spec"]
    filt_real = m115.RealPerceptionFilter(spec)
    raw = pd.read_csv(Path(RESULTS) / "final" / "nuplan_real_perception_idm_raw.csv")
    raw_pdm = pd.read_csv(Path(RESULTS) / "final" / "nuplan_real_perception_pdm_closed_raw.csv")
    assert (raw.set_index(["scenario", "iteration"]).n_tracks_cheap
            .equals(raw_pdm.set_index(["scenario", "iteration"]).n_tracks_cheap))
    old = pd.read_csv(Path(RESULTS) / "final" / "benchmark_nuplan_signals.csv").set_index(["scenario", "iteration"])

    builder = m91.NuPlanScenarioBuilder(
        data_root="/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini",
        map_root="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0",
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = m91.ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                             num_scenarios_per_type=None, limit_total_scenarios=60, timestamp_threshold_s=None,
                             ego_displacement_minimum_m=None, expand_scenarios=False,
                             remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, m91.Sequential())}
    assert set(scen) == set(raw.scenario)

    rows, X, keys, bad_n, bad_crit = [], [], [], 0, 0
    for n_st, r0 in enumerate(raw.sort_values(["scenario", "iteration"]).itertuples(), 1):
        sc, it = scen[r0.scenario], int(r0.iteration)
        objs, ego = branch(filt_real, sc, it, "cheap")
        prev = branch(filt_real, sc, it - 1, "cheap") if it > 0 else None
        row = {"scenario": r0.scenario, "log": r0.log, "iteration": it, **gate_row(filt, objs, ego, sc, it, prev)}
        bad_n += int(row["n_cheap"] != int(r0.n_tracks_cheap))
        # diagnostics on the logged tracks
        ref = list(sc.get_tracked_objects_at_iteration(it).tracked_objects)
        gl = m91.geometry(filt, ref, ego)
        tok = np.array([o.metadata.token for o in ref])
        rc, _ = spec[(r0.scenario, it, "cheap")]
        rf, _ = spec[(r0.scenario, it, "full")]
        mc_ = np.isin(tok, list(rc)).astype(float)
        mf_ = np.isin(tok, list(rf)).astype(float)
        assert mc_.sum() == len(rc) and mf_.sum() == len(rf), "a removed token is not among the logged tracks"
        row["crit_sum_gt"] = float(gl["crit"][gl["fov"]].sum())
        row["dE_E1_fn_only"] = float(mc_.sum() - mf_.sum())
        row["dE_E6_risk_weighted"] = float((gl["crit"] * (mc_ - mf_)).sum())
        bad_crit += int(abs(row["crit_sum_gt"] - float(old.loc[(r0.scenario, it), "crit_sum_gt"])) > 1e-9)
        rows.append(row)
        X.append(m104.track_vector(objs, ego, np.ones(len(objs), bool), filt))
        keys.append((r0.scenario, it))
        if n_st % 240 == 0:
            print(f"  {n_st}/1440 states", flush=True)

    checks = {"states": len(rows), "branch_count_mismatch_vs_n_tracks_cheap": bad_n,
              "crit_sum_gt_mismatch_vs_benchmark": bad_crit}
    print("  checks:", checks, flush=True)
    assert bad_n == 0 and bad_crit == 0, "rebuilt branch or reference criticality does not match"
    d = pd.DataFrame(rows)
    d.to_csv(run / "nuplan_real_signals.csv", index=False)
    k = pd.DataFrame(keys, columns=["scenario", "iteration"])
    np.savez_compressed(run / "nuplan_real_track_features.npz", scenario=k.scenario.to_numpy().astype("U32"),
                        iteration=k.iteration.to_numpy(), X=np.asarray(X))
    (run / "roles.json").write_text(json.dumps({"gate_features": m91.GATE_FEATURES, "diagnostic": DIAGNOSTIC,
                                                "checks": checks}, indent=1))
    print(d[m91.GATE_FEATURES + DIAGNOSTIC].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print("  wrote", run)


if __name__ == "__main__":
    main()

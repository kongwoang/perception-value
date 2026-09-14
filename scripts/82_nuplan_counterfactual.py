#!/usr/bin/env python
"""Phase 0G Track B: per-state counterfactual value of perception compute, for published planners.

For every selected state and every external planner q, the same state is presented three times --
with the reference tracks, with the tracks a CHEAP detector would report, and with the tracks a
FULL detector would report -- and the trajectory the planner proposes is scored each time against
the **true** tracked objects.  So

    V_i^q = J_q(CHEAP) - J_q(FULL)

with J external to the planner: geometric collision against real objects, worst clearance, and
progress shortfall.  That is deliberately unlike the Phase 0F Planner C cost, which compared a
planner with itself and was therefore near-circular with PKL.

State handling.  Each state comes from the log, and a *fresh planner instance* is built and
initialised for each branch, so the three branches cannot share hidden state -- PDM-Closed carries
internal state across steps, so re-using one instance would leak the first branch into the second.
This makes the comparison a genuine per-state counterfactual and makes the evaluation open-loop at
the state level, which is stated as a limitation: PDM-Closed is a closed-loop-capable planner but
we are not closing the loop around it.

The perception intervention is `rap.nuplan_perception`, driven by the detector miss model measured
on real YOLOv8s outcomes.  Objects outside the intervention camera, and classes the model was never
fitted on, are passed through untouched.
"""
from __future__ import annotations

import argparse, json, sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))
sys.path.insert(0, str(ROOT / "third_party" / "tuplan_garage"))

from nuplan.common.actor_state.state_representation import TimePoint               # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (   # noqa: E402
    NuPlanScenarioBuilder)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter        # noqa: E402
from nuplan.planning.simulation.history.simulation_history_buffer import (         # noqa: E402
    SimulationHistoryBuffer)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import (                  # noqa: E402
    PlannerInitialization, PlannerInput)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (  # noqa: E402
    SimulationIteration)
from nuplan.planning.utils.multithreading.worker_sequential import Sequential      # noqa: E402

from rap.detector_model import DetectorMissModel                                   # noqa: E402
from rap.nuplan_perception import PerceptionFilter                                 # noqa: E402
from rap.paths import CACHE, RESULTS                                              # noqa: E402

HORIZON_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
MODES = ("reference", "cheap", "full")
SANITY_MODES = ("reference", "drop_all")


def make_planner(kind: str):
    """A fresh instance at the released default configuration."""
    if kind == "idm":
        from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
        return IDMPlanner(target_velocity=10.0, min_gap_to_lead_agent=1.0, headway_time=1.5,
                          accel_max=1.0, decel_max=3.0, planned_trajectory_samples=16,
                          planned_trajectory_sample_interval=0.5, occupancy_map_radius=40.0)
    if kind == "pdm_closed":
        from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import (
            PDMClosedPlanner)
        from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
            PDMObservation)
        from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
            BatchIDMPolicy)
        return PDMClosedPlanner(
            trajectory_sampling=_sampling(8.0, 0.1), proposal_sampling=_sampling(4.0, 0.1),
            idm_policies=BatchIDMPolicy(speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                                        fallback_target_velocity=15.0, min_gap_to_lead_agent=1.0,
                                        headway_time=1.5, accel_max=1.5, decel_max=3.0),
            lateral_offsets=[-1.0, 1.0], map_radius=50.0)
    raise ValueError(kind)


def route_for(kind: str, scenario, iteration: int, route_fix: bool = True):
    """Route roadblocks for a planner instantiated at `iteration` of `scenario`.

    PDM-Closed corrects its route internally (tuPlan Garage's route_roadblock_correction) and receives the scenario
    route unchanged.  The devkit's IDMPlanner looks for the ego's lane only in the first two route roadblocks, so when
    it is instantiated at a state beyond them, or the ego is off the scenario route (pickup/drop-off and stationary
    scenarios), it plans along a wrong path -- tens to hundreds of metres from the ego.  For IDM the route is therefore
    corrected with the same route_roadblock_correction and trimmed to start at the ego's roadblock.  IDM's policy and
    parameters are unchanged.  `route_fix=False` reproduces the original (buggy) behaviour.
    """
    ids = scenario.get_route_roadblock_ids()
    if kind != "idm" or not route_fix:
        return ids
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from shapely.geometry import Point
    from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction

    def block(i):
        return (scenario.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK)
                or scenario.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK_CONNECTOR))

    ego = scenario.get_ego_state_at_iteration(iteration)
    blocks = {i: block(i) for i in dict.fromkeys(ids)}
    corrected = route_roadblock_correction(ego, scenario.map_api, {i: b for i, b in blocks.items() if b is not None})
    pt = Point(ego.center.x, ego.center.y)
    first = 0
    for k, i in enumerate(corrected):
        b = block(i)
        if b is not None and b.polygon.distance(pt) <= 0.5:
            first = k
            break
    trimmed = corrected[first:]
    return trimmed if len(trimmed) >= 2 else corrected


def _sampling(horizon: float, interval: float):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return TrajectorySampling(num_poses=int(round(horizon / interval)), interval_length=interval)


def score(traj, scenario, iteration: int, dt: float) -> dict:
    """Score a proposed trajectory against the REAL tracked objects.

    External to the planner by construction: overlap with true object footprints, worst clearance,
    and how far short of the log's own progress the proposal falls.
    """
    t0 = scenario.get_ego_state_at_iteration(iteration).time_point
    n_iter = scenario.get_number_of_iterations()
    hit, clear, dev = 0.0, np.inf, []
    for s in HORIZON_S:
        try:
            st = traj.get_state_at_time(TimePoint(int(t0.time_us + s * 1e6)))
        except Exception:
            break
        k = min(iteration + int(round(s / dt)), n_iter - 1)
        ego_poly = st.car_footprint.geometry
        for o in scenario.get_tracked_objects_at_iteration(k).tracked_objects:
            g = o.box.geometry
            if ego_poly.intersects(g):
                hit = 1.0
                clear = min(clear, 0.0)
            else:
                clear = min(clear, float(ego_poly.distance(g)))
        true_ego = scenario.get_ego_state_at_iteration(k)
        dev.append(float(np.hypot(st.rear_axle.x - true_ego.rear_axle.x,
                                  st.rear_axle.y - true_ego.rear_axle.y)))
    return {"collision": hit,
            "min_clearance": (float(clear) if np.isfinite(clear) else 20.0),
            "log_deviation_mean": float(np.mean(dev)) if dev else np.nan,
            "log_deviation_final": float(dev[-1]) if dev else np.nan}


def filtered_history(buf, filt, scenario_name: str, iteration: int, mode: str, scenario):
    """A copy of the history buffer with each observation passed through the intervention."""
    egos = list(buf.ego_states)
    obs = []
    for j, (e, o) in enumerate(zip(egos, buf.observations)):
        it = max(iteration - (len(egos) - 1 - j), 0)
        obs.append(filt.apply(o, e, scenario_name, it, mode))
    return SimulationHistoryBuffer(deque(egos), deque(obs), buf.sample_interval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True, choices=["idm", "pdm_closed"])
    ap.add_argument("--scenarios", type=int, default=8)
    ap.add_argument("--per_scenario", type=int, default=6)
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--model", default=str(Path(CACHE) / "detector_miss_model.npz"))
    ap.add_argument("--data_root", default="/home/kongwoang/datasets/nuplan")
    ap.add_argument("--map_root", default="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0")
    ap.add_argument("--out", default=str(Path(RESULTS) / "final"))
    ap.add_argument("--log_limit", type=int, default=2)
    ap.add_argument("--no_route_fix", dest="route_fix", action="store_false",
                    help="hand IDM the scenario route unchanged (the original, buggy behaviour)")
    ap.add_argument("--sanity", action="store_true",
                    help="B6 wiring probe: compare the reference branch with one that removes "
                         "every object in the camera, instead of comparing fidelities")
    args = ap.parse_args()

    z = np.load(args.model, allow_pickle=False)
    filt = PerceptionFilter(DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"]))

    builder = NuPlanScenarioBuilder(
        data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None,
                         map_names=None, num_scenarios_per_type=None,
                         limit_total_scenarios=args.scenarios, timestamp_threshold_s=None,
                         ego_displacement_minimum_m=None, expand_scenarios=False,
                         remove_invalid_goals=True, shuffle=False)
    scenarios = builder.get_scenarios(flt, Sequential())
    print(f"  {len(scenarios)} scenarios, planner={args.planner}")

    rows = []
    for sc in scenarios:
        n = sc.get_number_of_iterations()
        dt = sc.database_interval
        init = PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                    mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
        idxs = np.linspace(args.buffer, max(n - int(4.5 / dt) - 1, args.buffer),
                           args.per_scenario).astype(int)
        for it in sorted(set(int(i) for i in idxs)):
            init = PlannerInitialization(route_roadblock_ids=route_for(args.planner, sc, it, args.route_fix),
                                         mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
            base = SimulationHistoryBuffer.initialize_from_scenario(
                args.buffer, sc, DetectionsTracks)
            # roll the buffer forward to `it` so the state really is this iteration's
            for k in range(max(it - args.buffer + 1, 0), it + 1):
                base.append(sc.get_ego_state_at_iteration(k),
                            sc.get_tracked_objects_at_iteration(k))
            row = {"scenario": sc.token, "log": sc.log_name, "iteration": it,
                   "planner": args.planner, "n_tracks": len(
                       sc.get_tracked_objects_at_iteration(it).tracked_objects)}
            for mode in (SANITY_MODES if args.sanity else MODES):
                p = make_planner(args.planner)
                p.initialize(init)
                hist = filtered_history(base, filt, sc.token, it, mode, sc)
                row[f"n_tracks_{mode}"] = len(hist.observations[-1].tracked_objects)
                try:
                    traj = p.compute_trajectory(PlannerInput(
                        iteration=SimulationIteration(
                            sc.get_ego_state_at_iteration(it).time_point, it),
                        history=hist,
                        traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
                    for k, v in score(traj, sc, it, dt).items():
                        row[f"{k}_{mode}"] = v
                    row[f"ok_{mode}"] = 1
                except Exception as e:                       # noqa: BLE001
                    row[f"ok_{mode}"] = 0
                    row[f"err_{mode}"] = type(e).__name__
            rows.append(row)
            print(f"    {sc.token[:10]} it={it:3d} tracks {row['n_tracks']:3d} "
                  f"-> cheap {row.get('n_tracks_cheap')} full {row.get('n_tracks_full')}",
                  flush=True)

    d = pd.DataFrame(rows)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = "_sanity" if args.sanity else ""
    f = out / f"phase0g_external_{args.planner}{tag}_raw.csv"
    d.to_csv(f, index=False)
    print(f"\n  {len(d)} states, wrote {f}")
    if len(d):
        modes = SANITY_MODES if args.sanity else MODES
        print(f"  branches that computed a trajectory: "
              + "  ".join(f"{m} {int(d[f'ok_{m}'].sum())}/{len(d)}" for m in modes))
        if args.sanity:
            same = 0
            for _, r in d.iterrows():
                a = (r.get("collision_reference"), r.get("min_clearance_reference"),
                     r.get("log_deviation_mean_reference"))
                b = (r.get("collision_drop_all"), r.get("min_clearance_drop_all"),
                     r.get("log_deviation_mean_drop_all"))
                same += int(a == b)
            print(f"\n  B6 WIRING PROBE: {same}/{len(d)} states unchanged when every object in "
                  f"the camera is removed")
            print("  -> " + ("FAIL: the filtered observation is not reaching the planner"
                             if same == len(d) else
                             "PASS: the planner responds to the filtered observation"))


if __name__ == "__main__":
    main()

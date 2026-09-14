#!/usr/bin/env python
"""Diagnostic, train+val states only: does IDMPlanner fail because it is initialised with the scenario-start route
while being called mid-scenario?  For a few states with large and small reference deviation, compare IDM with the
full scenario route (as the pipeline does) and with the route trimmed to start at the ego's current roadblock."""
import importlib.util, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path("/home/kongwoang/research/risk-aware-perception")
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("cf82", ROOT / "scripts" / "82_nuplan_counterfactual.py")
cf82 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf82)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.maps.maps_datatypes import SemanticMapLayer


class Catch(logging.Handler):
    def __init__(self):
        super().__init__(); self.n = 0
    def emit(self, record):
        if "could not find valid path" in record.getMessage():
            self.n += 1


catch = Catch(); logging.getLogger("nuplan.planning.simulation.planner.idm_planner").addHandler(catch)

splits = json.loads((ROOT / "configs/benchmark_splits.json").read_text())["nuplan"]
tv = set(splits["train"] + splits["val"])
raw = pd.read_csv(ROOT / "results/final/phase0g_external_idm_raw.csv")
raw = raw[raw.log.isin(tv)]
pick = pd.concat([raw.nlargest(8, "log_deviation_mean_reference"), raw.nsmallest(3, "log_deviation_mean_reference")])

builder = NuPlanScenarioBuilder(data_root="/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini",
                                map_root="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0", sensor_root=None,
                                db_files=None, map_version="nuplan-maps-v1.0")
flt = ScenarioFilter(scenario_types=None, scenario_tokens=sorted(set(pick.scenario)), log_names=None, map_names=None,
                     num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=None,
                     ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}


def ego_roadblock_index(sc, ids, ego):
    pt = Point(ego.center.x, ego.center.y)
    best, bestd = None, np.inf
    for i, rid in enumerate(ids):
        b = sc.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK) or sc.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if b is None:
            continue
        d = b.polygon.distance(pt)
        if d == 0:
            return i, 0.0
        if d < bestd:
            best, bestd = i, d
    return best, bestd


def run(sc, it, ids):
    base = SimulationHistoryBuffer.initialize_from_scenario(4, sc, DetectionsTracks)
    for k in range(max(it - 3, 0), it + 1):
        base.append(sc.get_ego_state_at_iteration(k), sc.get_tracked_objects_at_iteration(k))
    p = cf82.make_planner("idm")
    p.initialize(PlannerInitialization(route_roadblock_ids=ids, mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    catch.n = 0
    ego = sc.get_ego_state_at_iteration(it)
    traj = p.compute_trajectory(PlannerInput(iteration=SimulationIteration(ego.time_point, it), history=base,
                                             traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
    s = cf82.score(traj, sc, it, sc.database_interval)
    st1 = traj.get_state_at_time(type(ego.time_point)(ego.time_point.time_us + 500_000))
    return {"collision": s["collision"], "logdev_mean": round(s["log_deviation_mean"], 2), "route_warn": catch.n,
            "path_to_ego_m": round(p._ego_path_linestring.distance(Point(ego.center.x, ego.center.y)), 2),
            "jump_0.5s_m": round(float(np.hypot(st1.rear_axle.x - ego.rear_axle.x, st1.rear_axle.y - ego.rear_axle.y)), 2),
            "ego_speed": round(ego.dynamic_car_state.rear_axle_velocity_2d.magnitude(), 2)}


rows = []
for r in pick.itertuples():
    sc, it = scen[r.scenario], int(r.iteration)
    ids = sc.get_route_roadblock_ids()
    ego = sc.get_ego_state_at_iteration(it)
    idx, dist = ego_roadblock_index(sc, ids, ego)
    full = run(sc, it, ids)
    trimmed = run(sc, it, ids[idx:]) if idx is not None and len(ids[idx:]) >= 2 else {}
    rows.append({"scenario": r.scenario[:10], "it": it, "trackb_logdev": round(r.log_deviation_mean_reference, 2),
                 "route_len": len(ids), "ego_roadblock_idx": idx, "ego_to_rb_m": round(dist, 2),
                 **{f"full_{k}": v for k, v in full.items()}, **{f"trim_{k}": v for k, v in trimmed.items()}})
    print(rows[-1], flush=True)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).to_string(index=False))

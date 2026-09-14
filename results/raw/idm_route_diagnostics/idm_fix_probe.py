#!/usr/bin/env python
"""Diagnostic on train+val states only: IDM reference plan with (i) the scenario route as used so far, (ii) the route
corrected by tuPlan Garage's route_roadblock_correction, (iii) corrected and trimmed to start at the ego's roadblock."""
import importlib.util, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
from shapely.geometry import Point
ROOT = Path("/home/kongwoang/research/risk-aware-perception")
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("cf82", ROOT / "scripts/82_nuplan_counterfactual.py"); cf82 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf82)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction
logging.getLogger("nuplan.planning.simulation.planner.idm_planner").setLevel(logging.ERROR)
S = Path(sys.argv[1])
geo = pd.read_csv(S / "route_geometry.csv")
sp = json.loads((ROOT / "configs/benchmark_splits.json").read_text())["nuplan"]; tv = set(sp["train"] + sp["val"])
geo = geo[geo.log.isin(tv)]
rng = np.random.default_rng(0)
pick = pd.concat([geo[~geo.ego_on_route_block].groupby("scenario").head(1),                       # off route: one per scenario
                  geo[geo.ego_on_route_block & (geo.d_route_first2_m > 2)].sample(8, random_state=0),  # on route, beyond the first two
                  geo[geo.d_route_first2_m <= 2].sample(4, random_state=0)])                         # valid already
builder = NuPlanScenarioBuilder(data_root="/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini", map_root="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0",
                                sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
flt = ScenarioFilter(scenario_types=None, scenario_tokens=sorted(set(pick.scenario)), log_names=None, map_names=None, num_scenarios_per_type=None,
                     limit_total_scenarios=None, timestamp_threshold_s=None, ego_displacement_minimum_m=None, expand_scenarios=False,
                     remove_invalid_goals=True, shuffle=False)
scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
def block(sc, i):
    return sc.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK) or sc.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK_CONNECTOR)
def plan(sc, it, ids):
    base = SimulationHistoryBuffer.initialize_from_scenario(4, sc, DetectionsTracks)
    for k in range(max(it - 3, 0), it + 1):
        base.append(sc.get_ego_state_at_iteration(k), sc.get_tracked_objects_at_iteration(k))
    p = cf82.make_planner("idm")
    p.initialize(PlannerInitialization(route_roadblock_ids=ids, mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    ego = sc.get_ego_state_at_iteration(it)
    try:
        traj = p.compute_trajectory(PlannerInput(iteration=SimulationIteration(ego.time_point, it), history=base,
                                                 traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
    except Exception as e:                                                   # noqa: BLE001
        return {"err": type(e).__name__}
    s = cf82.score(traj, sc, it, sc.database_interval)
    st = traj.get_state_at_time(TimePoint(ego.time_point.time_us + 500_000))
    return {"coll": s["collision"], "dev": round(s["log_deviation_mean"], 2),
            "jump05": round(float(np.hypot(st.rear_axle.x - ego.rear_axle.x, st.rear_axle.y - ego.rear_axle.y)), 2)}
rows = []
for r in pick.itertuples():
    sc, it = scen[r.scenario], int(r.iteration)
    ego = sc.get_ego_state_at_iteration(it)
    ids = sc.get_route_roadblock_ids()
    rdict = {i: b for i in dict.fromkeys(ids) if (b := block(sc, i)) is not None}
    corr = route_roadblock_correction(ego, sc.map_api, rdict)
    pt = Point(ego.center.x, ego.center.y)
    first = next((k for k, i in enumerate(corr) if (b := block(sc, i)) is not None and b.polygon.distance(pt) <= 0.5), 0)
    trim = corr[first:] if len(corr[first:]) >= 2 else corr
    rows.append({"scenario": r.scenario[:8], "it": it, "on_route": r.ego_on_route_block, "d_first2": round(r.d_route_first2_m, 1),
                 "trackb_dev": round(r.idm_logdev, 2), "pdm_dev": round(r.pdm_logdev, 2), "route_len": len(ids), "corr_len": len(corr), "trim_start": first,
                 **{f"orig_{k}": v for k, v in plan(sc, it, ids).items()}, **{f"corr_{k}": v for k, v in plan(sc, it, corr).items()},
                 **{f"trim_{k}": v for k, v in plan(sc, it, trim).items()}})
    print(rows[-1], flush=True)
pd.set_option("display.width", 260)
d = pd.DataFrame(rows); print(d.to_string(index=False))
for c in ("orig", "corr", "trim"):
    print(c, "median dev %.2f, max dev %.2f, jumps > 5 m at 0.5 s: %d, collisions %d" % (d[f"{c}_dev"].median(), d[f"{c}_dev"].max(), int((d[f"{c}_jump05"] > 5).sum()), int(d[f"{c}_coll"].sum())))

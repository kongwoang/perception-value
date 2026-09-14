#!/usr/bin/env python
"""Descriptive geometry check at every benchmark state (no outcome is recomputed): where is the ego relative to the
scenario's route roadblocks and to the map's roads, and how does that relate to IDM's and PDM-Closed's reference
deviation from Track B?"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path("/home/kongwoang/research/risk-aware-perception")
sys.path.insert(0, str(ROOT / "src"))
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

out = Path(sys.argv[1])
idm = pd.read_csv(ROOT / "results/final/phase0g_external_idm_raw.csv")
pdm = pd.read_csv(ROOT / "results/final/phase0g_external_pdm_closed_raw.csv")
builder = NuPlanScenarioBuilder(data_root="/home/kongwoang/datasets/nuplan/nuplan-v1.1/splits/mini",
                                map_root="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0", sensor_root=None,
                                db_files=None, map_version="nuplan-maps-v1.0")
flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None, num_scenarios_per_type=None,
                     limit_total_scenarios=60, timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                     expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
LAYERS = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
rows = []
for tok, g in idm.groupby("scenario"):
    sc = scen[tok]
    ids = sc.get_route_roadblock_ids()
    blocks = [sc.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK) or sc.map_api.get_map_object(i, SemanticMapLayer.ROADBLOCK_CONNECTOR)
              for i in ids]
    n_missing = sum(b is None for b in blocks)
    for it in sorted(g.iteration):
        ego = sc.get_ego_state_at_iteration(int(it))
        pt = Point(ego.center.x, ego.center.y)
        d_all = min((b.polygon.distance(pt) for b in blocks if b is not None), default=np.inf)
        d_first2 = min((b.polygon.distance(pt) for b in blocks[:2] if b is not None), default=np.inf)
        prox = sc.map_api.get_proximal_map_objects(ego.center.point, 0.5, LAYERS)
        on_road = any(len(v) for v in prox.values())
        on_route_blocks = {o.id for v in prox.values() for o in v} & set(ids)
        rows.append({"scenario": tok, "log": sc.log_name, "type": sc.scenario_type, "map": sc.map_api.map_name, "iteration": int(it),
                     "route_len": len(ids), "route_ids_missing_in_map": n_missing, "d_route_any_m": d_all,
                     "d_route_first2_m": d_first2, "ego_on_any_road": on_road, "ego_on_route_block": bool(on_route_blocks),
                     "ego_speed": ego.dynamic_car_state.rear_axle_velocity_2d.magnitude()})
    print(f"  {tok[:10]} {sc.scenario_type[:28]:28s} route {len(ids):3d} missing {n_missing} d_any(first state) {rows[-len(g)]['d_route_any_m']:.1f}", flush=True)
d = pd.DataFrame(rows)
d = d.merge(idm[["scenario", "iteration", "collision_reference", "log_deviation_mean_reference"]].rename(
    columns={"collision_reference": "idm_coll", "log_deviation_mean_reference": "idm_logdev"}), on=["scenario", "iteration"])
d = d.merge(pdm[["scenario", "iteration", "collision_reference", "log_deviation_mean_reference"]].rename(
    columns={"collision_reference": "pdm_coll", "log_deviation_mean_reference": "pdm_logdev"}), on=["scenario", "iteration"])
d.to_csv(out, index=False)
d["off_route"] = d.d_route_any_m > 2.0
d["first2_far"] = d.d_route_first2_m > 2.0
pd.set_option("display.width", 250)
for col in ("off_route", "first2_far", "ego_on_route_block"):
    print(f"\n== by {col}:")
    print(d.groupby(col).agg(n=("iteration", "size"), scenarios=("scenario", "nunique"), idm_coll=("idm_coll", "mean"),
                             idm_logdev_mean=("idm_logdev", "mean"), idm_logdev_median=("idm_logdev", "median"),
                             pdm_coll=("pdm_coll", "mean"), pdm_logdev_mean=("pdm_logdev", "mean")).round(3).to_string())
s = d.groupby(["scenario", "type", "log"]).agg(n=("iteration", "size"), d_any_med=("d_route_any_m", "median"),
                                               d_first2_med=("d_route_first2_m", "median"), on_route=("ego_on_route_block", "mean"),
                                               idm_logdev=("idm_logdev", "mean"), pdm_logdev=("pdm_logdev", "mean"),
                                               idm_coll=("idm_coll", "mean"), speed=("ego_speed", "mean"))
print("\n== scenarios with IDM mean log deviation > 10 m:")
print(s[s.idm_logdev > 10].round(2).sort_values("idm_logdev", ascending=False).to_string())

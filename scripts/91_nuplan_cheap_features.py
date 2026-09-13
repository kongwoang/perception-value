#!/usr/bin/env python
"""Benchmark, nuPlan track: per-state allocation signals, rebuilt offline from the Track B states.

Track B stored costs, not what each branch saw.  The perception intervention is deterministic -- a
hash of scenario, iteration and track token -- so the CHEAP and FULL track sets at every state can
be regenerated exactly without running a planner.  From them:

  * cheap-side gate features: only the tracks the CHEAP branch kept, at the decision iteration and
    the one before (both are in the planner's history buffer), the ego state, and traffic-light
    status -- what a vehicle holds before deciding whether to escalate;
  * `unc_proxy`, kept OUT of the gate: the miss model's own detection probability, which no real
    detector reports, so it is privileged;
  * diagnostics that need FULL or ground truth: composite criticality over every track in the
    camera, and the two perception-error variants the miss model can express -- E1 (missed tracks)
    and E6 (criticality-weighted missed tracks).

The regenerated CHEAP and FULL counts are checked against Track B's `n_tracks_cheap/full` for every
state, so a drift in the filter or the model weights cannot pass silently.  Run with scripts/pynuplan.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (   # noqa: E402
    NuPlanScenarioBuilder)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter        # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential      # noqa: E402

from rap.detector_model import NUPLAN_TO_COARSE, DetectorMissModel                 # noqa: E402
from rap.geometry import PRIMARY                                                   # noqa: E402
from rap.nuplan_perception import PerceptionFilter                                 # noqa: E402
from rap.paths import CACHE, RESULTS                                              # noqa: E402

CORRIDOR_HALF_W = 2.0
FAR = 100.0
TTC_CAP = 20.0
CLASSES = sorted(set(NUPLAN_TO_COARSE.values()) - {"static"})

# the only columns a nuPlan gate may use; pre-registered 2026-09-13 23:55
GATE_FEATURES = (["ego_speed", "ego_accel", "n_cheap", "n_cheap_fov"]
                 + [f"n_cheap_fov_{c}" for c in CLASSES]
                 + ["n_static_fov", "min_range_fov", "min_gap_corridor", "n_corridor_20",
                    "n_corridor_40", "min_ttc_corridor", "n_fov_band_30_40", "crit_cheap_sum",
                    "crit_cheap_max", "n_red_lights", "d_n_cheap_fov"])
PRIVILEGED = ["unc_proxy"]
DIAGNOSTIC = ["crit_sum_gt", "dE_E1_fn_only", "dE_E6_risk_weighted"]


def geometry(filt, objs, ego):
    """Per-track camera membership, class, and composite criticality from true track geometry."""
    n = len(objs)
    if n == 0:
        z = np.zeros(0)
        return dict(dist=z, lat=z, fwd=z, coarse=np.zeros(0, "U10"), fov=np.zeros(0, bool),
                    measured=np.zeros(0, bool), long_near=z, ttc=z, crit=z, pc=z)
    dist, lat, coarse, fwd = filt._geometry(objs, ego)
    fov = filt.in_intervention_fov(fwd, lat)
    h = ego.rear_axle.heading
    ch, sh = np.cos(h), np.sin(h)
    v_ego = ego.dynamic_car_state.rear_axle_velocity_2d.x
    length = np.array([o.box.length for o in objs], float)
    width = np.array([o.box.width for o in objs], float)
    v_long = np.array([0.0 if getattr(o, "velocity", None) is None
                       else ch * o.velocity.x + sh * o.velocity.y for o in objs], float)
    long_near = fwd - length / 2.0
    closing = v_ego - v_long
    ttc = np.where(closing > 0.1, np.maximum(long_near, 0.0) / np.maximum(closing, 0.1), np.inf)
    crit = PRIMARY(long_near, lat - width / 2.0, lat + width / 2.0, ttc)
    pc, _, _ = filt.model.probabilities(dist, lat, coarse)
    return dict(dist=dist, lat=lat, fwd=fwd, coarse=coarse, fov=fov, measured=fov & (coarse != "static"),
                long_near=long_near, ttc=ttc, crit=crit, pc=pc)


def n_cheap_fov(filt, sc, it):
    objs = list(sc.get_tracked_objects_at_iteration(it).tracked_objects)
    ego = sc.get_ego_state_at_iteration(it)
    if not objs:
        return 0
    g = geometry(filt, objs, ego)
    keep = filt.keep_mask(objs, ego, sc.token, it, "cheap")
    return int((keep & g["measured"]).sum())


def state_row(filt, sc, it):
    objs = list(sc.get_tracked_objects_at_iteration(it).tracked_objects)
    ego = sc.get_ego_state_at_iteration(it)
    g = geometry(filt, objs, ego)
    kc = filt.keep_mask(objs, ego, sc.token, it, "cheap") if objs else np.zeros(0, bool)
    kf = filt.keep_mask(objs, ego, sc.token, it, "full") if objs else np.zeros(0, bool)
    corridor = (np.abs(g["lat"]) < CORRIDOR_HALF_W) & (g["fwd"] > 0)
    seen = kc & g["fov"]                      # everything CHEAP reports inside the camera
    seen_m = kc & g["measured"]               # ... of the classes the intervention acts on
    mc = kc & corridor
    r = {"ego_speed": ego.dynamic_car_state.rear_axle_velocity_2d.x,
         "ego_accel": ego.dynamic_car_state.rear_axle_acceleration_2d.x,
         "n_cheap": int(kc.sum()), "n_cheap_fov": int(seen_m.sum())}
    for c in CLASSES:
        r[f"n_cheap_fov_{c}"] = int((seen_m & (g["coarse"] == c)).sum())
    r["n_static_fov"] = int((seen & (g["coarse"] == "static")).sum())
    r["min_range_fov"] = float(g["dist"][seen].min()) if seen.any() else FAR
    r["min_gap_corridor"] = float(g["long_near"][mc].min()) if mc.any() else FAR
    r["n_corridor_20"] = int((mc & (g["long_near"] < 20)).sum())
    r["n_corridor_40"] = int((mc & (g["long_near"] < 40)).sum())
    r["min_ttc_corridor"] = float(min(g["ttc"][mc].min(), TTC_CAP)) if mc.any() else TTC_CAP
    r["n_fov_band_30_40"] = int((seen_m & (g["dist"] >= 30) & (g["dist"] < 40)).sum())
    r["crit_cheap_sum"] = float(g["crit"][seen].sum())
    r["crit_cheap_max"] = float(g["crit"][seen].max()) if seen.any() else 0.0
    r["n_red_lights"] = int(sum(1 for t in sc.get_traffic_light_status_at_iteration(it)
                                if t.status.name == "RED"))
    r["d_n_cheap_fov"] = r["n_cheap_fov"] - (n_cheap_fov(filt, sc, it - 1) if it > 0 else r["n_cheap_fov"])
    # privileged and diagnostic
    r["unc_proxy"] = float((1.0 - g["pc"][seen_m]).sum())
    miss_c = (g["measured"] & ~kc).astype(float)
    miss_f = (g["measured"] & ~kf).astype(float)
    r["crit_sum_gt"] = float(g["crit"][g["fov"]].sum())
    r["dE_E1_fn_only"] = float(miss_c.sum() - miss_f.sum())
    r["dE_E6_risk_weighted"] = float((g["crit"] * (miss_c - miss_f)).sum())
    r["_n_cheap_check"], r["_n_full_check"] = int(kc.sum()), int(kf.sum())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path(CACHE) / "detector_miss_model.npz"))
    ap.add_argument("--data_root", default="/home/kongwoang/datasets/nuplan")
    ap.add_argument("--map_root", default="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0")
    args = ap.parse_args()

    z = np.load(args.model, allow_pickle=False)
    filt = PerceptionFilter(DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"]))
    raw = {p: pd.read_csv(Path(RESULTS) / "final" / f"phase0g_external_{p}_raw.csv")
           for p in ("idm", "pdm_closed")}
    states = raw["idm"][["scenario", "iteration"]].drop_duplicates()
    assert len(states) == len(raw["pdm_closed"][["scenario", "iteration"]].drop_duplicates().merge(states))

    builder = NuPlanScenarioBuilder(
        data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60,
                         timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                         expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert set(scen) == set(states.scenario), "scenario set differs from Track B"

    rows = []
    for tok, g in states.groupby("scenario", sort=True):
        sc = scen[tok]
        for it in sorted(g.iteration):
            rows.append({"scenario": tok, "log": sc.log_name, "iteration": int(it),
                         **state_row(filt, sc, int(it))})
        print(f"  {tok[:12]} {len(g)} states", flush=True)
    d = pd.DataFrame(rows)

    # every regenerated count must match what the Track B branches actually saw
    for p, r in raw.items():
        m = d.merge(r[["scenario", "iteration", "n_tracks_cheap", "n_tracks_full"]],
                    on=["scenario", "iteration"], validate="one_to_one")
        bad = int(((m._n_cheap_check != m.n_tracks_cheap) | (m._n_full_check != m.n_tracks_full)).sum())
        print(f"  {p}: regenerated CHEAP/FULL track counts differ from Track B on {bad}/{len(m)} states")
        assert bad == 0, "the offline intervention does not reproduce Track B"
    assert not set(GATE_FEATURES) & set(PRIVILEGED + DIAGNOSTIC)

    out = Path(RESULTS) / "final" / "benchmark_nuplan_signals.csv"
    d.drop(columns=["_n_cheap_check", "_n_full_check"]).to_csv(out, index=False)
    # the table script runs in the project env and cannot import this module, so the column roles
    # travel with the data
    out.with_suffix(".json").write_text(json.dumps(
        {"gate_features": GATE_FEATURES, "privileged": PRIVILEGED, "diagnostic": DIAGNOSTIC}, indent=1))
    print(f"  {len(d)} states, {len(GATE_FEATURES)} gate features; wrote {out}")
    print(d[GATE_FEATURES + PRIVILEGED + DIAGNOSTIC].describe().T[["mean", "std", "min", "max"]]
          .round(3).to_string())


if __name__ == "__main__":
    main()

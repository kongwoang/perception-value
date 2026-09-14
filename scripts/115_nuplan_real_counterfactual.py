#!/usr/bin/env python
"""Task 5 C1: PDM-Closed and IDM on real-perception branch tracks, through 82's filtered_history hook.

Pre-registered in RESEARCH_LOG.md (2026-09-14 14:44).  `make_planner`, `score` and `filtered_history` are
imported from scripts/82_nuplan_counterfactual.py unchanged; only the observation filter differs.
`RealPerceptionFilter` has the same `apply(detections, ego, scenario, iteration, mode)` interface as
`rap.nuplan_perception.PerceptionFilter` and returns, per branch, the logged tracks minus the eligible
in-camera tracks that branch's detections did not match, plus its false-positive agents (from 114).

  --phase ref       reference branch on all 1,440 states, the identity branch (every in-camera track
                    matched, no false positive) planned on states 1 and 13 of each scenario, and on
                    all states compared observation-for-observation with the reference.  Checks 6-7.
  --phase branches  cheap, full, cheap_nofp, full_nofp, full_s1, cheap_iou50, full_iou50.  Branches
                    whose four buffered observations are identical share one planner call.

Checkpointed per scenario.  Run with scripts/pynuplan, one planner per process.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))
sys.path.insert(0, str(ROOT / "third_party" / "tuplan_garage"))

_spec = importlib.util.spec_from_file_location("cf82", ROOT / "scripts" / "82_nuplan_counterfactual.py")
cf82 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf82)

from nuplan.common.actor_state.agent import Agent                                  # noqa: E402
from nuplan.common.actor_state.oriented_box import OrientedBox                     # noqa: E402
from nuplan.common.actor_state.scene_object import SceneObjectMetadata             # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D  # noqa: E402
from nuplan.common.actor_state.tracked_objects import TrackedObjects               # noqa: E402
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType      # noqa: E402
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

from rap.paths import CACHE, RESULTS                                              # noqa: E402

WORK = ROOT / "results" / "raw" / "nuplan_task5"
OUTC = Path(CACHE) / "nuplan_real"
SCORE_KEYS = ("collision", "min_clearance", "log_deviation_mean", "log_deviation_final")
REAL_BRANCHES = ("cheap", "full", "cheap_nofp", "full_nofp", "full_s1", "cheap_iou50", "full_iou50")
IDENTITY_STATE_POS = (0, 12)


class RealPerceptionFilter:
    """Branch observations from the 114 specification; the reference passes every track through."""

    def __init__(self, spec: dict):
        self.spec = spec

    @staticmethod
    def _fp_agent(fp, timestamp_us: int) -> Agent:
        token, cat, x, y, yaw, length, width, height = fp
        return Agent(tracked_object_type=TrackedObjectType[cat.upper()],
                     oriented_box=OrientedBox(StateSE2(x, y, yaw), width=width, length=length, height=height),
                     velocity=StateVector2D(0.0, 0.0), predictions=[], angular_velocity=np.nan,
                     metadata=SceneObjectMetadata(
                         timestamp_us=timestamp_us, token=token, track_token=token,
                         track_id=10 ** 9 + int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16),
                         category_name=cat))

    def apply(self, detections, ego, scenario: str, iteration: int, mode: str):
        objs = list(detections.tracked_objects.tracked_objects)
        if mode == "reference":
            return DetectionsTracks(TrackedObjects(objs))
        removed, fps = self.spec[(scenario, iteration, mode)]
        kept = [o for o in objs if o.metadata.token not in removed]
        ts = objs[0].metadata.timestamp_us if objs else ego.time_point.time_us
        return DetectionsTracks(TrackedObjects(kept + [self._fp_agent(f, ts) for f in fps]))


def obs_signature(hist) -> str:
    """Exact content of the buffered observations, in the order the planner receives them."""
    h = hashlib.blake2b(digest_size=20)
    for o in hist.observations:
        for ob in o.tracked_objects.tracked_objects:
            v = getattr(ob, "velocity", None)
            h.update(repr((ob.metadata.token, ob.tracked_object_type.name, ob.center.x, ob.center.y,
                           ob.center.heading, ob.box.length, ob.box.width, ob.box.height,
                           None if v is None else (v.x, v.y))).encode())
        h.update(b"|")
    return h.hexdigest()


def run_planner(planner, sc, it, init, hist, dt):
    p = cf82.make_planner(planner)
    p.initialize(init)
    out = {}
    try:
        traj = p.compute_trajectory(PlannerInput(
            iteration=SimulationIteration(sc.get_ego_state_at_iteration(it).time_point, it),
            history=hist, traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
        out.update(cf82.score(traj, sc, it, dt))
        out["ok"] = 1
    except Exception as e:                                                     # noqa: BLE001
        out.update(ok=0, err=type(e).__name__)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", required=True, choices=["idm", "pdm_closed"])
    ap.add_argument("--phase", required=True, choices=["ref", "branches"])
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--no_route_fix", action="store_true", help="hand IDM the scenario route unchanged (original behaviour)")
    ap.add_argument("--data_root", default="/home/kongwoang/datasets/nuplan")
    ap.add_argument("--map_root", default="/home/kongwoang/datasets/nuplan/nuplan-maps-v1.0")
    args = ap.parse_args()
    t_start = time.time()

    with open(OUTC / "branches.pkl", "rb") as f:
        filt = RealPerceptionFilter(pickle.load(f)["spec"])
    trackb = pd.read_csv(Path(RESULTS) / "final" / f"phase0g_external_{args.planner}_raw.csv")
    states = trackb[["scenario", "log", "iteration"]]
    assert len(states) == 1440 and not states.duplicated().any()

    builder = NuPlanScenarioBuilder(
        data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
        sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60,
                         timestamp_threshold_s=None, ego_displacement_minimum_m=None,
                         expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert set(scen) == set(states.scenario), "scenario set differs from Track B"

    ckpt = OUTC / f"rows_{args.planner}_{args.phase}"
    ckpt.mkdir(parents=True, exist_ok=True)
    ref_rows = {}
    if args.phase == "branches":
        for f in (OUTC / f"rows_{args.planner}_ref").glob("*.json"):
            for r in json.loads(f.read_text()):
                ref_rows[(r["scenario"], r["iteration"])] = r
        assert len(ref_rows) == 1440, f"reference phase incomplete: {len(ref_rows)} states"

    calls = 0
    for n_sc, (tok, g) in enumerate(states.groupby("scenario", sort=True), 1):
        out_f = ckpt / f"{tok}.json"
        if out_f.exists():
            continue
        sc = scen[tok]
        dt = sc.database_interval
        init = PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
        rows = []
        for pos, it in enumerate(sorted(int(i) for i in g.iteration)):
            init = PlannerInitialization(route_roadblock_ids=cf82.route_for(args.planner, sc, it, not args.no_route_fix),
                                         mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
            base = SimulationHistoryBuffer.initialize_from_scenario(args.buffer, sc, DetectionsTracks)
            for k in range(max(it - args.buffer + 1, 0), it + 1):
                base.append(sc.get_ego_state_at_iteration(k), sc.get_tracked_objects_at_iteration(k))
            row = {"scenario": tok, "log": sc.log_name, "iteration": it, "planner": args.planner}
            by_sig = {}
            h_ref = cf82.filtered_history(base, filt, tok, it, "reference", sc)
            sig_ref = obs_signature(h_ref)
            row["sig_reference"] = sig_ref
            row["n_tracks_reference"] = len(h_ref.observations[-1].tracked_objects)
            if args.phase == "ref":
                res = run_planner(args.planner, sc, it, init, h_ref, dt); calls += 1
                row.update({f"{k}_reference": v for k, v in res.items()})
                h_id = cf82.filtered_history(base, filt, tok, it, "identity", sc)
                row["sig_identity_equals_reference"] = obs_signature(h_id) == sig_ref
                row["n_tracks_identity"] = len(h_id.observations[-1].tracked_objects)
                if pos in IDENTITY_STATE_POS:            # planned without de-duplication, on purpose
                    res = run_planner(args.planner, sc, it, init, h_id, dt); calls += 1
                    row.update({f"{k}_identity": v for k, v in res.items()})
            else:
                ref = ref_rows[(tok, it)]
                assert ref["sig_reference"] == sig_ref, "reference observation changed between phases"
                by_sig[sig_ref] = ("reference", {k: ref.get(f"{k}_reference") for k in SCORE_KEYS + ("ok", "err")})
                for b in REAL_BRANCHES:
                    h = cf82.filtered_history(base, filt, tok, it, b, sc)
                    s = obs_signature(h)
                    row[f"n_tracks_{b}"] = len(h.observations[-1].tracked_objects)
                    if s in by_sig:
                        src, res = by_sig[s]
                        row[f"shared_{b}"] = src
                    else:
                        res = run_planner(args.planner, sc, it, init, h, dt); calls += 1
                        by_sig[s] = (b, res)
                        row[f"shared_{b}"] = ""
                    row.update({f"{k}_{b}": v for k, v in res.items() if v is not None})
            rows.append(row)
        out_f.write_text(json.dumps(rows))
        print(f"  [{n_sc:2d}/60] {tok} {len(rows)} states, {calls} planner calls so far, "
              f"{time.time() - t_start:6.0f}s", flush=True)

    d = pd.DataFrame([r for f in sorted(ckpt.glob("*.json")) for r in json.loads(f.read_text())])
    d = d.sort_values(["scenario", "iteration"]).reset_index(drop=True)
    d.to_csv(OUTC / f"{args.planner}_{args.phase}.csv", index=False)

    if args.phase == "ref":
        m = d.merge(trackb, on=["scenario", "log", "iteration"], suffixes=("", "_trackb"), validate="one_to_one")
        mism = {}
        for k in SCORE_KEYS:
            a, b = m[f"{k}_reference"].to_numpy(float), m[f"{k}_reference_trackb"].to_numpy(float)
            bad = ~((a == b) | (np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-9))
            mism[k] = int(bad.sum())
        ident = d[d.get("ok_identity").notna()] if "ok_identity" in d else d.iloc[0:0]
        id_bad = 0
        for k in SCORE_KEYS:
            a, b = ident[f"{k}_identity"].to_numpy(float), ident[f"{k}_reference"].to_numpy(float)
            id_bad += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())
        checks = {"planner": args.planner, "states": len(d),
                  "reference_ok": int(d.ok_reference.sum()),
                  "reference_vs_trackb_mismatches": mism,
                  "reference_reproduces_trackb": all(v == 0 for v in mism.values()),
                  "identity_observation_equal_states": int(d.sig_identity_equals_reference.sum()),
                  "identity_planned_states": len(ident), "identity_planner_mismatches": id_bad,
                  "identity_reproduces_reference": bool(d.sig_identity_equals_reference.all() and id_bad == 0
                                                        and len(ident) == 120),
                  "planner_calls": calls, "seconds": time.time() - t_start}
        (WORK / f"checks_reference_{args.planner}.json").write_text(json.dumps(checks, indent=1))
        print(json.dumps(checks, indent=1))
        if not (checks["reference_reproduces_trackb"] and checks["identity_reproduces_reference"]):
            print("  CHECK FAILED: stop before scoring any CHEAP/FULL branch")
            sys.exit(3)
    else:
        ref = pd.DataFrame(ref_rows.values())[["scenario", "iteration"] + [c for c in pd.DataFrame(ref_rows.values())
                                                                          .columns if c.endswith("_reference")]]
        full = d.drop(columns=[c for c in d.columns if c.endswith("_reference")]).merge(
            ref, on=["scenario", "iteration"], validate="one_to_one")
        out = Path(RESULTS) / "final" / f"nuplan_real_perception_{args.planner}_raw.csv"
        full.to_csv(out, index=False)
        shared = {b: int((full[f"shared_{b}"].fillna("") != "").sum()) for b in REAL_BRANCHES}
        print(f"  wrote {out}: {len(full)} states; planner calls {calls}; shared results {shared}; "
              f"ok " + " ".join(f"{b} {int(full[f'ok_{b}'].sum())}" for b in REAL_BRANCHES))


if __name__ == "__main__":
    main()

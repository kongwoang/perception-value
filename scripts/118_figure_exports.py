#!/usr/bin/env python
"""Task 6: figure data export -- export only, no new analysis, no change to any registered result.

nuScenes, YOLOv8s 320 -> 640, threshold 0.25, cached detections (data/cache/nusc_det_tv), the existing
class-agnostic greedy matching at IoU 0.5 (`rap.risk.match`), and the geometry the decision pipeline
used (`rap.decision.build`: mono = the cached monocular geometry; oracle = GT geometry for detections
that overlap a GT box at IoU >= 0.5).  Each frame is rebuilt through the same functions, in the same
order (the braking cost carries the previous action), and the rebuilt J_cheap / J_full are asserted equal
to the registered joined tables before anything is written.

  A  results/final/fig_bev_objects.csv.gz + fig_bev_constants.json
     one row per object per frame with V != 0: q_brake (mono, oracle) and q_plan ADE (oracle)
  B  results/final/fig_gallery/: q_brake mono, the 6 most negative and 6 most positive V frames
  C  results/final/fig_budget_curves.csv: benchmark_budget_routers.csv, unit ms, per cell + medians
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, geometry as G, planner as P                         # noqa: E402
from rap.cache import DetCache                                                 # noqa: E402
from rap.mono import box_iou                                                   # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                  # noqa: E402
from rap.nusc_submission import COARSE_TO_NUSC, SIZE_PRIOR                     # noqa: E402
from rap.paths import CACHE                                                    # noqa: E402
from rap.risk import RiskConfig, match                                         # noqa: E402

RAW = ROOT / "results" / "raw"
FINAL = ROOT / "results" / "final"
DET = Path(CACHE) / "nusc_det_tv"
CHEAP, FULL = "ns_cheap_320", "ns_full_640"
JOINED = {"oracle": RAW / "20260913_211441_phase0g_eta_fde_oracle" / "joined_frames.pkl",
          "mono": RAW / "20260913_214436_phase0g_eta_fde_mono" / "joined_frames.pkl"}
FIGURE1_EXCLUDE = {("scene-0032", 9), ("scene-0048", 3)}
EPS = 1e-9


def match_pairs(gt, dets_xyxy, conf, cfg):
    """`rap.risk.match`'s greedy confidence-ordered assignment, returning the (det, gt) pairs.

    The pairs are re-derived here because `match` returns only per-GT best IoU and a per-detection
    matched mask; both are asserted equal to `match`'s own output, so this is the same matching.
    """
    n_gt, n_det = len(gt), len(dets_xyxy)
    det_gt = np.full(n_det, -1, int)
    if n_gt and n_det:
        iou = box_iou(dets_xyxy, gt)
        taken = np.zeros(n_gt, bool)
        for di in np.argsort(-conf):
            row = np.where(taken, -1.0, iou[di])
            gi = int(np.argmax(row))
            if row[gi] >= cfg.iou_thr:
                taken[gi] = True
                det_gt[di] = gi
    best, matched = match(gt, np.array(["v"] * n_gt), dets_xyxy, conf, np.array(["v"] * n_det), cfg)
    assert np.array_equal(matched, det_gt >= 0)
    # the existing definition of "matched" for a GT object: best IoU >= 0.5, where an unassigned GT keeps its
    # best overlap with ANY detection (rap.risk.match); the registered miss counts use exactly this
    gt_hit = best >= cfg.iou_thr
    gt_det = np.full(n_gt, -1, int)
    gt_det[det_gt[det_gt >= 0]] = np.flatnonzero(det_gt >= 0)
    assert gt_hit[gt_det >= 0].all()
    shared = gt_hit & (gt_det < 0)            # hit through a detection assigned to another GT
    for gi in np.flatnonzero(shared):
        gt_det[gi] = int(iou[:, gi].argmax())
    return det_gt, gt_hit, gt_det, shared


def pair_fps(bc, bf, thr):
    """Cross-mode correspondence of unmatched detections: greedy by IoU, exclusive, IoU >= thr (0.5 as exported)."""
    pc, pf = np.full(len(bc), -1, int), np.full(len(bf), -1, int)
    if len(bc) and len(bf):
        iou = box_iou(bc, bf)
        for ci, fi in np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]:
            if iou[ci, fi] < thr:
                break
            if pc[ci] < 0 and pf[fi] < 0:
                pc[ci], pf[fi] = fi, ci
    return pc, pf


def rebuild(db, adapter, seqs, cfg, pp, cp, geometry):
    """decision.build's frame loop, keeping per-object state; asserts the registered J values."""
    rng = np.random.default_rng(0)
    frames = {}
    for s in seqs:
        geom = adapter.geometry(s)
        speeds = adapter.speeds(s)
        c, f = DetCache(DET / CHEAP / f"{s}.npz"), DetCache(DET / FULL / f"{s}.npz")
        prev_c = prev_f = None
        inst = {v: k for k, v in db._track_map[s].items()}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            j = f.index[fr]
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))
            out = {}
            for tag, cache, idx in (("cheap", c, i), ("full", f, j)):
                d, geo = cache.det(idx), cache.geo(idx)
                k = d["conf"] >= cfg.thr(tag)
                geo_k = {kk: vv[k] for kk, vv in geo.items()}
                bx = d["xyxy"][k].astype(float)
                z, lo, hi, tt = decision._apply_range_source(bx, geo_k, g, geometry, rng, 0.0)
                a, a_obj = P.required_decel(z, lo, hi, tt, v, pp)
                det_gt, gt_hit, gt_det, gt_shared = match_pairs(gt, bx, d["conf"][k].astype(float), cfg)
                out[tag] = dict(xyxy=bx, conf=d["conf"][k], coarse=d["coarse"][k], z=z, lo=lo, hi=hi, ttc=tt,
                                a=a, a_obj=a_obj, det_gt=det_gt, gt_hit=gt_hit, gt_det=gt_det, gt_shared=gt_shared)
            a_gt, a_gt_obj = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act_c, act_f = P.discrete_action(out["cheap"]["a"], pp), P.discrete_action(out["full"]["a"], pp)
            Jc = P.decision_cost(act_c, a_gt, prev_c, pp, cp)
            Jf = P.decision_cost(act_f, a_gt, prev_f, pp, cp)
            prev_c, prev_f = act_c, act_f
            frames[(s, fr)] = dict(v=v, g=g, gt=gt, a_gt=a_gt, a_gt_obj=a_gt_obj, act_c=act_c, act_f=act_f,
                                   act_ref=P.discrete_action(a_gt, pp), J_cheap=Jc["J"], J_full=Jf["J"],
                                   out=out, inst=inst)
    return frames


def corridor(lo, hi, z, pp):
    lat = P.in_corridor(lo, hi, pp.corridor_half_w)
    return lat, lat & (np.asarray(z) <= pp.max_range)


def plan_position(o, k, g, cfg):
    """Box centre in the ego frame as Planner C's oracle submission placed it (rap.nusc_submission)."""
    if len(g):
        iou = box_iou(o["xyxy"][k:k + 1], np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float))
        if iou.max() >= cfg.iou_thr:
            gi = int(iou.argmax())
            return float(g["lat_cen"][gi]), float(g["long_cen"][gi]), "gt_box_centre"
    L = SIZE_PRIOR[COARSE_TO_NUSC.get(str(o["coarse"][k]), "car")][1]
    return float((o["lo"][k] + o["hi"][k]) / 2), float(o["z"][k] + L / 2), "mono_lift_centre"


def object_rows(fr, pp, cfg, base):
    """The five exported object types for one frame, positions as each mode's controller sees them."""
    oc, of, g = fr["out"]["cheap"], fr["out"]["full"], fr["g"]
    rows, omitted = [], {"fp_in_both": 0, "missed_by_both": 0, "fp_in_both_iou30": 0}

    def det_view(o, k):
        lat, cons = corridor(o["lo"][k:k + 1], o["hi"][k:k + 1], o["z"][k:k + 1], pp)
        return dict(x=float((o["lo"][k] + o["hi"][k]) / 2), z=float(o["z"][k]), lo=float(o["lo"][k]),
                    hi=float(o["hi"][k]), inc=bool(lat[0]), cons=bool(cons[0]), a_obj=float(o["a_obj"][k]))

    def emit(typ, cls, gi, kc, kf, primary):
        r = dict(base, type=typ, **{"class": cls})
        pv = det_view(*primary)
        r.update(x_lateral_m=pv["x"], z_forward_m=pv["z"], lat_min_m=pv["lo"], lat_max_m=pv["hi"],
                 in_corridor=pv["inc"], considered_by_controller=pv["cons"], a_req_obj=pv["a_obj"],
                 conf_cheap=float(oc["conf"][kc]) if kc is not None else np.nan,
                 conf_full=float(of["conf"][kf]) if kf is not None else np.nan,
                 gt_id=int(g["track_id"][gi]) if gi is not None else np.nan,
                 instance_token=fr["inst"].get(int(g["track_id"][gi])) if gi is not None else None,
                 gt_type=str(g["type"][gi]) if gi is not None else None,
                 det_index_cheap=kc if kc is not None else -1, det_index_full=kf if kf is not None else -1,
                 det_shared_with_other_gt_cheap=bool(oc["gt_shared"][gi]) if (gi is not None and kc is not None) else None,
                 det_shared_with_other_gt_full=bool(of["gt_shared"][gi]) if (gi is not None and kf is not None) else None)
        if typ == "matched_both":
            fv = det_view(of, kf)
            r.update(x_lateral_full_m=fv["x"], z_forward_full_m=fv["z"], in_corridor_full=fv["inc"],
                     considered_by_controller_full=fv["cons"])
        rows.append(r)

    for gi in range(len(g)):
        hc, hf = bool(oc["gt_hit"][gi]), bool(of["gt_hit"][gi])
        kc = int(oc["gt_det"][gi]) if oc["gt_det"][gi] >= 0 else None
        kf = int(of["gt_det"][gi]) if of["gt_det"][gi] >= 0 else None
        if hc and hf:
            emit("matched_both", str(oc["coarse"][kc]), gi, kc, kf, (oc, kc))
        elif hf:
            emit("miss_recovered_by_full", str(of["coarse"][kf]), gi, None, kf, (of, kf))
        elif hc:
            emit("miss_lost_by_full", str(oc["coarse"][kc]), gi, kc, None, (oc, kc))
        else:
            omitted["missed_by_both"] += 1
    uc, uf = np.flatnonzero(oc["det_gt"] < 0), np.flatnonzero(of["det_gt"] < 0)
    pc, pf = pair_fps(oc["xyxy"][uc], of["xyxy"][uf], cfg.iou_thr)
    for a, k in enumerate(uf):
        if pf[a] < 0:
            emit("fp_added_by_full", str(of["coarse"][k]), None, None, int(k), (of, int(k)))
    for a, k in enumerate(uc):
        if pc[a] < 0:
            emit("fp_removed_by_full", str(oc["coarse"][k]), None, int(k), None, (oc, int(k)))
    omitted["fp_in_both"] += int((pc >= 0).sum())
    omitted["fp_in_both_iou30"] += int((pair_fps(oc["xyxy"][uc], of["xyxy"][uf], 0.3)[0] >= 0).sum())
    return rows, omitted


def main():
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    assert cfg.op_conf == 0.25 and cfg.op_conf_full is None and cfg.iou_thr == 0.5
    seqs = [p.stem for p in sorted((DET / CHEAP).glob("*.npz"))]
    db = NuScenesDB("/home/kongwoang/datasets/nuscenes/trainval", "v1.0-trainval")
    adapter = make_adapter(db)
    for s in seqs:
        adapter.geometry(s)                      # builds the track-id map used for instance tokens

    verify, rows, omitted = {}, [], {}
    frames = {}
    for geom in ("mono", "oracle"):
        jt = pd.read_pickle(JOINED[geom]).set_index(["seq", "frame"])
        frames[geom] = rebuild(db, adapter, seqs, cfg, pp, cp, geom)
        dc = np.array([frames[geom][k]["J_cheap"] - jt.loc[k, "J_cheap"] for k in frames[geom]])
        df = np.array([frames[geom][k]["J_full"] - jt.loc[k, "J_full"] for k in frames[geom]])
        da = np.array([(frames[geom][k]["act_c"] != jt.loc[k, "act_cheap"]) or (frames[geom][k]["act_f"] != jt.loc[k, "act_full"])
                       for k in frames[geom]])
        verify[geom] = {"frames": len(frames[geom]), "J_cheap_max_abs_diff": float(np.abs(dc).max()),
                        "J_full_max_abs_diff": float(np.abs(df).max()), "action_mismatches": int(da.sum())}
        print(f"  {geom}: rebuilt {len(frames[geom])} frames, {verify[geom]}", flush=True)
        assert len(frames[geom]) == len(jt) == 3376 and np.abs(dc).max() < EPS and np.abs(df).max() < EPS and not da.any(), \
            "rebuild does not reproduce the registered decision table"
        systems = [("brake", "J_cheap", "J_full")] + ([("plan_ade", "JC_ade_cheap", "JC_ade_full")] if geom == "oracle" else [])
        for system, cc, cf in systems:
            V = (jt[cc] - jt[cf])
            keep = V[(V.abs() > EPS) & jt[cc].notna() & jt[cf].notna()]
            om = {"fp_in_both": 0, "missed_by_both": 0, "fp_in_both_iou30": 0}
            for (s, fr), v in keep.items():
                base = dict(scene=s, frame=int(fr), sample_token=jt.loc[(s, fr), "sample_token"], system=system,
                            geometry=geom, V=float(v), sign_V=int(np.sign(v)))
                r, o = object_rows(frames[geom][(s, fr)], pp, cfg, base)
                if system == "plan_ade":
                    f_ = frames[geom][(s, fr)]
                    for x in r:
                        mode = "full" if x["type"] in ("fp_added_by_full", "miss_recovered_by_full") else "cheap"
                        k = x["det_index_full"] if mode == "full" else x["det_index_cheap"]
                        x["x_lateral_m"], x["z_forward_m"], x["position_source"] = plan_position(f_["out"][mode], k, f_["g"], cfg)
                rows += r
                for kk in om:
                    om[kk] += o[kk]
            omitted[f"{system}|{geom}"] = {"frames_V_nonzero": int(len(keep)), **om}
    bev = pd.DataFrame(rows)
    if "position_source" not in bev:
        bev["position_source"] = None
    bev["position_source"] = bev.position_source.fillna("controller_near_face")
    bev["position_basis"] = np.where(bev.system == "brake", "near_face", "box_centre")
    bev.to_csv(FINAL / "fig_bev_objects.csv.gz", index=False)
    counts = bev.groupby(["system", "geometry", "type"]).size().unstack(fill_value=0)
    shared_rows = int((bev.det_shared_with_other_gt_cheap.fillna(False).astype(bool)
                       | bev.det_shared_with_other_gt_full.fillna(False).astype(bool)).sum())
    print(f"  rows matched through a detection shared with another GT: {shared_rows}", flush=True)
    print(counts.to_string(), flush=True)

    onset = {f"{v:g}": {"decelerate_below_m": float(v ** 2 / (2 * pp.thr_decel) + pp.standoff + v * pp.reaction_t),
                        "hard_brake_below_m": float(v ** 2 / (2 * pp.thr_hard) + pp.standoff + v * pp.reaction_t)}
             for v in (5.0, 10.0, 15.0, 20.0)}
    const = {
        "source": "scripts/118_figure_exports.py",
        "setting": {"dataset": "nuScenes v1.0-trainval, 85 scenes, 3,376 CAM_FRONT keyframes", "detector": "YOLOv8s",
                    "cheap": CHEAP, "full": FULL, "threshold_cheap": cfg.thr("cheap"), "threshold_full": cfg.thr("full"),
                    "match_iou": cfg.iou_thr, "matching": "rap.risk.match: greedy, confidence-ordered, class-agnostic, one-to-one",
                    "min_gt_height_px": cfg.min_gt_height, "mono_camera_height_m": 1.51,
                    "registered_tables": {k: str(v.relative_to(ROOT)) for k, v in JOINED.items()}},
        "controller": {"corridor_half_width_m": pp.corridor_half_w, "max_range_m": pp.max_range, "standoff_m": pp.standoff,
                       "reaction_time_s": pp.reaction_t, "a_decelerate_mps2": pp.a_decel, "a_hard_brake_mps2": pp.a_hard,
                       "threshold_decelerate_mps2": pp.thr_decel, "threshold_hard_brake_mps2": pp.thr_hard,
                       "a_cap_mps2": pp.a_cap, "use_ttc": pp.use_ttc, "actions": list(P.ACTION_NAMES),
                       "braking_onset_distance_static_obstacle_by_ego_speed_mps": onset,
                       "onset_formula": "d = v^2 / (2 a_threshold) + standoff + v * reaction_time (ttc = inf)"},
        "columns": {
            "V": "J(CHEAP) - J(FULL) from the registered tables; brake = J, plan_ade = JC_ade (Planner C vs the real future)",
            "x_lateral_m": "ego frame lateral centre (lat_min + lat_max)/2, nuScenes ego y, POSITIVE TO THE LEFT",
            "z_forward_m": "ego frame forward range as the braking controller uses it: near-face range (mono: cached "
                           "monocular z; oracle: GT long_near for detections overlapping a GT box at IoU >= 0.5)",
            "position_source": "controller_near_face for brake rows; for plan_ade rows the box centre Planner C's oracle "
                               "submission used (gt_box_centre, or mono_lift_centre = z + class-prior length/2)",
            "position_basis": "near_face (q_brake rows) or box_centre (q_plan rows)",
            "in_corridor": "the controller's corridor test rap.planner.in_corridor(lat_min, lat_max, corridor_half_width)",
            "considered_by_controller": "in_corridor and z_forward_m <= max_range (the selection inside required_decel)",
            "x_lateral_full_m / z_forward_full_m / in_corridor_full": "matched_both only: the FULL detection's view "
                                                                      "(the primary columns are the CHEAP detection's)",
            "a_req_obj": "the controller's per-object required deceleration for that mode's detection",
            "class": "detector coarse class of the detection the row is about", "gt_id": "track id in rap.nusc geometry",
            "det_index_cheap / det_index_full": "index of the detection among that mode's detections at threshold 0.25 "
                                                "(the order of the cached detections), -1 if none",
        },
        "types": {"matched_both": "GT matched by CHEAP and by FULL", "miss_recovered_by_full": "GT missed by CHEAP, matched by FULL",
                  "miss_lost_by_full": "GT matched by CHEAP, missed by FULL",
                  "fp_added_by_full": "FULL detection matched to no GT and paired with no unmatched CHEAP detection (IoU >= 0.5)",
                  "fp_removed_by_full": "CHEAP detection matched to no GT and paired with no unmatched FULL detection (IoU >= 0.5)",
                  "not_exported": "GT missed by both modes, and unmatched detections present in both modes (counts below)"},
        "gt_matched_note": "a GT object counts as matched by a mode when rap.risk.match's best IoU is >= 0.5. For an "
                           "unassigned GT that value is its best overlap with any detection, so a GT can be matched through "
                           "a detection assigned to another GT; the registered miss counts use the same definition. Such "
                           "rows take position and confidence from that best-overlapping detection and have "
                           "det_shared_with_other_gt_cheap/full = true",
        "fp_in_both_iou30_note": "omitted_by_system_geometry.*.fp_in_both_iou30 counts unmatched detections present in both "
                                 "modes when the cross-mode pairing uses IoU >= 0.3 instead of 0.5 (recorded only; the "
                                 "exported types use 0.5)",
        "gallery_note": "the gallery JSONs additionally list every GT object missed by both modes and every FP present in "
                        "both modes (pairing IoU >= 0.5), so the BEV inset can show the whole scene; the BEV CSV does not",
        "plan_ade_note": "Planner C has no corridor test; in_corridor on plan_ade rows is the braking controller's test on the "
                         "braking geometry of the same detection, given for reference only.",
        "rows_by_system_geometry_type": {f"{a}|{b}|{c}": int(n) for (a, b, c), n in bev.groupby(["system", "geometry", "type"]).size().items()},
        "omitted_by_system_geometry": omitted,
        "rows_matched_through_shared_detection": shared_rows,
        "verification_rebuild_equals_registered_tables": verify,
    }
    (FINAL / "fig_bev_constants.json").write_text(json.dumps(const, indent=1))

    # ---------------------------------------------------------------- B: gallery
    gal = FINAL / "fig_gallery"
    gal.mkdir(parents=True, exist_ok=True)
    jm = pd.read_pickle(JOINED["mono"])
    jm["V"] = jm.J_cheap - jm.J_full
    jm = jm[(jm.V.abs() > EPS) & ~jm.apply(lambda r: (r.seq, int(r.frame)) in FIGURE1_EXCLUDE, axis=1)]
    picked, used = [], set()
    for sign, order in (("negative", jm.sort_values(["V", "seq", "frame"])),
                        ("positive", jm.sort_values(["V", "seq", "frame"], ascending=[False, True, True]))):
        n = 0
        for _, r in order.iterrows():
            if n == 6 or (sign == "negative" and r.V >= 0) or (sign == "positive" and r.V <= 0):
                break
            if r.seq in used:
                continue
            used.add(r.seq)
            picked.append((sign, n + 1, r))
            n += 1
    index = []
    name = lambda a: P.ACTION_NAMES[int(a)]
    for sign, rank, r in picked:
        fr = frames["mono"][(r.seq, int(r.frame))]
        src = db.image_path(r.sample_token)
        stem = f"{sign}_{rank}_{r.seq}_{int(r.frame):02d}"
        shutil.copy2(src, gal / f"{stem}{src.suffix}")

        def boxes(mode):
            o = fr["out"][mode]
            lat, cons = corridor(o["lo"], o["hi"], o["z"], pp)
            return [{"det_index": k, "xyxy": [float(v) for v in o["xyxy"][k]], "class": str(o["coarse"][k]), "conf": float(o["conf"][k]),
                     "matched_gt_id": int(fr["g"]["track_id"][o["det_gt"][k]]) if o["det_gt"][k] >= 0 else None,
                     "x_lateral_m": float((o["lo"][k] + o["hi"][k]) / 2), "z_forward_m": float(o["z"][k]),
                     "lat_min_m": float(o["lo"][k]), "lat_max_m": float(o["hi"][k]), "in_corridor": bool(lat[k]),
                     "considered_by_controller": bool(cons[k]), "a_req_obj": float(o["a_obj"][k])} for k in range(len(o["conf"]))]
        g = fr["g"]
        lat, cons = corridor(g["lat_min"], g["lat_max"], g["long_near"], pp)
        gts = [{"gt_id": int(g["track_id"][i]), "instance_token": fr["inst"].get(int(g["track_id"][i])), "type": str(g["type"][i]),
                "xyxy_projected": [float(g[c][i]) for c in ("x1", "y1", "x2", "y2")],
                "x_lateral_m": float(g["lat_cen"][i]), "z_forward_m": float(g["long_near"][i]),
                "lat_min_m": float(g["lat_min"][i]), "lat_max_m": float(g["lat_max"][i]), "ttc_s": float(min(g["ttc"][i], 1e3)),
                "in_corridor": bool(lat[i]), "considered_by_controller": bool(cons[i]),
                "matched_by_cheap": bool(fr["out"]["cheap"]["gt_hit"][i]), "matched_by_full": bool(fr["out"]["full"]["gt_hit"][i]),
                "a_req_obj": float(fr["a_gt_obj"][i])} for i in range(len(g))]
        oc_, of_ = fr["out"]["cheap"], fr["out"]["full"]
        missed = [{**gt, "class": gt["type"]} for gt in gts if not gt["matched_by_cheap"] and not gt["matched_by_full"]]
        cb, fb = boxes("cheap"), boxes("full")
        ucg, ufg = np.flatnonzero(oc_["det_gt"] < 0), np.flatnonzero(of_["det_gt"] < 0)
        pcg, _ = pair_fps(oc_["xyxy"][ucg], of_["xyxy"][ufg], cfg.iou_thr)
        fpboth = []
        for a, kc in enumerate(ucg):
            if pcg[a] >= 0:
                c_, f_ = cb[int(kc)], fb[int(ufg[pcg[a]])]
                fpboth.append({"class": c_["class"], "class_full": f_["class"], "conf_cheap": c_["conf"], "conf_full": f_["conf"],
                               "xyxy": c_["xyxy"], "xyxy_full": f_["xyxy"], "det_index_cheap": int(kc),
                               "det_index_full": int(ufg[pcg[a]]), "x_lateral_m": c_["x_lateral_m"],
                               "z_forward_m": c_["z_forward_m"], "in_corridor": c_["in_corridor"],
                               "considered_by_controller": c_["considered_by_controller"],
                               "x_lateral_full_m": f_["x_lateral_m"], "z_forward_full_m": f_["z_forward_m"],
                               "in_corridor_full": f_["in_corridor"], "considered_by_controller_full": f_["considered_by_controller"]})
        rec = {"scene": r.seq, "frame": int(r.frame), "sample_token": r.sample_token, "system": "brake", "geometry": "mono",
               "image_path": str(src), "image_copy": f"{stem}{src.suffix}", "V": float(r.V), "J_cheap": float(r.J_cheap),
               "J_full": float(r.J_full), "ego_speed_mps": fr["v"],
               "action": {"cheap": name(fr["act_c"]), "full": name(fr["act_f"]), "reference": name(fr["act_ref"])},
               "a_req": {"cheap": float(fr["out"]["cheap"]["a"]), "full": float(fr["out"]["full"]["a"]), "reference": float(fr["a_gt"])},
               "cheap_boxes": cb, "full_boxes": fb, "gt_boxes": gts, "missed_by_both": missed, "fp_in_both": fpboth,
               "conventions": "boxes in original image pixels; x_lateral_m positive to the left; z_forward_m = near-face range; "
                              "missed_by_both uses the GT projected box and GT geometry; fp_in_both primary fields are the "
                              "CHEAP detection's (FULL's in *_full), paired at IoU >= 0.5 "
                              "(mono geometry for detections, GT long_near for GT boxes); see fig_bev_constants.json"}
        (gal / f"{stem}.json").write_text(json.dumps(rec, indent=1))
        index.append({**{k: rec[k] for k in ("scene", "frame", "sample_token", "V", "image_copy")},
                      "sign": sign, "rank": rank, "action": rec["action"]})
        print(f"  gallery {stem}: V {r.V:+.3f} actions {rec['action']}", flush=True)
    (gal / "index.json").write_text(json.dumps({"selection": "q_brake mono, 6 most negative and 6 most positive V, excluding "
                                                             "scene-0032 frame 9 and scene-0048 frame 3, at most one frame per scene "
                                                             "(across all 12), ties broken by scene then frame",
                                                "frames": index}, indent=1))

    # ---------------------------------------------------------------- C: budget curves
    b = pd.read_csv(FINAL / "benchmark_budget_routers.csv")
    b = b[b.unit == "ms"].copy()
    keep = ["track", "geometry", "system", "target", "signal", "budget_level", "budget_per_frame", "n_frames", "feasible",
            "overhead", "overhead_source", "escalated_frac", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo",
            "minus_random_hi", "note"]
    cells = b[keep].assign(row_type="cell")
    med = (b.groupby(["track", "signal", "budget_level"], dropna=False)
           .agg(eta=("eta", "median"), escalated_frac=("escalated_frac", "median"), n_cells=("eta", "size"))
           .reset_index().assign(row_type="median_over_cells", geometry="all", system="all", target="all"))
    pd.concat([cells, med], ignore_index=True).to_csv(FINAL / "fig_budget_curves.csv", index=False)
    print(f"  budget curves: {len(cells)} cell rows, {len(med)} median rows")


if __name__ == "__main__":
    main()

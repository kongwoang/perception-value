#!/usr/bin/env python
"""Task 12: statistics hardening -- unnormalised paired intervals, leave-one-unit-out influence, trivial baselines.

Pre-registered in RESEARCH_LOG.md (Task 12), committed before this script was run.  Nothing official is written:

  results/final/statistics_hardening.csv   sections: sanity, paired_gain (a), influence (b), harm (d)

(a) the realised decision value of a signal minus random's, in loss units and as a share of the all-cheap loss,
    with a 1,000-draw paired unit bootstrap keeping every draw, and the same statistic under the official 25%
    prize filter;
(b) at the 20% quota, each test unit dropped in turn, nDG and raw gain recomputed on the remainder;
(c) four trivial baselines scored through the same protocol (rows of section paired_gain with trivial=True);
(d) harmed inputs as a share of affected inputs and of all inputs.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


t120 = _load("t120", "120_nuplan_real_allocation.py")
t92, t103 = t120.t92, t120.t103

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS, QUOTAS = t92.EPS, t92.QUOTAS
NBOOT = 1000
R1 = ["R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
GATES = ["gate_ridge", "gate_gbm"]
R1_RUN = RAW / "20260915_014442_routers_r1"
NR_RUN = RAW / "20260915_033311_nuplan_real_allocation"
_r2 = sorted(glob.glob(str(RAW / "*_router_r2")))
R2_RUN = Path(_r2[-1]) if _r2 else None
TRIVIAL_CORE = {"trivial_ego_speed": "v_ego", "trivial_n_det": "feat_n_det",
                "trivial_risk_cheap": "feat_crit_sum", "trivial_area_max": "feat_area_frac_max"}
TRIVIAL_NUPLAN = {"trivial_ego_speed": "nr_ego_speed", "trivial_n_det": "nr_n_cheap",
                  "trivial_risk_cheap": "nr_crit_cheap_sum"}
NO_AREA = "the nuPlan real-perception features carry no detection area"


def col(d, name):
    return pd.to_numeric(d[name], errors="coerce").fillna(-np.inf).to_numpy(float)


# ------------------------------------------------------------------------------------------------ cells

def core_cells(splits):
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            scores = {s: col(d, c["cols"][s]) for s in c["cols"] if c["cols"][s] in d.columns}
            scores.update(t92.gate_predictions(d, c["fcols"], v))
            z = np.load(R1_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz", allow_pickle=False)
            assert (z["seq"].astype(str) == d.seq.astype(str).to_numpy()).all() and (z["frame"] == d.frame.to_numpy()).all()
            assert np.array_equal(z["V"], v)
            scores.update({r: np.asarray(z[r], float) for r in R1})
            f2 = None if R2_RUN is None else R2_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz"
            if f2 is not None and f2.exists():
                y = np.load(f2, allow_pickle=False)
                r2 = pd.DataFrame({"seq": y["seq"].astype(str), "frame": y["frame"].astype(int),
                                   "s": np.asarray(y["R2_cnn_clf"], float)})
                j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(r2, on=["seq", "frame"], how="left",
                                                                                 validate="one_to_one")
                scores["R2_cnn_clf"] = np.nan_to_num(j.s.to_numpy(float), nan=-np.inf)
            trivial = {k: col(d, v_) for k, v_ in TRIVIAL_CORE.items() if v_ in d.columns}
            missing = {k: f"column {v_} not in this cell's frame table" for k, v_ in TRIVIAL_CORE.items() if v_ not in d.columns}
            yield dict(dataset=c["track"], track=c["track"], geometry=c["geometry"], system=c["system"],
                       target=c["target"], d=d, v_all=v, cheap=c["cheap"], scores=scores, trivial=trivial,
                       trivial_missing=missing, official_table="core")


def nuplan_cells(splits):
    for c in t120.cells(splits):
        d = c["d"].reset_index(drop=True)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(NR_RUN / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        assert (z["scenario"].astype(str) == d.scenario.astype(str).to_numpy()).all()
        assert (z["iteration"] == d.iteration.to_numpy()).all() and np.array_equal(z["V"], v)
        scores = {a: np.asarray(z[a], float) for a in GATES + R1}
        scores["criticality_cheap"] = col(d, "nr_crit_cheap_sum")
        for s, c_ in t120.DIAGNOSTIC.items():
            if c_ in d.columns:
                scores[s] = col(d, c_)
        trivial = {k: col(d, v_) for k, v_ in TRIVIAL_NUPLAN.items() if v_ in d.columns}
        missing = {k: f"column {v_} not in this cell's feature table" for k, v_ in TRIVIAL_NUPLAN.items() if v_ not in d.columns}
        missing["trivial_area_max"] = NO_AREA
        yield dict(dataset="nuPlan", track="nuPlan", geometry="n/a", system=c["system"], target=c["target"], d=d,
                   v_all=v, cheap=c["cheap"], scores=scores, trivial=trivial, trivial_missing=missing,
                   official_table="nuplan_real")


# ------------------------------------------------------------------------------------------------ statistic

def gains(score, v, ks):
    """Expected gain of the top-k selection for every k, with the official tie expectation."""
    return np.asarray(t92.topk_expect(score, [v], ks)[0][0], float)


def official_reference():
    out = {}
    for name in ("benchmark_table.csv", "benchmark_table_routers.csv", "benchmark_table_nuplan_real.csv"):
        x = pd.read_csv(FINAL / name)
        x["geometry"] = x.geometry.fillna("n/a")
        real = name.endswith("nuplan_real.csv")
        x = x[(x.split == "test") & x.quota.notna() & (x.track != "nuPlan" if not real else x.track == "nuPlan")]
        for r in x.itertuples():
            if getattr(r, "available", True) is False:
                continue
            out[(r.track, r.geometry, r.system, r.target, r.signal, round(r.quota, 2))] = r.eta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    args = ap.parse_args()
    run = runmeta.new_run("statistics_hardening", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    t120.register_features()
    official = official_reference()
    rows, cells = [], []

    for gen in (core_cells(splits), nuplan_cells(splits)):
        for c in gen:
            d = c["d"]
            te = (d.split == "test").to_numpy()
            c["test"], c["v"] = te, c["v_all"][te]
            c["units"] = d.unit.astype(str).to_numpy()[te]
            c["uniq"] = np.array(sorted(set(c["units"])))
            c["idx"] = [np.flatnonzero(c["units"] == u) for u in c["uniq"]]
            c["upos"] = {u: i for i, u in enumerate(c["uniq"])}
            c["n"] = int(te.sum())
            c["tot_cheap"] = float(d[c["cheap"]].to_numpy(float)[te].sum())
            c["ks"] = [max(int(round(q * c["n"])), 1) for q in QUOTAS]
            c["prize"] = gains(c["v"], c["v"], c["ks"])
            c["n_aff"] = int((np.abs(c["v"]) > EPS).sum())
            c["undef"] = (f"near-zero oracle: {c['n_aff']} affected states on this split (< {t120.MIN_AFFECTED})"
                          if c["track"] == "nuPlan" and c["n_aff"] < t120.MIN_AFFECTED else "")
            c["series"] = {s: np.nan_to_num(np.asarray(x, float)[te], nan=-np.inf) for s, x in c["scores"].items()}
            c["series"].update({s: np.nan_to_num(np.asarray(x, float)[te], nan=-np.inf) for s, x in c["trivial"].items()})
            c["series"]["oracle"] = c["v"]
            cells.append(c)
            print(f"  cell {len(cells):2d} {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:8s} "
                  f"test {c['n']:5d} frames, {len(c['uniq'])} units, {len(c['series'])} signals", flush=True)

    key = ("track", "geometry", "system", "target")

    # ---- (d) harm shares, and the sanity check on the official nDG
    for c in cells:
        for split, m in (("test", c["test"]), ("all", np.ones(len(c["v_all"]), bool))):
            v = c["v_all"][m]
            aff = int((np.abs(v) > EPS).sum())
            harmed = int((v < -EPS).sum())
            rows.append({"section": "harm", **dict(zip(key, (c["track"], c["geometry"], c["system"], c["target"]))),
                         "split": split, "n_inputs": len(v), "n_affected": aff, "n_harmed": harmed,
                         "harmed_over_affected": harmed / aff if aff else np.nan,
                         "harmed_over_all_inputs": harmed / len(v)})
    n_ok = n_cmp = 0
    for c in cells:
        for s, sc in c["series"].items():
            g = gains(sc, c["v"], c["ks"])
            for qi, q in enumerate(QUOTAS):
                ref = official.get((c["track"], c["geometry"], c["system"], c["target"], s, round(q, 2)))
                if ref is None or c["undef"]:
                    continue
                mine = g[qi] / c["prize"][qi] if c["prize"][qi] > EPS else np.nan
                n_cmp += 1
                ok = (pd.isna(ref) and pd.isna(mine)) or (not pd.isna(ref) and not pd.isna(mine)
                                                          and round(float(mine), 3) == round(float(ref), 3))
                n_ok += ok
                rows.append({"section": "sanity", **dict(zip(key, (c["track"], c["geometry"], c["system"], c["target"]))),
                             "signal": s, "quota": q, "ndg_recomputed": mine, "ndg_released": ref, "matches_3dp": bool(ok)})
    print(f"  sanity: official nDG reproduced in {n_ok}/{n_cmp} comparisons", flush=True)

    # ---- (a) point estimates
    point = {}
    for ci, c in enumerate(cells):
        rnd = np.array([k / c["n"] * c["v"].sum() for k in c["ks"]])
        for s, sc in c["series"].items():
            g = gains(sc, c["v"], c["ks"])
            point[(ci, s)] = g
            for qi, q in enumerate(QUOTAS):
                delta = float(g[qi] - rnd[qi])
                rows.append({"section": "paired_gain", **dict(zip(key, (c["track"], c["geometry"], c["system"], c["target"]))),
                             "signal": s, "trivial": s.startswith("trivial_"), "quota": q, "n_test_frames": c["n"],
                             "n_test_units": len(c["uniq"]), "all_cheap_loss": c["tot_cheap"],
                             "gain": float(g[qi]), "gain_random": float(rnd[qi]), "delta_vs_random": delta,
                             "delta_share_all_cheap": delta / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan,
                             "ndg": (g[qi] / c["prize"][qi]) if (c["prize"][qi] > EPS and not c["undef"]) else np.nan,
                             "oracle_prize": float(c["prize"][qi]), "undefined_reason": c["undef"]})
        for k, why in c["trivial_missing"].items():
            rows.append({"section": "paired_gain", **dict(zip(key, (c["track"], c["geometry"], c["system"], c["target"]))),
                         "signal": k, "trivial": True, "available": False, "undefined_reason": why})

    # ---- (a) bootstrap: every draw kept, and the 25% prize filter as a second column
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units_of = {ds: sorted(set().union(*[set(c["uniq"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    draws = {(ci, s): np.full((args.nboot, len(QUOTAS)), np.nan) for ci, c in enumerate(cells) for s in c["series"]}
    keep = {ci: np.zeros((args.nboot, len(QUOTAS)), bool) for ci in range(len(cells))}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for b in range(args.nboot):
        pick = {ds: rng.integers(0, len(units_of[ds]), len(units_of[ds])) for ds in datasets}
        for ci, c in enumerate(cells):
            us = [u for u in (units_of[c["dataset"]][i] for i in pick[c["dataset"]]) if u in c["upos"]]
            if not us:
                continue
            t = np.concatenate([c["idx"][c["upos"][u]] for u in us])
            vt = c["v"][t]
            ks = [max(int(round(q * len(t))), 1) for q in QUOTAS]
            pz = gains(vt, vt, ks)
            keep[ci][b] = pz > np.maximum(EPS, 0.25 * c["prize"])
            rnd_t = np.array([k / len(t) * vt.sum() for k in ks])
            for s, sc in c["series"].items():
                draws[(ci, s)][b] = gains(sc[t], vt, ks) - rnd_t
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.nboot} [{time.time() - t0:.0f}s]", flush=True)

    idx = {(r["track"], r["geometry"], r["system"], r["target"], r["signal"], round(r["quota"], 2)): i
           for i, r in enumerate(rows) if r["section"] == "paired_gain" and r.get("available", True) and "quota" in r}
    for ci, c in enumerate(cells):
        for s in c["series"]:
            x = draws[(ci, s)]
            for qi, q in enumerate(QUOTAS):
                i = idx.get((c["track"], c["geometry"], c["system"], c["target"], s, round(q, 2)))
                if i is None:
                    continue
                col_all = x[:, qi]
                lo, hi = t92.ci(col_all)
                sub = col_all[keep[ci][:, qi]]
                flo, fhi = t92.ci(sub)
                rows[i].update({"delta_lo_all_draws": lo, "delta_hi_all_draws": hi,
                                "delta_share_lo_all_draws": lo / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan,
                                "delta_share_hi_all_draws": hi / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan,
                                "delta_lo_filtered": flo, "delta_hi_filtered": fhi,
                                "draws_dropped_by_filter": int(args.nboot - len(sub)),
                                "beats_random_all_draws": bool(lo > 0), "beats_random_filtered": bool(flo > 0)})

    # ---- (b) leave-one-test-unit-out at the 20% quota
    qi20 = QUOTAS.index(0.20)
    for ci, c in enumerate(cells):
        for s, sc in c["series"].items():
            full_ndg = (point[(ci, s)][qi20] / c["prize"][qi20]) if (c["prize"][qi20] > EPS and not c["undef"]) else np.nan
            full_gain = float(point[(ci, s)][qi20])
            out = []
            for ui, u in enumerate(c["uniq"]):
                m = np.ones(c["n"], bool)
                m[c["idx"][ui]] = False
                vt = c["v"][m]
                k = max(int(round(0.20 * m.sum())), 1)
                pz = float(gains(vt, vt, [k])[0])
                g = float(gains(sc[m], vt, [k])[0])
                out.append((u, g, (g / pz) if (pz > EPS and not c["undef"]) else np.nan))
            g_arr = np.array([o[1] for o in out])
            n_arr = np.array([o[2] for o in out], float)
            if np.isfinite(n_arr).any():
                j = int(np.nanargmax(np.abs(n_arr - full_ndg)))
            else:
                j = int(np.argmax(np.abs(g_arr - full_gain)))
            rows.append({"section": "influence", **dict(zip(key, (c["track"], c["geometry"], c["system"], c["target"]))),
                         "signal": s, "trivial": s.startswith("trivial_"), "quota": 0.20, "n_test_units": len(c["uniq"]),
                         "ndg_full": full_ndg, "ndg_min_leave_one_out": float(np.nanmin(n_arr)) if np.isfinite(n_arr).any() else np.nan,
                         "ndg_max_leave_one_out": float(np.nanmax(n_arr)) if np.isfinite(n_arr).any() else np.nan,
                         "gain_full": full_gain, "gain_min_leave_one_out": float(g_arr.min()),
                         "gain_max_leave_one_out": float(g_arr.max()),
                         "most_influential_unit": out[j][0], "ndg_without_that_unit": out[j][2],
                         "gain_without_that_unit": out[j][1],
                         "ndg_change_without_that_unit": (out[j][2] - full_ndg) if np.isfinite(full_ndg) else np.nan,
                         "undefined_reason": c["undef"]})
        print(f"  influence {c['track']:8s} {c['system']:10s} {c['target']:8s}", flush=True)

    df = pd.DataFrame(rows)
    for out in (FINAL / "statistics_hardening.csv", run / "statistics_hardening.csv"):
        df.to_csv(out, index=False)
    print(f"  wrote {FINAL / 'statistics_hardening.csv'} ({len(df)} rows)")
    bad = df[(df.section == "sanity") & (df.matches_3dp == False)]                      # noqa: E712
    print(f"  sanity rows failing the 3-decimal match: {len(bad)}")
    if len(bad):
        print(bad.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

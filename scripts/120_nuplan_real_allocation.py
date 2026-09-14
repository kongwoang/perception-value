#!/usr/bin/env python
"""Task 7: the nuPlan allocation track on real-perception decision values -- frame quotas and measured budgets.

Pre-registered in RESEARCH_LOG.md (Task 7 pre-registration).  Labels are Task 5's primary branches; every
pre-escalation input comes from the real CHEAP branch (119).  The statistics are the benchmark's own code:
92's `evaluate` (exact tie expectation, 1,000 log-level draws, paired against random), 92's
`gate_predictions`, 103's `fit_score` and 93's two-level cascade with `signal_overhead`.  Nothing existing is
overwritten:

  results/final/benchmark_table_nuplan_real.csv   frame quotas 10/20/30/50%, test (all signals) and all
                                                  units (non-learned signals)
  results/final/benchmark_budget_nuplan_real.csv  ms and mJ budgets with nuPlan's own Task 5 detector costs

nDG is undefined (NaN, with the reason) when the split's oracle prize at a quota is <= 1e-9 or the split has
fewer than 10 affected states.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import features as F, runmeta                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


t92 = _load("t92", "92_benchmark_table.py")
t93 = _load("t93", "93_budget_allocation.py")
t103 = _load("t103", "103_routers_r1.py")

FINAL = Path(RESULTS) / "final"
QUOTAS, EPS = t92.QUOTAS, t92.EPS
MIN_AFFECTED = 10
GATE = ["ego_speed", "ego_accel", "n_cheap", "n_cheap_fov", "n_cheap_fov_bicycle", "n_cheap_fov_pedestrian",
        "n_cheap_fov_vehicle", "n_static_fov", "min_range_fov", "min_gap_corridor", "n_corridor_20", "n_corridor_40",
        "min_ttc_corridor", "n_fov_band_30_40", "crit_cheap_sum", "crit_cheap_max", "n_red_lights", "d_n_cheap_fov"]
SOURCE = {c: ("ego_state" if c in ("ego_speed", "ego_accel", "n_red_lights") else
              "cheap_prev" if c == "d_n_cheap_fov" else "cheap_det") for c in GATE}
DIAGNOSTIC = {"criticality_gt": "crit_sum_gt", "dE_E1_fn_only": "dE_E1_fn_only",
              "dE_E6_risk_weighted": "dE_E6_risk_weighted"}
R1 = ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf")
SIGNALS = ["random", "criticality_cheap", "gate_ridge", "gate_gbm", *R1, *DIAGNOSTIC, "oracle"]
LEARNED = {"gate_ridge", "gate_gbm", *R1}
DEPLOYABLE = {"random": True, "criticality_cheap": True, "gate_ridge": True, "gate_gbm": True, **{r: True for r in R1}}
NOTE = {"pdm_closed": "PDM-Closed, published (tuPlan Garage); real YOLOv8s perception (Task 5 primary)",
        "idm": "IDMPlanner, published (nuPlan devkit); real YOLOv8s perception (Task 5 primary)"}


def register_features():
    cols = [f"nr_{c}" for c in GATE]
    for c in GATE:
        F._reg(f"nr_{c}", "nuplan_real", SOURCE[c])
    F.assert_no_leakage(cols)
    refused = {}
    for c in ["crit_sum_gt", "dE_E1_fn_only", "dE_E6_risk_weighted", "unc_proxy"]:
        try:
            F.assert_no_leakage([c])
            refused[c] = False
        except ValueError:
            refused[c] = True
    assert all(refused.values()), f"a diagnostic or privileged column passed the leakage guard: {refused}"
    assert not [c for c in GATE if any(w in c for w in ("full", "reference", "_gt", "n_tracks"))]
    return cols, refused


def cells(splits):
    run = sorted(glob.glob(str(ROOT / "results" / "raw" / "*_nuplan_real_features")))[-1]
    sig = pd.read_csv(Path(run) / "nuplan_real_signals.csv")
    sig = sig.rename(columns={c: f"nr_{c}" for c in GATE})
    z = np.load(Path(run) / "nuplan_real_track_features.npz", allow_pickle=False)
    trk = pd.DataFrame({"scenario": z["scenario"].astype(str), "iteration": z["iteration"].astype(int),
                        "_row": np.arange(len(z["scenario"]))})
    split_of = {u: k for k in ("train", "val", "test") for u in splits["nuplan"][k]}
    for planner in ("pdm_closed", "idm"):
        r = pd.read_csv(FINAL / f"nuplan_real_perception_{planner}_raw.csv")
        assert (r[["ok_cheap", "ok_full"]] == 1).all().all()
        d = r.merge(sig, on=["scenario", "log", "iteration"], how="inner", validate="one_to_one")
        assert len(d) == 1440, len(d)
        j = d[["scenario", "iteration"]].merge(trk, on=["scenario", "iteration"], how="left", validate="one_to_one")
        assert j._row.notna().all()
        X = z["X"][j._row.to_numpy(int)]
        cc, cf = t92.x83.costs(d, "cheap"), t92.x83.costs(d, "full")
        d["unit"], d["split"] = d.log, d.log.map(split_of)
        assert d.split.notna().all()
        for target in ("safety", "scalar_J"):
            dd = d.copy()
            dd[f"{target}_cheap"], dd[f"{target}_full"] = cc[target], cf[target]
            yield dict(track="nuPlan", geometry="n/a", system=planner, target=target, d=dd, X=X, run=run,
                       cheap=f"{target}_cheap", full=f"{target}_full")


def undefined(prize, n_aff):
    if n_aff < MIN_AFFECTED:
        return f"near-zero oracle: {n_aff} affected states on this split (< {MIN_AFFECTED})"
    if prize <= EPS:
        return "oracle prize is zero at this quota"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()
    run = runmeta.new_run("nuplan_real_allocation", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    fcols, refused = register_features()
    print(f"  {len(fcols)} gate features pass assert_no_leakage; refused as expected: {refused}", flush=True)
    rng = np.random.default_rng(0)
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    det = json.loads((ROOT / "results" / "raw" / "nuplan_task5" / "detect_summary.json").read_text())
    cost = {"cheap": {"ms": det["ns_cheap_320"]["ms_total_median"], "mJ": det["energy"]["ns_cheap_320"]["cpu_gpu_mj_per_frame"]},
            "640": {"ms": det["ns_full_640"]["ms_total_median"], "mJ": det["energy"]["ns_full_640"]["cpu_gpu_mj_per_frame"]}}
    print(f"  nuPlan detector costs (Task 5): {cost}", flush=True)

    rows, brows, stash = [], [], []
    for c in cells(splits):
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        fit = d.split.isin(["train", "val"]).to_numpy()
        gates = t92.gate_predictions(d, fcols, v_all)
        r1 = t103.fit_score(c["X"], v_all, fit)
        key = dict(track="nuPlan", geometry="n/a", system=c["system"], target=c["target"], system_note=NOTE[c["system"]],
                   labels="Task 5 real perception, primary", features_run=Path(c["run"]).name)
        np.savez_compressed(run / f"scores__{c['system']}__{c['target']}.npz", scenario=d.scenario.to_numpy().astype("U32"),
                            iteration=d.iteration.to_numpy(), unit=d.unit.to_numpy().astype("U40"),
                            split=d.split.to_numpy().astype("U8"), V=v_all, **gates, **r1)
        for split in ("test", "all"):
            m = (d.split == "test").to_numpy() if split == "test" else np.ones(len(d), bool)
            v, units = v_all[m], d.unit.to_numpy()[m]
            desc = t92.describe(v, d[c["cheap"]].to_numpy(float)[m])
            n_aff, n_pos, n_neg = int((np.abs(v) > EPS).sum()), int((v > EPS).sum()), int((v < -EPS).sum())
            scores, avail = {}, {}
            for s in SIGNALS:
                if s == "random":
                    scores[s] = None
                elif s == "oracle":
                    scores[s] = v
                elif s in LEARNED and split != "test":
                    avail[s] = "learned signals are scored on the test split only"
                elif s in ("gate_ridge", "gate_gbm"):
                    scores[s] = gates[s][m]
                elif s in R1:
                    scores[s] = r1[s][m]
                elif s == "criticality_cheap":
                    scores[s] = d["nr_crit_cheap_sum"].to_numpy(float)[m]
                else:
                    scores[s] = d[DIAGNOSTIC[s]].to_numpy(float)[m]
            ks, prize0, point, draws, dropped = t92.evaluate(v, units, scores, args.nboot, rng)
            for s in SIGNALS:
                base = {**key, "split": split, "signal": s, "deployable": DEPLOYABLE.get(s, False), "privileged": False,
                        "input_dim": (c["X"].shape[1] if s in R1 else len(fcols) if s.startswith("gate_") else np.nan),
                        "n_units": int(len(np.unique(units))), "n_frames": int(len(v)), "n_affected": n_aff,
                        "n_v_pos": n_pos, "n_v_neg": n_neg, "cell_harmed_among_affected": desc["harmed_among_affected"],
                        "cell_destroyed_benefit_D": desc["destroyed_benefit_D"],
                        "cell_all_full_reduction": desc["all_full_reduction"],
                        "cell_oracle20_reduction": desc["oracle20_reduction"]}
                if s not in scores:
                    rows.append({**base, "available": False, "na_reason": avail[s]})
                    continue
                p = point[s]
                for qi, q in enumerate(QUOTAS):
                    why = undefined(prize0[qi], n_aff)
                    dr = draws[s][:, qi] - draws["random"][:, qi]
                    lo, hi = t92.ci(draws[s][:, qi])
                    dlo, dhi = t92.ci(dr)
                    ok = not why
                    rows.append({**base, "available": True, "quota": q, "k": int(ks[qi]), "ndg_defined": ok,
                                 "undefined_reason": why, "eta": float(p["eta"][qi]) if ok else np.nan,
                                 "eta_lo": lo if ok else np.nan, "eta_hi": hi if ok else np.nan,
                                 "minus_random": float(np.nanmean(dr)) if ok else np.nan,
                                 "minus_random_lo": dlo if ok else np.nan, "minus_random_hi": dhi if ok else np.nan,
                                 "p_le_random": (float(np.mean(dr[np.isfinite(dr)] <= 0)) if (ok and s != "random") else np.nan),
                                 "gain": float(p["gain"][qi]), "prize": float(prize0[qi]),
                                 "reduction_frac": float(p["gain"][qi] / max(d[c["cheap"]].to_numpy(float)[m].sum(), EPS)),
                                 "tie_frac": float(p["tie"][qi]), "responsive_frac": float(p["resp"][qi]),
                                 "boot_dropped": int(dropped[qi])})
            if split == "test":
                stash.append((c, v, units, n_aff, {k: scores[k] for k in scores if k not in DIAGNOSTIC}))
        r20 = [x for x in rows[-(len(SIGNALS) * 4 + 10):] if x.get("quota") == 0.2 and x["split"] == "test"]
        print(f"  {c['system']:10s} {c['target']:8s} test affected {stash[-1][3]:3d}  @20 " +
              "  ".join(f"{x['signal']}={x['eta']:+.2f}" for x in r20 if x["signal"] in
                        ("criticality_cheap", "gate_ridge", "gate_gbm", "R1_mlp_clf", "R1_gbm_reg")), flush=True)

    # ------------------------------------------------------------------ budget track (93's two-level protocol)
    brng = np.random.default_rng(0)
    for c, v, units, n_aff, scores in stash:
        scores = dict(scores)
        scores["gate_gbm_batched"] = scores["gate_gbm"]
        uniq = np.unique(units)
        idx = [np.flatnonzero(units == u) for u in uniq]
        draws = [np.concatenate([idx[i] for i in brng.integers(0, len(uniq), len(uniq))]) for _ in range(args.nboot)]
        key = dict(track="nuPlan", geometry="n/a", system=c["system"], target=c["target"], labels="Task 5 real perception, primary",
                   cost_source="Task 5 detect_summary.json: pre+inference+postprocess median ms (no decode); CPU+GPU mJ over idle")
        for unit in ("ms", "mJ"):
            c0, c1 = cost["cheap"][unit], cost["640"][unit]
            for f in QUOTAS:
                budget = c0 + f * c1

                def gain_at(sc, frac, t=None):
                    vv = v if t is None else v[t]
                    k = int(np.floor(frac * len(vv) + 1e-9))
                    if k <= 0:
                        return 0.0
                    if sc is None:
                        return k / len(vv) * vv.sum()
                    return float(t92.topk_expect(sc if t is None else sc[t], [vv], [k])[0][0][0])

                prize = gain_at(v, f)
                prize_t = np.array([gain_at(v, f, t) for t in draws])
                okb = prize_t > max(EPS, 0.25 * prize)
                rand_t = np.array([gain_at(None, f, t) for t in draws]) / np.where(okb, prize_t, 1)
                why = undefined(prize, n_aff)
                for s in list(scores) + list(DIAGNOSTIC):
                    base = {**key, "unit": unit, "budget_level": f, "budget_per_frame": budget, "cheap_cost": c0,
                            "full_cost": c1, "signal": s, "n_frames": len(v), "n_affected": n_aff}
                    if s in DIAGNOSTIC:
                        brows.append({**base, "feasible": False,
                                      "note": f"needs FULL on every frame: >= {c0 + c1:.2f} {unit} per frame before computing the metric"})
                        continue
                    o_ms, o_mj, src = t93.signal_overhead(ov, "nuPlan", s)
                    o = o_ms if unit == "ms" else o_mj
                    frac = max((budget - c0 - o) / c1, 0.0)
                    g = gain_at(scores[s], frac)
                    e_t = np.array([gain_at(scores[s], frac, t) for t in draws]) / np.where(okb, prize_t, 1)
                    e_t, dr = np.where(okb, e_t, np.nan), np.where(okb, e_t - rand_t, np.nan)
                    lo, hi = t92.ci(e_t)
                    dlo, dhi = t92.ci(dr)
                    ok = not why
                    brows.append({**base, "feasible": True, "overhead": o, "overhead_source": src, "escalated_frac": frac,
                                  "escalations_lost_to_overhead": f - frac, "ndg_defined": ok, "undefined_reason": why,
                                  "gain": g, "prize": prize,
                                  "eta": (g / prize if prize > EPS else np.nan) if ok else np.nan,
                                  "eta_lo": lo if ok else np.nan, "eta_hi": hi if ok else np.nan,
                                  "minus_random": float(np.nanmean(dr)) if ok else np.nan,
                                  "minus_random_lo": dlo if ok else np.nan, "minus_random_hi": dhi if ok else np.nan,
                                  "boot_dropped": int((~okb).sum())})
        print(f"  budget {c['system']} {c['target']}", flush=True)

    tab, bud = pd.DataFrame(rows), pd.DataFrame(brows)
    for name, df in (("benchmark_table_nuplan_real", tab), ("benchmark_budget_nuplan_real", bud)):
        df.to_csv(FINAL / f"{name}.csv", index=False)
        df.to_csv(run / f"{name}.csv", index=False)
        print(f"  wrote {FINAL / (name + '.csv')} ({len(df)} rows)")
    (run / "meta.json").write_text(json.dumps({"costs": cost, "leakage_guard_refused": refused, "min_affected": MIN_AFFECTED,
                                               "gate_feature_sources": SOURCE}, indent=1))


if __name__ == "__main__":
    main()

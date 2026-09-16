#!/usr/bin/env python
"""Task 11: causal streaming allocation with a threshold frozen before the test stream.

Pre-registered in RESEARCH_LOG.md (Task 11), committed before this script was run.  The benchmark scores an
allocator as a ranking that fills the budget over the whole test split; here the same cached scores are applied
causally: one threshold, calibrated before the test stream, applied in timestamp order, with and without a running
budget cap.  Nothing official is written:

  results/final/causal_threshold.csv   per cell, signal, rate, policy and variant, plus the pooled rows

Policies: A = frozen threshold; B = A plus a causal running cap; C = the official top-k (hindsight reference).
Variants: V1 = models refit on train only, threshold calibrated on validation; V2 = official models, threshold
cross-fitted inside train + val; TEST = threshold calibrated on the test scores (the registered sanity check).
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta                                                # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


t120 = _load("t120", "120_nuplan_real_allocation.py")
t92, t93, t103 = t120.t92, t120.t93, t120.t103

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS = t92.EPS
RATES = (0.10, 0.20, 0.30, 0.50)
BUDGET_LEVELS = (0.20, 0.50)
GATES = ["gate_ridge", "gate_gbm"]
R1 = ["R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
LEARNED = GATES + R1
PLAIN = ["uncertainty", "criticality_cheap"]
SIGNALS = ["random"] + PLAIN + LEARNED + ["R2_cnn_clf"]
POOLED_RATE, NBOOT, NFOLD, NRAND, MIN_AFFECTED = 0.20, 1000, 5, 32, t120.MIN_AFFECTED
R1_RUN = RAW / "20260915_014442_routers_r1"
NR_RUN = RAW / "20260915_033311_nuplan_real_allocation"
_r2 = sorted(glob.glob(str(RAW / "*_router_r2")))
R2_RUN = Path(_r2[-1]) if _r2 else None
R2_REASON = "R2's cached scores cover the test frames only; calibrating a threshold would need the image model re-run"
UNC_REASON = "no uncertainty signal exists for the real-perception nuPlan cells (unc_proxy is not among 119's signals)"


def seed_of(*parts):
    return abs(hash(tuple(map(str, parts)))) % (2 ** 32)


# ------------------------------------------------------------------------------------------------ models

def gate_scores(d, fcols, v, fit):
    """92's gate_predictions with an explicit fitting mask."""
    X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                      nan=0.0, posinf=1e6, neginf=-1e6)
    out = {}
    for name, mdl in (("gate_ridge", "linear"), ("gate_gbm", "gbm")):
        m = predict.make_model(mdl, "reg", 0)
        m.fit(X[fit], v[fit])
        out[name] = m.predict(X)
    return out


def learned_scores(c, fit):
    return {**gate_scores(c["d"], c["fcols"], c["v_all"], fit), **t103.fit_score(c["X"], c["v_all"], fit)}


# ------------------------------------------------------------------------------------------------ cells

def core_cells(splits):
    mats = {"nuScenes": t103.frame_matrix(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "nuScenes"),
            "KITTI": t103.frame_matrix(Path(CACHE) / "det", "cheap_320", "KITTI")}
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            keys, M = mats[c["track"]]
            j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(
                keys.assign(_row=np.arange(len(keys))), on=["seq", "frame"], how="left", validate="one_to_one")
            assert j._row.notna().all()
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            z = np.load(R1_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz", allow_pickle=False)
            assert (z["seq"].astype(str) == d.seq.astype(str).to_numpy()).all() and (z["frame"] == d.frame.to_numpy()).all()
            assert np.array_equal(z["V"], v)
            official = {r: np.asarray(z[r], float) for r in R1}
            for s in PLAIN:
                official[s] = pd.to_numeric(d[c["cols"][s]], errors="coerce").fillna(-np.inf).to_numpy(float)
            if R2_RUN is not None and (R2_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz").exists():
                y = np.load(R2_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz", allow_pickle=False)
                r2 = pd.DataFrame({"seq": y["seq"].astype(str), "frame": y["frame"].astype(int), "s": np.asarray(y["R2_cnn_clf"], float)})
                m = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(r2, on=["seq", "frame"], how="left",
                                                                                 validate="one_to_one")
                official["R2_cnn_clf"] = m.s.to_numpy(float)
            order = np.lexsort((d.frame.to_numpy(int), d.seq.astype(str).to_numpy()))
            yield dict(dataset=c["track"], track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                       d=d, v_all=v, X=M[j._row.to_numpy(int)], fcols=c["fcols"], cheap=c["cheap"], official=official,
                       order=order, cost=t93.profile_costs(c["track"]))


def nuplan_cells(splits, fcols):
    meta = pd.read_csv(ROOT / "configs" / "benchmark_nuplan_scenarios.csv")[["scenario", "t0"]]
    for c in t120.cells(splits):
        d = c["d"].reset_index(drop=True)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(NR_RUN / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        assert (z["scenario"].astype(str) == d.scenario.astype(str).to_numpy()).all()
        assert (z["iteration"] == d.iteration.to_numpy()).all() and np.array_equal(z["V"], v)
        official = {a: np.asarray(z[a], float) for a in LEARNED}
        official["criticality_cheap"] = d["nr_crit_cheap_sum"].to_numpy(float)
        t0 = d[["scenario"]].merge(meta, on="scenario", how="left", validate="many_to_one").t0.to_numpy(float)
        assert np.isfinite(t0).all(), "a scenario has no t0 in configs/benchmark_nuplan_scenarios.csv"
        order = np.lexsort((d.iteration.to_numpy(int), t0))
        yield dict(dataset="nuPlan", track="nuPlan", geometry="n/a", system=c["system"], target=c["target"], d=d, v_all=v,
                   X=c["X"], fcols=fcols, cheap=c["cheap"], official=official, order=order,
                   cost=t93.profile_costs("nuPlan"))


# ------------------------------------------------------------------------------------------------ policies

def calibrate(scores, k):
    """(tau, p) with P(s > tau) + p P(s = tau) = k, the tie convention of topk_expect."""
    s = np.round(np.asarray(scores, float), 9)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0 or k <= 0:
        return np.inf, 0.0
    if k >= 1:
        return -np.inf, 1.0
    kk = k * n
    tau = float(np.sort(s)[::-1][int(np.ceil(kk)) - 1])
    n_gt, n_eq = int((s > tau).sum()), int((s == tau).sum())
    return tau, float(np.clip((kk - n_gt) / max(n_eq, 1), 0.0, 1.0))


def mask_A(scores, tau, p, rng):
    s = np.round(np.nan_to_num(np.asarray(scores, float), nan=-np.inf), 9)
    m = s > tau
    tie = s == tau
    if p > 0 and tie.any():
        m = m | (tie & (rng.random(len(s)) < p))
    return m


def apply_cap(mask, k, unit_id, stream):
    """Policy B: at most floor(1 + k t) escalations after t inputs of the unit, in stream order."""
    out = np.zeros(len(mask), bool)
    t = {}
    c = {}
    for i in stream:
        u = unit_id[i]
        ti = t.get(u, 0) + 1
        ci = c.get(u, 0)
        if mask[i] and ci < int(np.floor(1 + k * ti)):
            out[i] = True
            ci += 1
        t[u], c[u] = ti, ci
    return out


def per_unit(v, mask, idx_list):
    """(per-unit gain, per-unit rate) with unit index arrays prepared once."""
    g = np.array([v[i][mask[i]].sum() for i in idx_list])
    r = np.array([mask[i].mean() for i in idx_list])
    return g, r


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--max_cells", type=int, default=0, help="debug only: stop after this many cells")
    args = ap.parse_args()
    run = runmeta.new_run("causal_threshold", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t120.register_features()
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    rows, cells, refit_check = [], [], []

    # ---- cells, scores of both calibration variants
    for gen in (core_cells(splits), nuplan_cells(splits, nr_fcols)):
        for c in gen:
            t_start = time.time()
            d = c["d"]
            split = d.split.to_numpy()
            tr, va, te = split == "train", split == "val", split == "test"
            fitset = tr | va
            n = int(te.sum())
            pos = np.full(len(d), -1)
            pos[te] = np.arange(n)
            c.update(test=te, n=n, v=c["v_all"][te], units=d.unit.astype(str).to_numpy(),
                     tot_cheap=float(d[c["cheap"]].to_numpy(float)[te].sum()))
            c["units_test"] = c["units"][te]
            c["uniq"] = np.array(sorted(set(c["units_test"])))
            c["idx_list"] = [np.flatnonzero(c["units_test"] == u) for u in c["uniq"]]
            c["upos"] = {u: i for i, u in enumerate(c["uniq"])}
            c["stream"] = np.array([pos[i] for i in c["order"] if te[i]], int)
            c["n_aff"] = int((np.abs(c["v"]) > EPS).sum())
            c["unit_id_test"] = c["units_test"]

            v1 = learned_scores(c, tr)
            full = learned_scores(c, fitset)
            for a in LEARNED:
                if a in c["official"]:
                    refit_check.append({"track": c["track"], "geometry": c["geometry"], "system": c["system"],
                                        "target": c["target"], "signal": a,
                                        "max_abs_diff_vs_official": float(np.max(np.abs(full[a] - c["official"][a])))})
                else:
                    c["official"][a] = full[a]
            oof = {a: np.full(len(d), np.nan) for a in LEARNED}
            fold = {u: i % NFOLD for i, u in enumerate(sorted(set(c["units"][fitset])))}
            fold_of = np.array([fold.get(u, -1) for u in c["units"]])
            for f in range(NFOLD):
                hold = fitset & (fold_of == f)
                if not hold.any():
                    continue
                sc = learned_scores(c, fitset & ~hold)
                for a in LEARNED:
                    oof[a][hold] = sc[a][hold]
            c["scores"] = {}
            for s in SIGNALS:
                if s == "random" or s not in c["official"]:
                    continue
                off = np.asarray(c["official"][s], float)
                if s in LEARNED:
                    c["scores"][("V1", s)] = (v1[s][va], v1[s][te])
                    c["scores"][("V2", s)] = (oof[s][fitset], off[te])
                elif np.isfinite(off[fitset]).any():
                    c["scores"][("V1", s)] = (off[va], off[te])
                    c["scores"][("V2", s)] = (off[fitset], off[te])
                c["scores"][("TEST", s)] = (off[te], off[te])
            cells.append(c)
            print(f"  cell {len(cells):2d} {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']:8s} "
                  f"test {n:5d} frames, {len(c['uniq'])} units, affected {c['n_aff']} [{time.time() - t_start:.0f}s]", flush=True)
            if args.max_cells and len(cells) >= args.max_cells:
                break
        if args.max_cells and len(cells) >= args.max_cells:
            break

    # ---- point estimates; per-unit gains for the bootstrap
    series, row_of = {}, {}

    def add(ci, key, value, row_index):
        series[(ci, key)] = value
        row_of[(ci, key)] = row_index

    for ci, c in enumerate(cells):
        v, n, uniq = c["v"], c["n"], c["uniq"]
        undef_cell = (f"near-zero oracle: {c['n_aff']} affected states on this split (< {MIN_AFFECTED})"
                      if c["track"] == "nuPlan" and c["n_aff"] < MIN_AFFECTED else "")
        for k in RATES:
            kn = max(int(round(k * n)), 1)
            prize = float(t92.topk_expect(v, [v], [kn])[0][0][0])
            why = undef_cell or ("oracle prize is zero at this quota" if prize <= EPS else "")
            base = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                        n_test_frames=n, n_test_units=len(uniq), n_affected=c["n_aff"], rate_target=k,
                        oracle_prize=prize, all_cheap_loss=c["tot_cheap"], ndg_defined=not why, undefined_reason=why)

            def emit(variant, signal, policy, mask=None, gain=None, rate=None, rmin=None, rmax=None, tau=np.nan, p=np.nan):
                if mask is not None:
                    g_u, r_u = per_unit(v, mask, c["idx_list"])
                    gain, rate, rmin, rmax = float(g_u.sum()), float(mask.mean()), float(r_u.min()), float(r_u.max())
                rows.append({**base, "variant": variant, "signal": signal, "policy": policy, "available": True,
                             "rate_realised": rate, "rate_unit_min": rmin, "rate_unit_max": rmax, "gain": gain,
                             "gain_share_all_cheap": gain / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan,
                             "ndg": (gain / prize) if (prize > EPS and not why) else np.nan, "tau": tau, "tie_p": p})
                return len(rows) - 1, (g_u if mask is not None else None)

            for pol in ("A", "B"):
                gs, rs, mins, maxs = [], [], [], []
                for r in range(NRAND):
                    m = np.random.default_rng(1000 + r).random(n) < k
                    if pol == "B":
                        m = apply_cap(m, k, c["unit_id_test"], c["stream"])
                    g_u, r_u = per_unit(v, m, c["idx_list"])
                    gs.append(g_u); rs.append(m.mean()); mins.append(r_u.min()); maxs.append(r_u.max())
                g_mean = np.mean(gs, axis=0)
                i_row, _ = emit("V1+V2", "random", pol, gain=float(g_mean.sum()), rate=float(np.mean(rs)),
                                rmin=float(np.mean(mins)), rmax=float(np.mean(maxs)))
                for variant in ("V1", "V2"):
                    add(ci, ("random", pol, k, variant), g_mean, i_row)

            for s in SIGNALS:
                if s == "random":
                    continue
                if s not in c["official"]:
                    reason = UNC_REASON if s == "uncertainty" else "signal not defined for this track"
                    for variant in ("V1", "V2"):
                        rows.append({**base, "variant": variant, "signal": s, "policy": "A", "available": False,
                                     "undefined_reason": reason})
                    continue
                off_test = np.asarray(c["official"][s], float)[c["test"]]
                gC = float(t92.topk_expect(np.nan_to_num(off_test, nan=-np.inf), [v], [kn])[0][0][0])
                i_row, _ = emit("official", s, "C", gain=gC, rate=kn / n, rmin=np.nan, rmax=np.nan)
                add(ci, (s, "C", k, "official"), ("topk", np.nan_to_num(off_test, nan=-np.inf)), i_row)
                for variant in ("V1", "V2", "TEST"):
                    if (variant, s) not in c["scores"]:
                        if variant != "TEST":
                            rows.append({**base, "variant": variant, "signal": s, "policy": "A", "available": False,
                                         "undefined_reason": R2_REASON})
                        continue
                    cal, test_s = c["scores"][(variant, s)]
                    tau, p = calibrate(cal, k)
                    if variant == "TEST":
                        st = np.round(np.nan_to_num(test_s, nan=-np.inf), 9)
                        g_exp = float(v[st > tau].sum() + p * v[st == tau].sum())
                        emit("TEST", s, "A_expected", gain=g_exp,
                             rate=float((st > tau).mean() + p * (st == tau).mean()), rmin=np.nan, rmax=np.nan,
                             tau=tau, p=p)
                        continue
                    rng = np.random.default_rng(seed_of(c["system"], c["target"], c["geometry"], s, k, variant))
                    mA = mask_A(test_s, tau, p, rng)
                    for pol, m in (("A", mA), ("B", apply_cap(mA, k, c["unit_id_test"], c["stream"]))):
                        i_row, g_u = emit(variant, s, pol, mask=m, tau=tau, p=p)
                        add(ci, (s, pol, k, variant), g_u, i_row)
        print(f"  point estimates {c['track']:8s} {c['system']:10s} {c['target']:8s}", flush=True)

    # ---- measured budgets: policy B at the feasible share, plus all-cheap and uniform full fidelity
    for ci, c in enumerate(cells):
        v, n = c["v"], c["n"]
        c0, c1 = c["cost"]["cheap"]["ms"], c["cost"]["640"]["ms"]
        undef_cell = (f"near-zero oracle: {c['n_aff']} affected states on this split (< {MIN_AFFECTED})"
                      if c["track"] == "nuPlan" and c["n_aff"] < MIN_AFFECTED else "")
        for f in BUDGET_LEVELS:
            budget = c0 + f * c1
            feasible_full = budget >= c1
            prize_full = float(t92.topk_expect(v, [v], [n])[0][0][0])
            gain_full = float(v.sum())
            base_b = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                          n_test_frames=n, n_test_units=len(c["uniq"]), n_affected=c["n_aff"], budget_level=f,
                          budget_ms=budget, cheap_ms=c0, full_ms=c1, all_cheap_loss=c["tot_cheap"],
                          ndg_defined=not undef_cell, undefined_reason=undef_cell)
            rows.append({**base_b, "variant": "baseline", "signal": "all_cheap", "policy": "budget", "available": True,
                         "rate_target": 0.0, "rate_realised": 0.0, "gain": 0.0, "gain_share_all_cheap": 0.0, "ndg": 0.0})
            rows.append({**base_b, "variant": "baseline", "signal": "uniform_full", "policy": "budget",
                         "available": bool(feasible_full),
                         "rate_target": 1.0 if feasible_full else np.nan, "rate_realised": 1.0 if feasible_full else np.nan,
                         "gain": gain_full if feasible_full else np.nan,
                         "gain_share_all_cheap": (gain_full / c["tot_cheap"]) if feasible_full and c["tot_cheap"] > EPS else np.nan,
                         "ndg": (gain_full / prize_full) if feasible_full and prize_full > EPS and not undef_cell else np.nan,
                         "undefined_reason": undef_cell or ("" if feasible_full else
                                                            f"a full pass needs {c1:.2f} ms > {budget:.2f} ms budget")})
            for s in SIGNALS:
                if s == "random" or s not in c["official"]:
                    continue
                o_ms = t93.signal_overhead(ov, c["track"], s)[0]
                k = max((budget - c0 - o_ms) / c1, 0.0)
                kn = max(int(round(k * n)), 1)
                prize = float(t92.topk_expect(v, [v], [kn])[0][0][0])
                why = undef_cell or ("oracle prize is zero at this budget" if prize <= EPS else "")
                for variant in ("V1", "V2"):
                    if (variant, s) not in c["scores"]:
                        rows.append({**base_b, "variant": variant, "signal": s, "policy": "B", "available": False,
                                     "undefined_reason": R2_REASON})
                        continue
                    cal, test_s = c["scores"][(variant, s)]
                    tau, p = calibrate(cal, k)
                    rng = np.random.default_rng(seed_of(c["system"], c["target"], c["geometry"], s, f, variant, "budget"))
                    m = apply_cap(mask_A(test_s, tau, p, rng), k, c["unit_id_test"], c["stream"])
                    g_u, r_u = per_unit(v, m, c["idx_list"])
                    gain = float(g_u.sum())
                    rows.append({**base_b, "variant": variant, "signal": s, "policy": "B", "available": True,
                                 "rate_target": k, "overhead_ms": o_ms, "rate_realised": float(m.mean()),
                                 "rate_unit_min": float(r_u.min()), "rate_unit_max": float(r_u.max()), "gain": gain,
                                 "gain_share_all_cheap": gain / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan,
                                 "ndg": (gain / prize) if (prize > EPS and not why) else np.nan, "oracle_prize": prize,
                                 "ndg_defined": not why, "undefined_reason": why, "tau": tau, "tie_p": p})
        print(f"  budgets {c['track']:8s} {c['system']:10s} {c['target']:8s}", flush=True)

    # ---- joint bootstrap: units resampled once per dataset per draw, shared by every cell and series
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units_of = {ds: sorted(set().union(*[set(c["uniq"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    by_cell_rate = {}
    for (ci, key) in series:
        by_cell_rate.setdefault((ci, key[2]), []).append(key)
    for c in cells:
        c["prize0"] = {k: float(t92.topk_expect(c["v"], [c["v"]], [max(int(round(k * c["n"])), 1)])[0][0][0]) for k in RATES}
    draws = {kk: np.full(args.nboot, np.nan) for kk in series}
    dropped = {kk: 0 for kk in series}
    rng = np.random.default_rng(0)
    t_start = time.time()
    for b in range(args.nboot):
        pick = {ds: rng.integers(0, len(units_of[ds]), len(units_of[ds])) for ds in datasets}
        for ci, c in enumerate(cells):
            us = [u for u in (units_of[c["dataset"]][i] for i in pick[c["dataset"]]) if u in c["upos"]]
            if not us:
                continue
            counts = np.bincount([c["upos"][u] for u in us], minlength=len(c["uniq"])).astype(float)
            t = np.concatenate([c["idx_list"][c["upos"][u]] for u in us])
            vt = c["v"][t]
            for k in RATES:
                kn = max(int(round(k * len(t))), 1)
                pz = float(t92.topk_expect(vt, [vt], [kn])[0][0][0])
                ok = pz > max(EPS, 0.25 * c["prize0"][k])
                for key in by_cell_rate.get((ci, k), ()):
                    if not ok:
                        dropped[(ci, key)] += 1
                        continue
                    val = series[(ci, key)]
                    g = (float(t92.topk_expect(val[1][t], [vt], [kn])[0][0][0]) if isinstance(val, tuple)
                         else float(val @ counts))
                    draws[(ci, key)][b] = g / pz
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{args.nboot} [{time.time() - t_start:.0f}s]", flush=True)

    for (ci, key), x in draws.items():
        i_row = row_of[(ci, key)]
        lo, hi = t92.ci(x)
        rows[i_row]["ndg_lo"], rows[i_row]["ndg_hi"] = lo, hi
        rows[i_row]["boot_dropped"] = int(dropped[(ci, key)])
        if key[0] != "random":
            rnd = draws.get((ci, ("random", key[1] if key[1] != "C" else "A", key[2],
                                  "V1" if key[3] == "official" else key[3])))
            if rnd is not None:
                dr = x - rnd
                dlo, dhi = t92.ci(dr)
                rows[i_row]["minus_random"] = float(np.nanmean(dr))
                rows[i_row]["minus_random_lo"], rows[i_row]["minus_random_hi"] = dlo, dhi

    # ---- pooled primary statistic
    pooled_cells = [i for i, c in enumerate(cells) if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    if args.max_cells:
        print("  debug run: the pooled statistic needs all 12 cells, skipping it")
        pd.DataFrame(rows).to_csv(run / "causal_threshold.csv", index=False)
        return
    assert len(pooled_cells) == 12, len(pooled_cells)
    per_signal, left_out = {}, None
    for s in LEARNED:
        diffs = [draws[(ci, (s, "B", POOLED_RATE, "V1"))] - draws[(ci, (s, "C", POOLED_RATE, "official"))]
                 for ci in pooled_cells if (ci, (s, "B", POOLED_RATE, "V1")) in draws]
        if diffs:
            arr = np.array(diffs)
            left_out = int(np.isnan(arr).any(axis=0).sum()) if left_out is None else left_out
            per_signal[s] = np.nanmean(arr, axis=0)
    pooled_all = np.nanmean(np.array(list(per_signal.values())), axis=0)
    for name, arr in [("all_learned", pooled_all)] + list(per_signal.items()):
        lo, hi = t92.ci(arr)
        rows.append({"track": "pooled", "geometry": "n/a", "system": "12 cells", "target": "B_minus_official",
                     "variant": "V1", "signal": name, "policy": "B-C", "rate_target": POOLED_RATE, "available": True,
                     "ndg": float(np.nanmean(arr)), "ndg_lo": lo, "ndg_hi": hi, "boot_dropped": left_out})
    lo, hi = t92.ci(pooled_all)
    reading = "streaming holds" if lo > -0.05 else "streaming costs" if hi < -0.05 else "inconclusive"
    rows.append({"track": "pooled", "geometry": "n/a", "system": "12 cells", "target": "reading", "variant": "V1",
                 "signal": "all_learned", "policy": "B-C", "rate_target": POOLED_RATE, "available": True,
                 "ndg": float(np.nanmean(pooled_all)), "ndg_lo": lo, "ndg_hi": hi, "undefined_reason": reading})

    df = pd.DataFrame(rows)
    chk = pd.DataFrame(refit_check)
    for out in (FINAL / "causal_threshold.csv", run / "causal_threshold.csv"):
        df.to_csv(out, index=False)
    chk.to_csv(run / "refit_matches_official.csv", index=False)
    print(f"  wrote {FINAL / 'causal_threshold.csv'} ({len(df)} rows)")
    print(f"  refit on train+val vs official scores: max abs diff {chk.max_abs_diff_vs_official.max():.3g} over {len(chk)} rows")
    san = df[(df.variant == "TEST") & df.ndg.notna()][["track", "geometry", "system", "target", "signal", "rate_target", "ndg"]]
    off = df[(df.variant == "official") & df.ndg.notna()][["track", "geometry", "system", "target", "signal", "rate_target", "ndg"]]
    m = san.merge(off, on=["track", "geometry", "system", "target", "signal", "rate_target"], suffixes=("_test_tau", "_official"))
    bad = m[(m.ndg_test_tau.round(3) != m.ndg_official.round(3))]
    print(f"  sanity: test-calibrated threshold reproduces the official nDG in {len(m) - len(bad)}/{len(m)} rows")
    if len(bad):
        print(bad.head(10).to_string(index=False))
    print(f"  pooled B - official (V1, 20%, 12 cells, six learned signals): {np.nanmean(pooled_all):+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}] -> {reading}")


if __name__ == "__main__":
    main()

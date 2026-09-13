#!/usr/bin/env python
"""Benchmark: allocation under measured latency and energy budgets instead of a frame count.

Pre-registered in RESEARCH_LOG.md (2026-09-13 23:55).  Until now every result used a quota of
frames and the Jetson profile only decorated the tables.  Here the budget is mean milliseconds or
millijoules per frame, charged with costs measured on this board:

  cascade   CHEAP runs on every frame; the allocator adds its own per-frame overhead; escalated
            frames additionally run the chosen higher fidelity.
  costs     end-to-end median latency and GPU-rail energy per frame from the profile runs cited in
            docs/full_project_report.md (KITTI 20260912_021305, nuScenes 20260912_073228).  nuPlan
            uses the KITTI YOLOv8s numbers, since its miss model was fitted on those outcomes.
  overhead  measured by this script on an otherwise idle board: single-frame CPU latency of the
            cheap-side features and of gate inference, and CPU-rail power over idle while they run.
            Diagnostics need FULL on every frame, so no budget below CHEAP + FULL pays for them.

With two fidelity levels a ms budget is a linear relabelling of a frame quota; what changes there is
only how many escalations the overhead eats, and it is reported as that.  Where cost can change the
decision itself is a choice among fidelities: KITTI mono, cascade 320 -> {384, 512, 640}, whose
latency and energy disagree about the intermediate levels.  The multi-fidelity oracle is a greedy
walk along each frame's concave cost-gain hull, which solves the LP relaxation up to one fractional
frame; that bound is reported next to it.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import features as F, geometry as G, power, predict, runmeta          # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

_spec = importlib.util.spec_from_file_location("bench92", ROOT / "scripts" / "92_benchmark_table.py")
t92 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t92)

QUOTAS, EPS = t92.QUOTAS, t92.EPS
UNITS = {"ms": "lat_e2e_ms_median", "mJ": "energy_gpu_mj_per_frame"}
PROFILE = {"nuScenes": ("20260912_073228_profile", {"cheap": "ns_cheap_320", "640": "ns_full_640"}),
           "KITTI": ("20260912_021305_profile", {"cheap": "cheap_320", "384": "cheap_384",
                                                 "512": "cheap_512", "640": "full_640"})}
PROFILE["nuPlan"] = PROFILE["KITTI"]
# docs/full_project_report.md, (e2e median ms, GPU mJ); the profile files must agree within 2%
REPORT = {"cheap_320": (13.16, 13.9), "cheap_384": (14.28, 19.0), "cheap_512": (16.72, 27.4),
          "full_640": (18.34, 32.8), "ns_cheap_320": (12.80, 23.7), "ns_full_640": (19.36, 66.5)}
DEPLOYABLE = ("random", "uncertainty", "criticality_cheap", "gate_ridge", "gate_gbm")
DIAGNOSTIC = ("criticality_gt", "dE_exact", "dE_E6_risk_weighted", "PKL", "TIP")


def profile_costs(track):
    run, modes = PROFILE[track]
    s = json.loads((ROOT / "results" / "raw" / run / "profile_summary.json").read_text())
    out = {}
    for lvl, mode in modes.items():
        ms, mj = s[mode][UNITS["ms"]], s[mode][UNITS["mJ"]]
        rms, rmj = REPORT[mode]
        assert abs(ms / rms - 1) < 0.02 and abs(mj / rmj - 1) < 0.02, (mode, ms, rms, mj, rmj)
        out[lvl] = {"ms": float(ms), "mJ": float(mj), "mode": mode, "run": run}
    return out


# ----------------------------------------------------------------------------------------------
# overhead, measured here

def _rails_during(fn, seconds):
    samples, stop = [], threading.Event()

    def loop():
        while not stop.is_set():
            samples.append(power.read_power_mw())
            time.sleep(0.05)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    t0 = time.time()
    while time.time() - t0 < seconds:
        fn()
    stop.set()
    th.join()
    keys = samples[0].keys() if samples else []
    return {k: float(np.mean([s[k] for s in samples if k in s])) for k in keys}


def _frames(det_dir, mode, limit):
    out = []
    for f in sorted((det_dir / mode).glob("*.npz")):
        c = DetCache(f)
        for i in range(len(c.frames)):
            sc = c.scalars(i)
            img = {k: v for k, v in sc.items()
                   if k.startswith(("img_bright", "img_dark", "img_lap", "img_edge", "img_contrast", "motion"))}
            out.append((c.det(i), c.geo(i), img, sc.get("img_w", 1600.0) * sc.get("img_h", 900.0)))
            if len(out) >= limit:
                return out
    return out


def _median_ms(fn, items, passes=3):
    ts = []
    for _ in range(passes):
        for it in items:
            t = time.perf_counter()
            fn(it)
            ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts))


def measure_overheads(n_frames=400):
    cfg, model = RiskConfig(), G.PRIMARY
    res = {"power_available": power.AVAILABLE}
    X_by = {}
    for track, det_dir, mode in (("nuScenes", Path(CACHE) / "nusc_det_tv", "ns_cheap_320"),
                                 ("KITTI", Path(CACHE) / "det", "cheap_320")):
        fr = _frames(det_dir, mode, n_frames)
        keep = lambda it: it[0]["conf"] >= cfg.op_conf                     # noqa: E731
        res[track] = {
            "uncertainty_ms": _median_ms(lambda it: F.uncertainty_features(it[0], keep(it), cfg.op_conf), fr),
            "criticality_cheap_ms": _median_ms(lambda it: F.criticality_features(it[0], keep(it), it[1], model), fr),
            "features_ms": _median_ms(lambda it: F.frame_features(it[0], it[1], it[2], it[3], model, cfg.op_conf), fr)}
        X_by[track] = np.nan_to_num(np.array(
            [list(F.frame_features(*it, model, cfg.op_conf).values()) for it in fr], float))
    X = X_by["nuScenes"]
    y = X[:, :5].sum(1) + np.random.default_rng(0).normal(size=len(X))  # timing only: shape of the model is fixed
    for name, mdl in (("ridge", "linear"), ("gbm", "gbm")):
        m = predict.make_model(mdl, "reg", 0).fit(X, y)
        rows = [X[i:i + 1] for i in range(len(X))]
        res[f"{name}_predict_ms"] = _median_ms(lambda r: m.predict(r), rows)
    if power.AVAILABLE:
        fr = _frames(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", 200)
        gbm = predict.make_model("gbm", "reg", 0).fit(X, y)
        idx = {"i": 0}

        def work():
            it = fr[idx["i"] % len(fr)]
            idx["i"] += 1
            x = np.nan_to_num(np.array([list(F.frame_features(*it, G.PRIMARY, cfg.op_conf).values())], float))
            gbm.predict(x)

        idle = _rails_during(lambda: time.sleep(0.02), 8.0)
        busy = _rails_during(work, 8.0)
        res["rails_idle_mw"], res["rails_busy_mw"] = idle, busy
        cpu = [k for k in busy if "CPU" in k.upper()]
        res["cpu_rail"] = cpu[0] if cpu else None
        res["cpu_mw_over_idle"] = float(busy[cpu[0]] - idle[cpu[0]]) if cpu else np.nan
        res["all_rails_mw_over_idle"] = float(sum(busy[k] - idle.get(k, 0.0) for k in busy))
    else:
        res["cpu_mw_over_idle"] = np.nan
    return res


def signal_overhead(ov, track, signal, n_models=1):
    """(ms, mJ) per frame an allocator adds on top of CHEAP, and where the number comes from."""
    base = ov["nuScenes"] if track == "nuPlan" else ov[track]
    src = "measured" if track != "nuPlan" else "nuScenes feature time as proxy (track features not timed)"
    ms = {"random": 0.0, "oracle": 0.0, "uncertainty": base["uncertainty_ms"],
          "criticality_cheap": base["criticality_cheap_ms"],
          "gate_ridge": base["features_ms"] + n_models * ov["ridge_predict_ms"],
          "gate_gbm": base["features_ms"] + n_models * ov["gbm_predict_ms"]}[signal]
    mw = ov.get("cpu_mw_over_idle", np.nan)
    mj = ms * max(mw, 0.0) / 1e3 if np.isfinite(mw) else 0.0
    return ms, mj, (src if signal not in ("random", "oracle") else "none")


# ----------------------------------------------------------------------------------------------
# two fidelity levels

def two_level(cells, ov, nboot, rng):
    rows = []
    for c in cells:
        d = c["d"]
        cost = profile_costs(c["track"])
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        gates = t92.gate_predictions(d, c["fcols"], v_all)
        m = (d.split == "test").to_numpy()
        v, units = v_all[m], d.unit.to_numpy()[m]
        n = len(v)
        scores = {"random": None, "oracle": v, **{k: g[m] for k, g in gates.items()}}
        for s in ("uncertainty", "criticality_cheap"):
            if s in c["cols"]:
                scores[s] = pd.to_numeric(d[c["cols"][s]], errors="coerce").fillna(-np.inf).to_numpy()[m]
        uniq = np.unique(units)
        idx = [np.flatnonzero(units == u) for u in uniq]
        draws = [np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(nboot)]
        key = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"])
        for unit in UNITS:
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
                ok = prize_t > max(EPS, 0.25 * prize)
                rand_t = np.array([gain_at(None, f, t) for t in draws]) / np.where(ok, prize_t, 1)
                for s in list(scores) + [x for x in DIAGNOSTIC if x in c["cols"]]:
                    base = {**key, "unit": unit, "budget_level": f, "budget_per_frame": budget,
                            "cheap_cost": c0, "full_cost": c1, "signal": s, "n_frames": n}
                    if s in DIAGNOSTIC:
                        rows.append({**base, "feasible": False,
                                     "note": f"needs FULL on every frame: >= {c0 + c1:.2f} {unit} per frame "
                                             f"before computing the metric"})
                        continue
                    o_ms, o_mj, src = signal_overhead(ov, c["track"], s)
                    o = o_ms if unit == "ms" else o_mj
                    frac = max((budget - c0 - o) / c1, 0.0)
                    g = gain_at(scores[s], frac)
                    e_t = np.array([gain_at(scores[s], frac, t) for t in draws]) / np.where(ok, prize_t, 1)
                    e_t, dr = np.where(ok, e_t, np.nan), np.where(ok, e_t - rand_t, np.nan)
                    lo, hi = t92.ci(e_t)
                    dlo, dhi = t92.ci(dr)
                    rows.append({**base, "feasible": True, "overhead": o, "overhead_source": src,
                                 "escalated_frac": frac, "escalations_lost_to_overhead": f - frac,
                                 "eta": g / prize if prize > EPS else np.nan, "eta_lo": lo, "eta_hi": hi,
                                 "minus_random": float(np.nanmean(dr)), "minus_random_lo": dlo,
                                 "minus_random_hi": dhi, "boot_dropped": int((~ok).sum())})
        print(f"  two-level {c['track']:8s} {c['geometry']:6s} {c['system']:10s} {c['target']}", flush=True)
    return rows


# ----------------------------------------------------------------------------------------------
# several fidelity levels (KITTI mono)

def hull_increments(gains, costs):
    """Upper concave hull of (cost, gain) from the origin, per frame, as ordered increments."""
    n, L = gains.shape
    pc = np.r_[0.0, costs]
    fr, a_, b_ = [], [], []
    for i in range(n):
        g = np.r_[0.0, gains[i]]
        hull = [0]
        for j in range(1, L + 1):
            if g[j] <= g[hull[-1]]:
                continue
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (g[j] - g[a]) * (pc[b] - pc[a]) >= (g[b] - g[a]) * (pc[j] - pc[a]):
                    hull.pop()
                else:
                    break
            hull.append(j)
        for a, b in zip(hull[:-1], hull[1:]):
            fr.append(i); a_.append(a); b_.append(b)
    fr, a_, b_ = np.array(fr, int), np.array(a_, int), np.array(b_, int)
    return fr, a_, b_, pc[b_] - pc[a_]


def prepare(pred, true, costs):
    """Hull increments of the predicted gains, sorted by efficiency, with their true gain.

    Independent of budget and bootstrap weights, so it is computed once per (signal, cost unit)."""
    fr, a, b, dc = hull_increments(pred, costs)
    Gt = np.c_[np.zeros(len(true)), true]
    Gp = np.c_[np.zeros(len(pred)), pred]
    eff = (Gp[fr, b] - Gp[fr, a]) / dc if len(fr) else np.zeros(0)
    o = np.argsort(-eff, kind="stable")
    return {"fr": fr[o], "b": b[o], "dc": dc[o], "dg": (Gt[fr, b] - Gt[fr, a])[o], "n": len(pred)}


def greedy(inc, budget_total, weights):
    """Take increments in efficiency order until one no longer fits; score with true gains.

    Returns the gain, the level chosen per frame, and the LP bound gap: the fraction of the first
    increment that did not fit, which is all that separates this from the LP relaxation optimum."""
    level = np.zeros(inc["n"], int)
    if len(inc["fr"]) == 0:
        return 0.0, level, 0.0
    w = weights[inc["fr"]]
    cum = np.cumsum(inc["dc"] * w)
    take = cum <= budget_total + 1e-9
    gain = float(np.sum(inc["dg"] * w * take))
    np.maximum.at(level, inc["fr"][take], inc["b"][take])
    nxt = np.flatnonzero(~take)
    lp = 0.0
    if len(nxt):
        j = nxt[0]
        spent = cum[j - 1] if j > 0 else 0.0
        lp = max((budget_total - spent) / (inc["dc"][j] * w[j]), 0.0) * max(inc["dg"][j], 0.0) * w[j]
    return gain, level, float(lp)


def multi_fidelity(ov, nboot, rng):
    cells = [c for c in t92.kitti_cells(json.loads((ROOT / "configs" / "benchmark_splits.json").read_text()))
             if c["geometry"] == "mono"]
    d = cells[0]["d"].copy()
    fcols = cells[0]["fcols"]
    lv = {}
    for res in ("384", "512"):
        t = pd.read_pickle(t92.CORE / f"KITTI__YOLOv8s__cheap_{res}tofull_640__mono.pkl")
        b = pd.read_pickle(t92.PLANB / f"planB__KITTI__YOLOv8s__cheap_{res}tofull_640__mono.pkl")
        for x in (t, b):
            x["seq"], x["frame"] = x.seq.astype(str), x.frame.astype(int)
        lv[res] = t[["seq", "frame", "J_cheap", "J_full"]].merge(
            b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
        lv[res].columns = ["seq", "frame", f"J_{res}", f"Jf_{res}", f"JB_{res}", f"JBf_{res}"]
        d = d.merge(lv[res], on=["seq", "frame"], validate="one_to_one")
        assert np.allclose(d[f"Jf_{res}"], d.J_full) and np.allclose(d[f"JBf_{res}"], d.JB_full), \
            f"FULL cost differs between the 320 and {res} tables"
    cost = profile_costs("KITTI")
    m = (d.split == "test").to_numpy()
    fit = d.split.isin(["train", "val"]).to_numpy()
    X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                      nan=0.0, posinf=1e6, neginf=-1e6)
    units = d.unit.to_numpy()[m]
    uniq = np.unique(units)
    rows, mixes = [], []
    for system, base, levels in (("brake", "J_cheap", ["J_384", "J_512", "J_full"]),
                                 ("traj", "JB_cheap", ["JB_384", "JB_512", "JB_full"])):
        gains = np.stack([(d[base] - d[c]).to_numpy(float) for c in levels], 1)        # n x 3
        preds = {"oracle": gains}
        for name, mdl in (("gate_ridge", "linear"), ("gate_gbm", "gbm")):
            P = np.zeros_like(gains)
            for j in range(3):
                P[:, j] = predict.make_model(mdl, "reg", 0).fit(X[fit], gains[fit, j]).predict(X)
            preds[name] = P
        gt = gains[m]
        n = len(gt)
        w1 = np.ones(n)
        wdraw = []
        for _ in range(nboot):
            w = np.zeros(n)
            for i in rng.integers(0, len(uniq), len(uniq)):
                w[units == uniq[i]] += 1
            wdraw.append(w)
        unc = pd.to_numeric(d["unc_sum"], errors="coerce").fillna(-np.inf).to_numpy()[m]
        for unit in UNITS:
            c320 = cost["cheap"][unit]
            lvl_cost = np.array([cost["384"][unit], cost["512"][unit], cost["640"][unit]])
            other = "mJ" if unit == "ms" else "ms"
            lvl_other = np.array([cost["384"][other], cost["512"][other], cost["640"][other]])
            incs = {name: prepare(P[m], gt, lvl_cost) for name, P in preds.items()}
            for f in QUOTAS:
                budget = c320 + f * cost["640"][unit]
                o_or, _, lp = greedy(incs["oracle"], n * f * cost["640"][unit], w1)
                o_t = np.array([greedy(incs["oracle"], w.sum() * f * cost["640"][unit], w)[0] for w in wdraw])
                ok = o_t > max(EPS, 0.25 * o_or)
                key = dict(system=system, unit=unit, budget_level=f, budget_per_frame=budget, n_frames=n)
                for name, P in preds.items():
                    o_ms, o_mj, _ = signal_overhead(ov, "KITTI", name, n_models=3)
                    o = 0.0 if name == "oracle" else (o_ms if unit == "ms" else o_mj)
                    extra = max(budget - c320 - o, 0.0)
                    g, level, _ = greedy(incs[name], n * extra, w1)
                    e_t = np.array([greedy(incs[name], w.sum() * extra, w)[0] for w in wdraw])
                    e_t = np.where(ok, e_t / np.where(ok, o_t, 1), np.nan)
                    lo, hi = t92.ci(e_t)
                    spend_other = float(lvl_other[level[level > 0] - 1].sum() / n)
                    rows.append({**key, "signal": f"{name} (multi-fidelity)", "overhead": o,
                                 "eta": g / o_or if o_or > EPS else np.nan, "eta_lo": lo, "eta_hi": hi,
                                 "gain": g, "oracle_gain": o_or, "oracle_lp_bound_gap": lp,
                                 "share_384": float((level == 1).mean()), "share_512": float((level == 2).mean()),
                                 "share_640": float((level == 3).mean()),
                                 f"extra_{other}_per_frame": spend_other,
                                 f"extra_{other}_budget_at_same_level": f * cost["640"][other],
                                 "boot_dropped": int((~ok).sum())})
                # single-level comparators: everything escalated goes to 640
                for name, sc in (("random", None), ("uncertainty", unc), ("gate_gbm", preds["gate_gbm"][m, 2]),
                                 ("oracle", gt[:, 2])):
                    o_ms, o_mj, _ = signal_overhead(ov, "KITTI", name)
                    o = o_ms if unit == "ms" else o_mj
                    frac = max((budget - c320 - o) / cost["640"][unit], 0.0)
                    k = int(np.floor(frac * n + 1e-9))
                    g = (k / n * gt[:, 2].sum() if sc is None else
                         float(t92.topk_expect(sc, [gt[:, 2]], [max(k, 1)])[0][0][0]) if k > 0 else 0.0)
                    rows.append({**key, "signal": f"{name} (640 only)", "overhead": o,
                                 "eta": g / o_or if o_or > EPS else np.nan, "gain": g, "oracle_gain": o_or,
                                 "share_640": k / n})
            print(f"  multi-fidelity KITTI mono {system} {unit}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--tag", default="benchmark_budget")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    rng = np.random.default_rng(0)

    ov = measure_overheads()
    print("  overheads:", json.dumps({k: v for k, v in ov.items() if not k.startswith("rails")}, default=float))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    cells = [c for gen in (t92.nuscenes_cells, t92.kitti_cells, t92.nuplan_cells) for c in gen(splits)]
    two = pd.DataFrame(two_level(cells, ov, args.nboot, rng))
    multi = pd.DataFrame(multi_fidelity(ov, args.nboot, rng))

    out = Path(RESULTS) / "final"
    costs = {t: profile_costs(t) for t in PROFILE}
    for name, obj in (("benchmark_budget_overheads.json", {"overheads": ov, "costs": costs}),):
        (out / name).write_text(json.dumps(obj, indent=1, default=float))
        (run / name).write_text(json.dumps(obj, indent=1, default=float))
    for name, df in (("benchmark_budget_two_level", two), ("benchmark_budget_multifidelity", multi)):
        df.to_csv(out / f"{name}.csv", index=False)
        df.to_csv(run / f"{name}.csv", index=False)
        print(f"  wrote {out / (name + '.csv')} ({len(df)} rows)")
    s = multi[(multi.budget_level == 0.2)][["system", "unit", "signal", "eta", "share_384", "share_512", "share_640"]]
    print(s.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

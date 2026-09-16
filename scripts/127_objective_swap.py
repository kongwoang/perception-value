#!/usr/bin/env python
"""Task 13 Part D: does a perception-centric evaluation objective pick a different allocator?

Pre-registered in RESEARCH_LOG.md (Task 13 Part D), committed before this script was run.  Task 9 asked which
target trains a better model and returned a pre-registered null; that null stands and is not revisited here.  The
question here is different: holding the method pool and the protocol fixed, which method does each *evaluation
objective* select?

  E_dec   realised decision value / decision-value oracle prize at the budget -- the official nDG.
  E_perc  the identical protocol with the per-input value replaced by a perception gain, normalised by the
          perception-gain oracle at the same budget.  Run twice: G^dE = dE_E1_fn_only (missed objects) and
          E_risk = dE_E6_risk_weighted, because a perception-first author could pick either.

Nothing official is written:

  results/final/objective_swap.csv   section=score   one row per cell x objective x quota x signal
                                     section=compare one row per cell x E_perc variant x quota x pool
                                     section=sanity  E_dec against the official held-out nDG

The method pool is split into three tiers and reported separately; the headline is tiers 1 and 3.
  tier 1  target-free, non-circular core: random, cheap-detection uncertainty, cheap-side criticality, ego speed.
  tier 2  V-trained, ADVANTAGED under E_dec and flagged as such: ridge gate, GBM gate, R1 routers, R2 pixel router.
  tier 3  perception diagnostics: exact perception gain, dE, E_risk, PKL, TIP.

Selection is made from the point estimate; the paired unit bootstrap then measures the realised decision value of
the two fixed methods, and EVERY draw is kept -- the official 25% prize filter is reported alongside, never
substituted.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                          # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402

RAW = ROOT / "results" / "raw"
FINAL = Path(RESULTS) / "final"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t92 = _load("t92", "92_benchmark_table.py")
t120 = _load("t120", "120_nuplan_real_allocation.py")

EPS, QUOTAS = t92.EPS, t92.QUOTAS
MIN_AFFECTED = t120.MIN_AFFECTED

TIER1 = ["random", "uncertainty", "criticality_cheap", "trivial_ego_speed"]
TIER2 = ["gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf"]
TIER3 = ["dE_exact", "dE_E1_fn_only", "dE_E6_risk_weighted", "PKL", "TIP"]
POOL = TIER1 + TIER2 + TIER3
TIER_OF = {**{s: 1 for s in TIER1}, **{s: 2 for s in TIER2}, **{s: 3 for s in TIER3}}
HEADLINE = [s for s in POOL if TIER_OF[s] in (1, 3)]

R1 = ["R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
OBJECTIVES = [("E_perc_dE", "dE_E1_fn_only"), ("E_perc_risk", "dE_E6_risk_weighted")]

NO_UNC_NUPLAN = "the real-perception features carry no cheap-detection uncertainty"
NO_R2_NUPLAN = "the pixel router was trained on nuScenes and KITTI images only"
NO_DE_EXACT_NUPLAN = "no detection-level error table exists for the real-perception nuPlan cells"


def latest(tag):
    d = [p for p in sorted(RAW.glob(f"*_{tag}")) if p.name.split("_", 2)[-1] == tag]
    if not d:
        raise SystemExit(f"no results/raw/*_{tag} run found")
    return d[-1]


def id_columns(files):
    for a, b in (("seq", "frame"), ("scenario", "iteration")):
        if a in files and b in files:
            return a, b
    raise SystemExit(f"no identifier columns found among {files}")


def aligned(z, keys, names):
    """Cached router scores aligned to the cell's rows by identifier; uncovered rows come back as NaN."""
    ida, idb = id_columns(list(z.files))
    src = pd.DataFrame({ida: z[ida].astype(str), idb: z[idb].astype(int)})
    for n in names:
        src[n] = np.asarray(z[n], float)
    m = keys.merge(src, on=[ida, idb], how="left", validate="one_to_one")
    return {n: m[n].to_numpy(float) for n in names}


def build_cells(splits):
    """The 14 official held-out cells: 10 core, 4 nuPlan real-perception."""
    r1_run, r2_run, nr_run = latest("routers_r1"), latest("router_r2"), latest("nuplan_real_allocation")
    cells = []
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            stem = f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz"
            keys = d[["seq", "frame"]].astype({"seq": str, "frame": int}).reset_index(drop=True)
            z1 = np.load(r1_run / stem, allow_pickle=False)
            assert np.array_equal(z1["V"], v), f"R1 cache does not match the cell's V: {stem}"
            scores = dict(aligned(z1, keys, R1))
            f2 = r2_run / stem
            if f2.exists():
                z2 = np.load(f2, allow_pickle=False)
                scores.update(aligned(z2, keys, ["R2_cnn_clf"]))
            gates = t92.gate_predictions(d, c["fcols"], v)
            scores.update({g: gates[g] for g in ("gate_ridge", "gate_gbm")})
            col = c["cols"]
            for s, src in (("uncertainty", col.get("uncertainty")), ("criticality_cheap", col.get("criticality_cheap")),
                           ("trivial_ego_speed", "v_ego"), ("dE_exact", col.get("dE_exact")),
                           ("dE_E1_fn_only", col.get("dE_E1_fn_only")),
                           ("dE_E6_risk_weighted", col.get("dE_E6_risk_weighted")),
                           ("PKL", col.get("PKL")), ("TIP", col.get("TIP"))):
                if src is not None and src in d.columns:
                    scores[s] = pd.to_numeric(d[src], errors="coerce").to_numpy(float)
            na = dict(c["na"])
            cells.append(dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"],
                              d=d, v=v, cheap=c["cheap"], scores=scores, na=na))

    fcols, _ = t120.register_features()
    for c in t120.cells(splits):
        d = c["d"].reset_index(drop=True)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(nr_run / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        assert np.array_equal(z["V"], v), f"nuPlan cache does not match V: {c['system']} {c['target']}"
        keys = d[["scenario", "iteration"]].astype({"scenario": str, "iteration": int}).reset_index(drop=True)
        scores = dict(aligned(z, keys, R1 + ["gate_ridge", "gate_gbm"]))
        for s, src in (("criticality_cheap", "nr_crit_cheap_sum"), ("trivial_ego_speed", "nr_ego_speed"),
                       ("dE_E1_fn_only", "dE_E1_fn_only"), ("dE_E6_risk_weighted", "dE_E6_risk_weighted")):
            assert src in d.columns, f"nuPlan cell is missing {src}"
            scores[s] = pd.to_numeric(d[src], errors="coerce").to_numpy(float)
        na = {"uncertainty": NO_UNC_NUPLAN, "R2_cnn_clf": NO_R2_NUPLAN, "dE_exact": NO_DE_EXACT_NUPLAN,
              "PKL": "PKL and TIP are defined on nuScenes rasters only",
              "TIP": "PKL and TIP are defined on nuScenes rasters only"}
        cells.append(dict(track="nuPlan", geometry="n/a", system=c["system"], target=c["target"],
                          d=d, v=v, cheap=c["cheap"], scores=scores, na=na))
    return cells


def eta_of(value, series, ks, prize):
    """nDG of one score series against one value vector, exact tie expectation, 92's convention."""
    if series is None:                                     # random: every input tied
        gain = ks / len(value) * value.sum()
    else:
        gain = t92.topk_expect(series, [value], ks)[0][0]
    return gain, np.where(prize > EPS, gain / np.where(prize > EPS, prize, 1.0), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--debug", action="store_true", help="two cells, 50 draws")
    args = ap.parse_args()
    run = runmeta.new_run("objective_swap", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    rng = np.random.default_rng(0)
    t0 = time.time()

    cells = build_cells(splits)
    if args.debug:
        cells = cells[:1] + cells[-1:]
    print(f"  {len(cells)} cells built in {time.time() - t0:.0f}s", flush=True)

    rows = []
    for c in cells:
        d = c["d"]
        te = (d.split == "test").to_numpy()
        v = c["v"][te]
        units = d.unit.to_numpy()[te]
        n = len(v)
        ks = t92.quota_k(n)
        key = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"])
        cheap_total = float(np.abs(d[c["cheap"]].to_numpy(float)[te]).sum())

        # the value vector of each objective, restricted to the test split
        values, val_na = {"E_dec": v}, {}
        for name, colname in OBJECTIVES:
            s = c["scores"].get(colname)
            if s is None:
                val_na[name] = c["na"].get(colname, "not defined for this track")
            else:
                values[name] = s[te]

        # scores of the pool on the test split; coverage is recorded, never silently filled
        series, avail, cover = {}, {}, {}
        for s in POOL:
            if s == "random":
                series[s] = None
                cover[s] = 1.0
                continue
            x = c["scores"].get(s)
            if x is None:
                avail[s] = c["na"].get(s, "not defined for this track")
                continue
            xt = x[te]
            ok = np.isfinite(xt)
            cover[s] = float(ok.mean())
            if not ok.any():
                avail[s] = "no cached score covers this cell's test inputs"
                continue
            series[s] = np.where(ok, xt, -np.inf)          # uncovered inputs rank below every covered one

        # ---- section: score ----------------------------------------------------------------
        etas = {}
        for obj, val in values.items():
            n_aff = int((np.abs(val) > EPS).sum())
            prize = t92.topk_expect(val, [val], ks)[0][0]
            for s in POOL:
                base = {**key, "section": "score", "objective": obj, "signal": s, "tier": TIER_OF[s],
                        "n_units": int(len(np.unique(units))), "n_frames": n, "n_affected": n_aff}
                if s not in series:
                    rows.append({**base, "available": False, "undefined_reason": avail[s]})
                    continue
                gain, eta = eta_of(val, series[s], ks, prize)
                etas[(obj, s)] = eta
                for qi, q in enumerate(QUOTAS):
                    why = ""
                    if n_aff < MIN_AFFECTED:
                        why = f"near-zero oracle: {n_aff} affected inputs on this split (< {MIN_AFFECTED})"
                    elif prize[qi] <= EPS:
                        why = "oracle prize is zero at this quota"
                    rows.append({**base, "available": True, "quota": q, "k": int(ks[qi]),
                                 "coverage": cover[s], "gain": float(gain[qi]), "prize": float(prize[qi]),
                                 "eta": float(eta[qi]), "ndg_defined": why == "", "undefined_reason": why})
        for name in val_na:
            rows.append({**key, "section": "score", "objective": name, "available": False,
                         "undefined_reason": val_na[name]})

        # ---- section: compare --------------------------------------------------------------
        # Regret is measured in decision value, so a cell the benchmark calls undefined under E_dec carries no
        # usable regret even when the perception objective is well defined there.  Flag it on every compare row.
        n_aff_dec = int((np.abs(v) > EPS).sum())
        prize_v = t92.topk_expect(v, [v], ks)[0][0]

        # every signal that any comparison selects, bootstrapped once per cell
        picks, compare = {}, []
        for obj, _ in OBJECTIVES:
            if obj not in values:
                continue
            for pool_name, pool in (("headline", HEADLINE), ("all", POOL)):
                have = [s for s in pool if s in series]
                for qi, q in enumerate(QUOTAS):
                    dec = [etas[("E_dec", s)][qi] for s in have]
                    per = [etas[(obj, s)][qi] for s in have]
                    if not np.isfinite(dec).any() or not np.isfinite(per).any():
                        continue
                    a_dec = have[int(np.nanargmax(dec))]
                    a_per = have[int(np.nanargmax(per))]
                    tau, p = kendalltau(dec, per, nan_policy="omit")
                    compare.append(dict(objective=obj, pool=pool_name, qi=qi, quota=q, have=have,
                                        argmax_dec=a_dec, argmax_perc=a_per, tau=tau, tau_p=p,
                                        eta_dec_of_dec=etas[("E_dec", a_dec)][qi],
                                        eta_dec_of_perc=etas[("E_dec", a_per)][qi]))
                    for s in (a_dec, a_per, "random"):
                        picks.setdefault(s, None)

        boot = {}
        if picks:
            uniq = np.unique(units)
            idx = [np.flatnonzero(units == u) for u in uniq]
            names = [s for s in picks if s in series]
            draws = {s: np.full((args.nboot, len(ks)), np.nan) for s in names}
            kept = np.zeros(len(ks), int)
            prize_dec = t92.topk_expect(v, [v], ks)[0][0]
            for b in range(args.nboot):
                t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
                vt = v[t]
                kt = t92.quota_k(len(t))
                pz = t92.topk_expect(vt, [vt], kt)[0][0]
                kept += pz > np.maximum(EPS, 0.25 * prize_dec)
                for s in names:
                    g = (kt / len(t) * vt.sum() if series[s] is None
                         else t92.topk_expect(series[s][t], [vt], kt)[0][0])
                    draws[s][b] = g                        # raw realised decision value, every draw kept
            boot = dict(draws=draws, kept=kept)

        for cmp in compare:
            qi = cmp["qi"]
            a_dec, a_per = cmp["argmax_dec"], cmp["argmax_perc"]
            r = {**key, "section": "compare", "objective": cmp["objective"], "pool": cmp["pool"],
                 "quota": cmp["quota"], "n_methods": len(cmp["have"]), "kendall_tau": cmp["tau"],
                 "kendall_p": cmp["tau_p"], "argmax_dec": a_dec, "argmax_perc": a_per,
                 "argmax_differs": a_dec != a_per, "eta_dec_of_dec": cmp["eta_dec_of_dec"],
                 "eta_dec_of_perc": cmp["eta_dec_of_perc"],
                 "regret_ndg": cmp["eta_dec_of_perc"] - cmp["eta_dec_of_dec"],
                 "perc_winner_worse_than_random": bool(cmp["eta_dec_of_perc"] < etas[("E_dec", "random")][qi]),
                 "cheap_loss_total": cheap_total,
                 # regret is measured in decision value: false where the benchmark calls this cell undefined
                 # under E_dec, even though the perception objective is well defined on the same inputs
                 "e_dec_defined": bool(n_aff_dec >= MIN_AFFECTED and prize_v[qi] > EPS),
                 "n_affected_dec": n_aff_dec}
            if boot and a_dec in boot["draws"] and a_per in boot["draws"]:
                dr = boot["draws"][a_per][:, qi] - boot["draws"][a_dec][:, qi]
                lo, hi = (float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5)))
                r.update(regret_gain=float(np.mean(dr)), regret_gain_lo=lo, regret_gain_hi=hi,
                         regret_share_cheap=float(np.mean(dr) / cheap_total) if cheap_total > EPS else np.nan,
                         regret_share_lo=lo / cheap_total if cheap_total > EPS else np.nan,
                         regret_share_hi=hi / cheap_total if cheap_total > EPS else np.nan,
                         n_draws=int(args.nboot), n_draws_kept_under_official_filter=int(boot["kept"][qi]))
            rows.append(r)
        print(f"  {c['track']:8s} {c['system']:10s} {c['target']:8s} "
              f"{len(series)}/{len(POOL)} signals, {len(compare)} comparisons  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    out = pd.DataFrame(rows)

    # ---- section: sanity: E_dec must reproduce the official held-out nDG ----------------------
    off = []
    for f, real in (("benchmark_table.csv", False), ("benchmark_table_routers.csv", False),
                    ("benchmark_table_nuplan_real.csv", True)):
        p = FINAL / f
        if not p.exists():
            continue
        o = pd.read_csv(p)
        o = o[o.split == "test"]
        if not real:
            o = o[o.track != "nuPlan"]                      # nuPlan's official cells are the real-perception ones
        off.append(o[["track", "geometry", "system", "target", "signal", "quota", "eta"]])
    off = pd.concat(off, ignore_index=True).rename(columns={"eta": "eta_official"})
    off["geometry"] = off.geometry.fillna("n/a")
    mine = out[(out.section == "score") & (out.objective == "E_dec") & out.available.fillna(False)]
    m = mine.merge(off, on=["track", "geometry", "system", "target", "signal", "quota"], how="inner")
    m["abs_diff"] = (m.eta - m.eta_official).abs()
    bad = m[m.abs_diff.round(3) > 0]
    for _, r in m.iterrows():
        rows.append({"section": "sanity", "track": r.track, "geometry": r.geometry, "system": r.system,
                     "target": r.target, "signal": r.signal, "quota": r.quota, "objective": "E_dec",
                     "eta": r.eta, "eta_official": r.eta_official, "abs_diff": r.abs_diff,
                     "matches_to_3dp": bool(round(r.abs_diff, 3) == 0)})
    # Rows whose official counterpart is NaN are cells the benchmark declares undefined; a NaN comparison fails
    # silently, so they are counted separately instead of being folded into the match rate.
    comparable = m[m.eta.notna() & m.eta_official.notna()]
    print(f"  sanity: {len(comparable) - len(bad)}/{len(comparable)} E_dec values reproduce the official held-out "
          f"nDG to 3 dp; {len(m) - len(comparable)} further rows have no official value (undefined cells)",
          flush=True)
    if len(bad):
        print(bad[["track", "system", "target", "signal", "quota", "eta", "eta_official", "abs_diff"]]
              .head(12).to_string(index=False), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(run / "objective_swap.csv", index=False)
    if args.debug:                                          # a two-cell smoke never lands in results/final
        print(f"  debug: wrote {run / 'objective_swap.csv'} only ({len(out)} rows)", flush=True)
    else:
        dst = FINAL / "objective_swap.csv"
        out.to_csv(dst, index=False)
        print(f"  wrote {dst} ({len(out)} rows) in {time.time() - t0:.0f}s", flush=True)
    assert len(bad) == 0 or args.debug, "E_dec does not reproduce the official held-out nDG"


if __name__ == "__main__":
    main()

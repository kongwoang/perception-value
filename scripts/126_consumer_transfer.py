#!/usr/bin/env python
"""Task 13 Part A: the consumer transfer matrix -- how much of an allocator's value survives a change of consumer.

Pre-registered in RESEARCH_LOG.md (Task 13 Part A), committed before this script was run.  Nothing official is
written:

  results/final/consumer_transfer.csv    one row per group x trained-for x evaluated-on x signal x quota,
                                         plus the per-group summary rows

Entry (A, B) is the allocator trained for consumer A, evaluated against consumer B's decision value on B's frozen
test split, with the benchmark's exact tie expectation.  R1 scores are the cached ones; the identifier columns are
detected from the file.  Gates are refit with the official hyperparameters on the train + val units of the consumer
they are trained for, which is the official protocol and keeps every evaluated test unit outside the fitting set.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta                                                # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


t120 = _load("t120", "120_nuplan_real_allocation.py")
t92 = t120.t92

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS, QUOTAS = t92.EPS, t92.QUOTAS
R1 = ["R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
GATES = ["gate_ridge", "gate_gbm"]
SIGNALS = GATES + R1
NBOOT = 1000
R1_RUN = Path(sorted(glob.glob(str(RAW / "*_routers_r1")))[-1])
NR_RUN = RAW / "20260915_033311_nuplan_real_allocation"
NUPLAN_CONSUMERS = [("pdm_closed", "safety"), ("pdm_closed", "scalar_J"), ("idm", "scalar_J")]


def id_columns(files):
    """Detect the identifier columns of a cached score file instead of assuming their names."""
    for a, b in (("seq", "frame"), ("scenario", "iteration")):
        if a in files and b in files:
            return a, b
    raise SystemExit(f"no identifier columns found among {files}")


def gate_models(d, fcols, v, fit):
    """The official gate fit (92's hyperparameters and seed), returned as fitted models."""
    X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                      nan=0.0, posinf=1e6, neginf=-1e6)
    out = {}
    for name, mdl in (("gate_ridge", "linear"), ("gate_gbm", "gbm")):
        m = predict.make_model(mdl, "reg", 0)
        m.fit(X[fit], v[fit])
        out[name] = m
    return out


def feature_matrix(d, fcols):
    return np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                         nan=0.0, posinf=1e6, neginf=-1e6)


def build_cells(splits, nr_fcols):
    cells = []
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            f = R1_RUN / f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz"
            z = np.load(f, allow_pickle=False)
            ida, idb = id_columns(list(z.files))
            keys = pd.DataFrame({ida: z[ida].astype(str), idb: z[idb].astype(int)})
            assert np.array_equal(z["V"], v), f
            cells.append(dict(group=f"{c['track']} {c['geometry']}", track=c["track"], geometry=c["geometry"],
                              system=c["system"], target=c["target"], dataset=c["track"], d=d, v_all=v,
                              fcols=c["fcols"], cheap=c["cheap"], id_cols=(ida, idb),
                              keys=d[[ida, idb]].astype({ida: str, idb: int}).reset_index(drop=True),
                              r1={r: np.asarray(z[r], float) for r in R1}, r1_keys=keys))
    for c in t120.cells(splits):
        if (c["system"], c["target"]) not in NUPLAN_CONSUMERS:
            continue
        d = c["d"].reset_index(drop=True)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(NR_RUN / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        ida, idb = id_columns(list(z.files))
        assert np.array_equal(z["V"], v)
        cells.append(dict(group="nuPlan real perception", track="nuPlan", geometry="n/a", system=c["system"],
                          target=c["target"], dataset="nuPlan", d=d, v_all=v, fcols=nr_fcols, cheap=c["cheap"],
                          id_cols=(ida, idb),
                          keys=d[[ida, idb]].astype({ida: str, idb: int}).reset_index(drop=True),
                          r1={r: np.asarray(z[r], float) for r in R1},
                          r1_keys=pd.DataFrame({ida: z[ida].astype(str), idb: z[idb].astype(int)})))
    return cells


def official_diagonal():
    out = {}
    for name, real in (("benchmark_table.csv", False), ("benchmark_table_routers.csv", False),
                       ("benchmark_table_nuplan_real.csv", True)):
        x = pd.read_csv(FINAL / name)
        x["geometry"] = x.geometry.fillna("n/a")
        x = x[(x.split == "test") & x.quota.notna()]
        x = x[x.track == "nuPlan"] if real else x[x.track != "nuPlan"]
        for r in x.itertuples():
            if getattr(r, "available", True) is False:
                continue
            out[(r.track, r.geometry, r.system, r.target, r.signal, round(r.quota, 2))] = r.eta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    args = ap.parse_args()
    run = runmeta.new_run("consumer_transfer", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t120.register_features()
    official = official_diagonal()
    cells = build_cells(splits, nr_fcols)
    print(f"  {len(cells)} consumer cells in {len(set(c['group'] for c in cells))} groups", flush=True)

    for c in cells:
        te = (c["d"].split == "test").to_numpy()
        c["test"] = te
        c["v"] = c["v_all"][te]
        c["n"] = int(te.sum())
        c["units"] = c["d"].unit.astype(str).to_numpy()[te]
        c["uniq"] = np.array(sorted(set(c["units"])))
        c["idx"] = [np.flatnonzero(c["units"] == u) for u in c["uniq"]]
        c["upos"] = {u: i for i, u in enumerate(c["uniq"])}
        c["ks"] = [max(int(round(q * c["n"])), 1) for q in QUOTAS]
        c["prize"] = np.asarray(t92.topk_expect(c["v"], [c["v"]], c["ks"])[0][0], float)
        c["tot_cheap"] = float(c["d"][c["cheap"]].to_numpy(float)[te].sum())
        c["n_aff"] = int((np.abs(c["v"]) > EPS).sum())
        c["undef"] = (f"near-zero oracle: {c['n_aff']} affected states (< {t120.MIN_AFFECTED})"
                      if c["track"] == "nuPlan" and c["n_aff"] < t120.MIN_AFFECTED else "")
        c["X"] = feature_matrix(c["d"], c["fcols"])
        c["fit"] = c["d"].split.isin(["train", "val"]).to_numpy()

    # scores of every (trained-for A, evaluated-on B) pair, on B's test frames
    series, rows, point = {}, [], {}
    for group in sorted({c["group"] for c in cells}):
        gcells = [c for c in cells if c["group"] == group]
        models = {(c["system"], c["target"]): gate_models(c["d"], c["fcols"], c["v_all"], c["fit"]) for c in gcells}
        for A in gcells:
            akey = (A["system"], A["target"])
            for B in gcells:
                ida, idb = B["id_cols"]
                left = B["keys"].loc[B["test"]].reset_index(drop=True)
                cov = {}
                sc = {}
                for s in GATES:
                    sc[s] = models[akey][s].predict(B["X"])[B["test"]]
                    cov[s] = 1.0
                a1 = A["r1_keys"].copy()
                for r in R1:
                    a1[r] = A["r1"][r]
                j = left.merge(a1, on=[ida, idb], how="left", validate="one_to_one")
                for r in R1:
                    col = j[r].to_numpy(float)
                    cov[r] = float(np.isfinite(col).mean())
                    fill = np.nanmin(col) if np.isfinite(col).any() else 0.0
                    sc[r] = np.where(np.isfinite(col), col, fill - 1.0)
                for s in SIGNALS:
                    g = np.asarray(t92.topk_expect(sc[s], [B["v"]], B["ks"])[0][0], float)
                    point[(group, akey, (B["system"], B["target"]), s)] = g
                    series[(group, akey, (B["system"], B["target"]), s)] = sc[s]
                    for qi, q in enumerate(QUOTAS):
                        rnd = B["ks"][qi] / B["n"] * B["v"].sum()
                        ndg = (g[qi] / B["prize"][qi]) if (B["prize"][qi] > EPS and not B["undef"]) else np.nan
                        ref = official.get((B["track"], B["geometry"], B["system"], B["target"], s, round(q, 2)))
                        diag = akey == (B["system"], B["target"])
                        rows.append({"group": group, "track": B["track"], "geometry": B["geometry"],
                                     "trained_for": f"{A['system']}/{A['target']}",
                                     "evaluated_on": f"{B['system']}/{B['target']}", "diagonal": diag,
                                     "signal": s, "quota": q, "coverage": cov[s], "n_test_frames": B["n"],
                                     "n_test_units": len(B["uniq"]), "gain": float(g[qi]),
                                     "gain_random": float(rnd), "delta_vs_random": float(g[qi] - rnd),
                                     "gain_share_all_cheap": float(g[qi]) / B["tot_cheap"] if B["tot_cheap"] > EPS else np.nan,
                                     "ndg": ndg, "official_ndg": ref if diag else np.nan,
                                     "matches_3dp": (bool(round(float(ndg), 3) == round(float(ref), 3))
                                                     if diag and ref is not None and not pd.isna(ndg) and not pd.isna(ref) else None),
                                     "undefined_reason": B["undef"]})
        print(f"  group {group}: {len(gcells)}x{len(gcells)} entries", flush=True)

    # transfer regret against the diagonal of the same (B, signal, quota)
    df = pd.DataFrame(rows)
    diag = df[df.diagonal][["group", "evaluated_on", "signal", "quota", "ndg"]].rename(columns={"ndg": "ndg_diag"})
    df = df.merge(diag, on=["group", "evaluated_on", "signal", "quota"], how="left")
    df["transfer_regret"] = df.ndg - df.ndg_diag

    # paired bootstrap, every draw kept, and the 25% prize filter beside it
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units_of = {ds: sorted(set().union(*[set(c["uniq"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    bcell = {(c["group"], (c["system"], c["target"])): c for c in cells}
    draws = {k: np.full((args.nboot, len(QUOTAS)), np.nan) for k in series}
    keep = {(c["group"], (c["system"], c["target"])): np.zeros((args.nboot, len(QUOTAS)), bool) for c in cells}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for b in range(args.nboot):
        pick = {ds: rng.integers(0, len(units_of[ds]), len(units_of[ds])) for ds in datasets}
        resample = {}
        for c in cells:
            us = [u for u in (units_of[c["dataset"]][i] for i in pick[c["dataset"]]) if u in c["upos"]]
            if not us:
                continue
            t = np.concatenate([c["idx"][c["upos"][u]] for u in us])
            vt = c["v"][t]
            ks = [max(int(round(q * len(t))), 1) for q in QUOTAS]
            pz = np.asarray(t92.topk_expect(vt, [vt], ks)[0][0], float)
            keep[(c["group"], (c["system"], c["target"]))][b] = pz > np.maximum(EPS, 0.25 * c["prize"])
            rnd = np.array([k / len(t) * vt.sum() for k in ks])
            resample[(c["group"], (c["system"], c["target"]))] = (t, vt, ks, rnd)
        for key, sc in series.items():
            group, akey, bkey, s = key
            r = resample.get((group, bkey))
            if r is None:
                continue
            t, vt, ks, rnd = r
            draws[key][b] = np.asarray(t92.topk_expect(sc[t], [vt], ks)[0][0], float) - rnd
        if (b + 1) % 200 == 0:
            print(f"  bootstrap {b + 1}/{args.nboot} [{time.time() - t0:.0f}s]", flush=True)

    new_cols = ["delta_lo_all_draws", "delta_hi_all_draws", "beats_random_all_draws", "delta_lo_filtered",
                "beats_random_filtered", "draws_dropped_by_filter", "delta_share_lo_all_draws"]
    buf = {c_: np.full(len(df), np.nan) for c_ in new_cols}
    idx = {(r.group, r.trained_for, r.evaluated_on, r.signal, round(r.quota, 2)): i for i, r in enumerate(df.itertuples())}
    for key, x in draws.items():
        group, akey, bkey, s = key
        c = bcell[(group, bkey)]
        for qi, q in enumerate(QUOTAS):
            i = idx.get((group, f"{akey[0]}/{akey[1]}", f"{bkey[0]}/{bkey[1]}", s, round(q, 2)))
            if i is None:
                continue
            col = x[:, qi]
            lo, hi = t92.ci(col)
            sub = col[keep[(group, bkey)][:, qi]]
            flo, fhi = t92.ci(sub)
            vals = [lo, hi, float(lo > 0), flo, float(flo > 0), float(args.nboot - len(sub)),
                    lo / c["tot_cheap"] if c["tot_cheap"] > EPS else np.nan]
            for c_, val in zip(new_cols, vals):
                buf[c_][i] = val

    for c_ in new_cols:
        df[c_] = buf[c_]
    for c_ in ("beats_random_all_draws", "beats_random_filtered"):
        df[c_] = df[c_].map({1.0: True, 0.0: False})

    # per-group summary
    summary = []
    for group, x in df[np.isclose(df.quota, 0.2)].groupby("group"):
        off = x[~x.diagonal]
        summary.append({"group": group, "track": x.track.iloc[0], "geometry": x.geometry.iloc[0],
                        "trained_for": "SUMMARY", "evaluated_on": "off-diagonal, 20%", "signal": "all",
                        "quota": 0.2, "n_entries": len(off),
                        "median_transfer_regret": float(off.transfer_regret.median()),
                        "off_diag_beating_random_all_draws": int(off.beats_random_all_draws.fillna(False).sum()),
                        "off_diag_beating_random_filtered": int(off.beats_random_filtered.fillna(False).sum()),
                        "diag_beating_random_all_draws": int(x[x.diagonal].beats_random_all_draws.fillna(False).sum()),
                        "n_diag_entries": int(x.diagonal.sum())})
    df = pd.concat([df, pd.DataFrame(summary)], ignore_index=True)

    for out in (FINAL / "consumer_transfer.csv", run / "consumer_transfer.csv"):
        df.to_csv(out, index=False)
    print(f"  wrote {FINAL / 'consumer_transfer.csv'} ({len(df)} rows)")
    d = df[df.diagonal == True]                                                  # noqa: E712
    ok = d.matches_3dp.dropna()
    print(f"  sanity: the diagonal reproduces the official nDG in {int(ok.sum())}/{len(ok)} comparisons")
    bad = d[d.matches_3dp == False]                                              # noqa: E712
    if len(bad):
        print(bad[["group", "evaluated_on", "signal", "quota", "ndg", "official_ndg"]].head(10).round(4).to_string(index=False))
    for s in summary:
        print(f"  {s['group']:24s} median off-diagonal regret {s['median_transfer_regret']:+.3f}; "
              f"{s['off_diag_beating_random_all_draws']}/{s['n_entries']} off-diagonal entries beat random")


if __name__ == "__main__":
    main()

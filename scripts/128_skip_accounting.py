#!/usr/bin/env python
"""Task 16 Part A: skipping cost accounting for the pixel router, beside the shipped cascade.

Pre-registered in RESEARCH_LOG.md (Task 16 Part A), committed before this script was written or run.

The benchmark charges a cascade: every input pays the allocator and CHEAP, and an escalated input also runs
FULL, so the mean cost per input is Cs + Cc + f*Cf.  A skipping router decides before CHEAP runs, and an
escalated input runs FULL instead of CHEAP: Cs + Cc + f*(Cf - Cc).  Only R2 is eligible: it reads the raw
camera frame, which exists before CHEAP.  R1 and the gates read CHEAP's detections.  Decision values are the
same under both designs, because escalation replaces CHEAP's output with FULL's either way; only the share an
allocator can afford changes.  Nothing is retrained: R2's cached scores are used as they are.  No official
result file is written:

  results/final/skip_accounting.csv
    section=validation  the cascade branch against every shipped R2 budget row (share exactly, eta to 1e-9)
    section=shares      constants, and the cascade and skipping shares per track, unit and budget
    section=evaluation  every row with a positive skipping share: nDG against the oracle escalating the same k,
                        random at the same share, a paired cluster bootstrap with every draw kept

The script stops before evaluating anything if the validation fails.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                          # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402

FINAL = Path(RESULTS) / "final"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t92 = _load("t92", "92_benchmark_table.py")
t93 = _load("t93", "93_budget_allocation.py")

EPS, QUOTAS = t92.EPS, t92.QUOTAS
SIGNAL = "R2_cnn_clf"
KEY = ["track", "geometry", "system", "target"]
INELIGIBLE = {
    "R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf":
        "reads CHEAP's detection list (103_routers_r1.py det_list_features), which exists only after CHEAP has run",
    "gate_ridge, gate_gbm": "reads features computed from CHEAP's detections",
}


def shares(Cc, Cf, Cs, q):
    """Budget, cascade share (93:342) and skipping share for one budget level."""
    B = Cc + q * Cf                                                              # 93:317
    return B, max((B - Cc - Cs) / Cf, 0.0), min(max((B - Cc - Cs) / (Cf - Cc), 0.0), 1.0)


def k_of(frac, n):
    return int(np.floor(frac * n + 1e-9))                                        # 93:321


def gains(score, v, ks):
    """Exact tie expectation at several k; k = 0 gains nothing; score None is random."""
    ks = np.asarray(ks, int)
    if score is None:
        return ks / len(v) * v.sum(), np.full(len(ks), np.nan)
    (g,), tie = t92.topk_expect(score, [v], np.maximum(ks, 1))
    return np.where(ks > 0, g, 0.0), np.where(ks > 0, tie, np.nan)


def build_cells(splits):
    routers = t93.load_router_scores()
    cells = []
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            d = c["d"]
            m = (d.split == "test").to_numpy()
            key = (c["track"], c["geometry"], c["system"], c["target"])
            s, cov = None, np.nan
            for keycols, rdf in routers.get(key, []):                           # 93:303-309
                if SIGNAL not in rdf.columns:
                    continue
                left = d.loc[m, list(keycols)].copy()
                for kc in keycols:
                    left[kc] = left[kc].astype(str) if kc in ("seq", "scenario") else left[kc].astype(int)
                j = left.merge(rdf[list(keycols) + [SIGNAL]], on=list(keycols), how="left", validate="one_to_one")
                cov = float(j[SIGNAL].notna().mean())
                s = j[SIGNAL].fillna(-np.inf).to_numpy(float)
            assert s is not None, f"no R2 scores for {key}"
            cells.append(dict(zip(KEY, key), v=(d[c["cheap"]] - d[c["full"]]).to_numpy(float)[m], s=s,
                              coverage=cov, units=d.unit.to_numpy()[m],
                              all_cheap=float(d[c["cheap"]].to_numpy(float)[m].sum())))
    return cells


def constants(ov, track, unit):
    cost = t93.profile_costs(track)
    o_ms, o_mj, _ = t93.signal_overhead(ov, track, SIGNAL)
    return cost["cheap"][unit], cost["640"][unit], (o_ms if unit == "ms" else o_mj)


def pct(x, p):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if len(x) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--debug", action="store_true", help="write only to the run directory")
    args = ap.parse_args()
    run = runmeta.new_run("skip_accounting", vars(args))
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    # pandas' default float parser can land one ULP off the written value; the gate compares exactly, so read exactly
    shipped = pd.read_csv(FINAL / "benchmark_budget_routers.csv", float_precision="round_trip")
    shipped = shipped[shipped.signal == SIGNAL].reset_index(drop=True)
    t0 = time.time()
    cells = build_cells(splits)
    print(f"  {len(cells)} R2 cells built in {time.time() - t0:.0f}s; test coverage "
          f"{sorted(set(round(c['coverage'], 4) for c in cells))}", flush=True)

    # ---- section: validation -------------------------------------------------------------------
    rows, bad_frac, bad_eta = [], 0, 0
    for c in cells:
        v, s, n = c["v"], c["s"], len(c["v"])
        for unit in ("ms", "mJ"):
            Cc, Cf, Cs = constants(ov, c["track"], unit)
            for q in QUOTAS:
                B, fc, _ = shares(Cc, Cf, Cs, q)
                r = shipped[(shipped.track == c["track"]) & (shipped.geometry == c["geometry"])
                            & (shipped.system == c["system"]) & (shipped.target == c["target"])
                            & (shipped.unit == unit) & np.isclose(shipped.budget_level, q)]
                assert len(r) == 1, (c["track"], c["system"], c["target"], unit, q, len(r))
                r = r.iloc[0]
                prize = gains(v, v, [k_of(q, n)])[0][0]                               # 93:328
                eta = gains(s, v, [k_of(fc, n)])[0][0] / prize if prize > EPS else np.nan
                frac_equal = fc == r.escalated_frac
                eta_ok = bool((np.isnan(eta) and np.isnan(r.eta))
                              or np.isclose(eta, r.eta, rtol=1e-9, atol=1e-12))
                bad_frac += not frac_equal
                bad_eta += not eta_ok
                rows.append({"section": "validation", **{k: c[k] for k in KEY}, "unit": unit, "budget_level": q,
                             "budget": B, "budget_shipped": r.budget_per_frame, "share_cascade": fc,
                             "share_cascade_shipped": r.escalated_frac, "share_equal": bool(frac_equal),
                             "eta_cascade": eta, "eta_cascade_shipped": r.eta, "eta_match": eta_ok})
    n_val = len(rows)
    print(f"  validation: cascade share equal in {n_val - bad_frac}/{n_val} shipped R2 rows; "
          f"eta within 1e-9 in {n_val - bad_eta}/{n_val}", flush=True)
    for track in ("nuScenes", "KITTI"):
        for q in (0.2, 0.5):
            med = np.median([x["share_cascade"] for x in rows if x["track"] == track and x["unit"] == "ms"
                             and np.isclose(x["budget_level"], q)])
            print(f"    {track:8s} ms budget {q:.0%}: cascade share {med:.1%}", flush=True)
    if bad_frac or bad_eta:
        pd.DataFrame(rows).to_csv(run / "skip_accounting_validation_FAILED.csv", index=False)
        raise SystemExit("validation failed: the cascade branch does not reproduce the shipped R2 rows; stopping")

    # ---- section: shares ---------------------------------------------------------------------------
    for track in ("nuScenes", "KITTI"):
        for unit in ("ms", "mJ"):
            Cc, Cf, Cs = constants(ov, track, unit)
            for q in QUOTAS:
                B, fc, fs = shares(Cc, Cf, Cs, q)
                rows.append({"section": "shares", "track": track, "unit": unit, "budget_level": q, "budget": B,
                             "Cc": Cc, "Cf": Cf, "Cs": Cs, "signal": SIGNAL, "eligible": True,
                             "share_cascade": fc, "share_skipping": fs})
    for sig, why in INELIGIBLE.items():
        rows.append({"section": "shares", "signal": sig, "eligible": False, "ineligible_reason": why})

    # ---- section: evaluation -------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    n_eval = 0
    for c in cells:
        v, s, units, n = c["v"], c["s"], c["units"], len(c["v"])
        todo = []
        for unit in ("ms", "mJ"):
            Cc, Cf, Cs = constants(ov, c["track"], unit)
            for q in QUOTAS:
                B, fc, fs = shares(Cc, Cf, Cs, q)
                if fs > 0:
                    todo.append((unit, q, B, Cc, Cf, Cs, fc, fs))
        if not todo:
            continue
        uniq = np.unique(units)
        idx = [np.flatnonzero(units == u) for u in uniq]
        fss = np.array([t[7] for t in todo])
        ks = np.array([k_of(f, n) for f in fss])
        g_r2, tie_r2 = gains(s, v, ks)
        g_rand = gains(None, v, ks)[0]
        g_or = gains(v, v, ks)[0]
        D = {x: np.full((args.nboot, len(todo)), np.nan) for x in ("r2", "rand", "or")}
        for b in range(args.nboot):
            t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
            vt = v[t]
            kt = np.array([k_of(f, len(t)) for f in fss])
            D["r2"][b] = gains(s[t], vt, kt)[0]
            D["rand"][b] = gains(None, vt, kt)[0]
            D["or"][b] = gains(vt, vt, kt)[0]
        n_pos, n_nonneg = int((v > EPS).sum()), int((v > -EPS).sum())
        capacity = float(np.maximum(v, 0.0).sum())
        shipped_c = shipped[(shipped.track == c["track"]) & (shipped.geometry == c["geometry"])
                            & (shipped.system == c["system"]) & (shipped.target == c["target"])]
        for i, (unit, q, B, Cc, Cf, Cs, fc, fs) in enumerate(todo):
            k = int(ks[i])
            regime = ("share_below_positives" if k < n_pos else
                      "share_above_nonnegative" if k > n_nonneg else "at_capacity")
            ratio = g_or[i] / capacity if capacity > EPS else np.nan
            if regime == "at_capacity":
                assert ratio >= 1 - 1e-9, (c["track"], c["system"], unit, q, ratio)
            else:
                assert ratio < 1 + 1e-12, (c["track"], c["system"], unit, q, ratio)
            diff = D["r2"][:, i] - D["rand"][:, i]
            ok = D["or"][:, i] > EPS
            den = np.where(ok, D["or"][:, i], np.nan)
            ndg_r2_t, ndg_diff_t = D["r2"][:, i] / den, diff / den
            lo, hi = pct(diff, 2.5), pct(diff, 97.5)
            cas = shipped_c[(shipped_c.unit == unit) & np.isclose(shipped_c.budget_level, q)].iloc[0]
            rows.append({
                "section": "evaluation", **{kk: c[kk] for kk in KEY}, "signal": SIGNAL, "unit": unit,
                "budget_level": q, "budget": B, "Cc": Cc, "Cf": Cf, "Cs": Cs,
                "share_cascade": fc, "share_skipping": fs, "k": k, "n_test": n,
                "n_units": int(len(uniq)), "coverage": c["coverage"],
                "n_positive": n_pos, "n_nonnegative": n_nonneg, "share_positive": n_pos / n,
                "share_nonnegative": n_nonneg / n, "share_skipping_exceeds_nonnegative": bool(fs > n_nonneg / n),
                "regime": regime, "denominator": "oracle escalating the same k (budget-constrained)",
                "oracle_same_k": float(g_or[i]), "oracle_capacity": capacity, "oracle_same_k_over_capacity": ratio,
                "gain_r2": float(g_r2[i]), "gain_random": float(g_rand[i]), "tie_frac_r2": float(tie_r2[i]),
                "ndg_r2": float(g_r2[i] / g_or[i]) if g_or[i] > EPS else np.nan,
                "ndg_random": float(g_rand[i] / g_or[i]) if g_or[i] > EPS else np.nan,
                "ndg_minus_random": float((g_r2[i] - g_rand[i]) / g_or[i]) if g_or[i] > EPS else np.nan,
                "all_cheap_loss": c["all_cheap"],
                "gain_share_all_cheap_r2": float(g_r2[i] / c["all_cheap"]),
                "gain_share_all_cheap_random": float(g_rand[i] / c["all_cheap"]),
                "n_draws": int(args.nboot), "n_draws_oracle_nonpositive": int((~ok).sum()),
                "gain_minus_random_mean": float(np.mean(diff)), "gain_minus_random_lo": lo,
                "gain_minus_random_hi": hi,
                "ndg_r2_lo": pct(ndg_r2_t, 2.5), "ndg_r2_hi": pct(ndg_r2_t, 97.5),
                "ndg_minus_random_lo": pct(ndg_diff_t, 2.5), "ndg_minus_random_hi": pct(ndg_diff_t, 97.5),
                "p_le_random": float(np.mean(diff <= 0)),
                "beats_random": bool(lo > 0), "loses_to_random": bool(hi < 0),
                "cascade_eta_shipped": float(cas.eta), "cascade_minus_random_lo_shipped": float(cas.minus_random_lo),
                "cascade_minus_random_hi_shipped": float(cas.minus_random_hi)})
            n_eval += 1
            print(f"  {c['track']:8s} {c['geometry']:6s} {c['system']:9s} {unit:2s} {q:.0%}: share {fs:6.1%} "
                  f"(cascade {fc:5.1%}), nDG R2 {rows[-1]['ndg_r2']:+.3f} vs random {rows[-1]['ndg_random']:+.3f}, "
                  f"gain diff [{lo:+.3f}, {hi:+.3f}], {regime}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(run / "skip_accounting.csv", index=False)
    ev = out[out.section == "evaluation"]
    print(f"  evaluation rows {n_eval}; beats random {int(ev.beats_random.sum())}, "
          f"loses to random {int(ev.loses_to_random.sum())}", flush=True)
    if args.debug:
        print(f"  debug: wrote {run / 'skip_accounting.csv'} only", flush=True)
    else:
        out.to_csv(FINAL / "skip_accounting.csv", index=False)
        print(f"  wrote {FINAL / 'skip_accounting.csv'} ({len(out)} rows) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

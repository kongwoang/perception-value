#!/usr/bin/env python
"""Phase 0F Stage 1 -- do published planning-aware perception metrics already solve
compute allocation?

Joins per-sample PKL (CVPR 2020) and TIP (ICML 2023) gains onto the nuScenes decision
table from Phase 0E and scores every allocation signal by the same eta as the rest of the
project: the share of the decision-value oracle's achievable cost reduction that a signal
captures at a compute quota.

Reads the PKL/TIP CSVs written by 61_run_planning_metric.py (chunked runs are
concatenated) and writes results/final/phase0f_planning_metric_eta.csv.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rap import budget, percep_metrics as PM, predict, runmeta                  # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

from importlib import import_module                                             # noqa: E402
_pm = import_module("50_percep_metrics")

QUOTAS = [0.10, 0.20, 0.30, 0.50]
# Planner C is PKL's own published planner used as the downstream decision maker: the cost of
# a mode is the metre displacement between the path that planner intends given that mode's
# detections and the path it intends given ground-truth boxes (see 66_planner_c_pkl_planner).
# It is a third task rather than a separate analysis so that every signal, quota, bootstrap
# and ranking statistic is computed by exactly the same code as for the two hand-written
# planners.
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}
PLANNER_C_TASK = {"plannerC_path_dev": ("JC_cheap", "JC_full")}
# Phase 0G: the same PKL planner scored against the REAL future trajectory instead of against
# its own ground-truth-conditioned output.  This is the non-circular version of Planner C --
# the target is external to the planner, so PKL and the cost no longer share a functional form.
PLANNER_C_TRUTH_TASK = {"plannerC_ade_truth": ("JC_ade_cheap", "JC_ade_full"),
                        # final displacement error, so the robustness of the planner target does
                        # not rest on ADE alone (an average can hide a divergent endpoint)
                        "plannerC_fde_truth": ("JC_fde_cheap", "JC_fde_full")}
NOCHK = lambda c: None


TIE_SEEDS = 8      # tie-breaks averaged over this many seeds


def eta_parts(df, score, col, quota=0.20):
    """(captured reduction, oracle's achievable reduction, all-cheap cost) at `quota`.

    Returned separately because eta is their ratio and the denominator can be tiny: with
    only tens of frames carrying non-zero dJ, a bootstrap draw can land on a near-zero
    oracle prize and send eta to +-10.  The absolute reduction is always interpretable.
    """
    def pick(s, seed):
        return budget.select_pooled(np.asarray(s, float), quota, seed=seed)

    allc = float(df[col[0]].sum())
    dj = (df[col[0]] - df[col[1]]).to_numpy()
    sc = np.asarray(score, float)
    # ties at the cut are broken at random and averaged, so a count-valued signal cannot be
    # scored by row order; the oracle is averaged the same way for symmetry
    orc = float(np.mean([budget.total_risk(df, pick(dj, sd), col) for sd in range(TIE_SEEDS)]))
    tot = float(np.mean([budget.total_risk(df, pick(sc, sd), col) for sd in range(TIE_SEEDS)]))
    return allc - tot, allc - orc, allc


def eta(df, score, col, quota=0.20):
    """Share of the decision oracle's achievable cost reduction captured at `quota`."""
    got, prize, _ = eta_parts(df, score, col, quota)
    return got / prize if prize > 1e-12 else np.nan


def eta_boot(df, score, col, quota, nboot=300, seed=0):
    """Scene-level bootstrap CI for eta.  Frames within a scene are strongly correlated,
    so scenes, not frames, are the resampling unit."""
    rng = np.random.default_rng(seed)
    scenes = df.seq.to_numpy()
    uniq = np.unique(scenes)
    idx_of = {u: np.flatnonzero(scenes == u) for u in uniq}
    sc = None if score is None else np.asarray(score, float)
    _, prize0, _ = eta_parts(df, np.zeros(len(df)), col, quota)
    vals, dropped = [], 0
    for b in range(nboot):
        take = np.concatenate([idx_of[u] for u in rng.choice(uniq, len(uniq), replace=True)])
        db = df.iloc[take]
        s_ = (np.random.default_rng(1000 + b).random(len(db)) if sc is None else sc[take])
        got, prize, _ = eta_parts(db, s_, col, quota)
        # A draw whose oracle prize collapses cannot say anything about the *share* of that
        # prize a signal captures; keeping it would report an interval dominated by division
        # by almost nothing.  Draws dropped are counted and reported.
        if prize <= max(1e-12, 0.25 * prize0):
            dropped += 1
            continue
        vals.append(got / prize)
    if len(vals) < nboot // 4:
        return np.nan, np.nan, dropped / max(nboot, 1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)),
            dropped / max(nboot, 1))


def scene_z(df, score):
    """Standardise a signal within each scene.

    PKL and TIP levels differ by 4x between scene groups (14 vs 52 in two slices here), and
    so do their cheap-minus-full differences, so a pooled top-quota ranking on the raw gain
    is partly a ranking of scenes rather than of frames.  Giving every signal this variant
    keeps the comparison from understating prior work.
    """
    s = pd.Series(np.asarray(score, float), index=df.index)
    g = s.groupby(df.seq)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()


def topk_overlap(a, b, frac):
    k = int(round(frac * len(a)))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / max(k, 1)


def inversion_rate(a, b, rng, npairs=200_000):
    """Fraction of frame pairs the two signals order differently (ties excluded)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    i = rng.integers(0, len(a), npairs)
    j = rng.integers(0, len(a), npairs)
    k = i != j
    i, j = i[k], j[k]
    sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    ok = (sa != 0) & (sb != 0)
    return float((sa[ok] != sb[ok]).mean()) if ok.any() else np.nan


# Direction of each published score, derived from the authors' own code.
#
# PKL (planning_kl.py) is a BCE divergence of the predicted trajectory heatmap from the
# ground-truth-conditioned one, so it is >= 0 and **higher is worse**.
#
# TIP (ti_planner.py, get_tip) is
#       TIP_t = [q(A*) - q(Ahat)] - [p(A*) - p(Ahat)]
# with p the ground-truth-conditioned distribution, q the predicted one, A* = argmax p and
# Ahat = argmin (p - q).  Because A* maximises p, p(A*) - p(Ahat) >= 0; because Ahat minimises
# p - q, q(A*) - q(Ahat) <= p(A*) - p(Ahat).  So **TIP <= 0 and more negative is worse**, the
# opposite of PKL, reaching 0 only when the predicted distribution induces the same preference
# gap as ground truth.  The comment "to conply with PKL's definition" in ti_planner.py sits on
# a `torch.from_numpy` call and concerns the tensor type, not the sign.
#
# The allocation signal must be positive when the expensive mode helps that frame, so:
WORSE_IS_HIGHER = {"pkl": True, "tip": False}


def gain(metric: str, cheap, full):
    """Per-frame value of running the expensive mode, positive when it helps."""
    return (cheap - full) if WORSE_IS_HIGHER[metric] else (full - cheap)


def load_metric(subs: Path, metric: str, variant: str) -> pd.DataFrame | None:
    """One CSV per chunk, or a single unchunked CSV."""
    parts = sorted(subs.glob(f"{metric}_{variant}_n*c*.csv"))
    single = subs / f"{metric}_{variant}.csv"
    if not parts and single.exists():
        parts = [single]
    if not parts:
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates("sample_token", keep="first")
    # recomputed from the stored per-mode scores rather than trusted from the CSV, so a
    # sign convention can be corrected without re-running the metric
    df[f"G_{metric.upper()}"] = gain(metric, df[f"{metric}_cheap"], df[f"{metric}_full"])
    return df


def token_map(seqs, dataroot, version) -> pd.DataFrame:
    """(scene name, frame) -> sample_token, in the same temporal order NuScenesDB uses.

    Reads scene.json and sample.json directly rather than instantiating NuScenesDB: the
    full table set costs ~3 GB, which the concurrent PKL/TIP runs cannot spare.
    """
    cache = Path(CACHE) / "nusc_token_map.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        if set(seqs) <= set(df.seq.unique()):
            return df[df.seq.isin(set(seqs))].reset_index(drop=True)
    t = Path(dataroot) / version
    scenes = json.loads((t / "scene.json").read_text())
    samples = {r["token"]: r for r in json.loads((t / "sample.json").read_text())}
    want = set(seqs)
    rows = []
    for sc in scenes:
        if sc["name"] not in want:
            continue
        tok, fr = sc["first_sample_token"], 0
        while tok:
            rows.append((sc["name"], fr, tok))
            tok, fr = samples[tok]["next"], fr + 1
    df = pd.DataFrame(rows, columns=["seq", "frame", "sample_token"])
    df.to_csv(cache, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "results/raw/20260912_071140_core_matrix"),
                    help="core_matrix run holding the pickled nuScenes decision tables")
    ap.add_argument("--subs", default=str(CACHE / "nusc_submissions"))
    ap.add_argument("--variant", default="oracle", help="submission geometry variant")
    ap.add_argument("--dataroot", default="/home/kongwoang/datasets/nuscenes/trainval")
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--tag", default="phase0f_eta")
    ap.add_argument("--skip_multimetric", action="store_true")
    ap.add_argument("--nboot", type=int, default=300)
    # Derived from --variant, never defaulted to a fixed file.  A fixed default meant the mono
    # run silently read the *oracle* truth-referenced costs, pairing mono signals with an oracle
    # target; the giveaway was that plannerC_ade_truth came out identical in both runs, which I
    # saw and explained away instead of checking the cost columns.
    ap.add_argument("--planner_c_truth", default=None,
                    help="default: planC_vs_truth[_<variant>].csv for the chosen variant")
    ap.add_argument("--no_planner_c", dest="planner_c", action="store_false",
                    help="skip the Planner C task even if its CSVs exist")
    ap.add_argument("--coverage", default="per_metric", choices=["per_metric", "intersect"],
                    help="evaluate on each metric's own frames, or only where all overlap")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    subs = Path(args.subs)
    rng = np.random.default_rng(0)

    # Phase 0E's nuScenes table.  The geometry variant of the decision table and of the
    # submission are kept the same so PKL/TIP see the boxes the planner saw.
    tbl = Path(args.tables) / \
        f"nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{args.variant}.pkl"
    d = pd.read_pickle(tbl)
    print(f"decision table {tbl.name}: {len(d)} frames, {d.seq.nunique()} scenes")

    d = d.merge(token_map(sorted(d.seq.unique()), args.dataroot, args.version),
                on=["seq", "frame"],
                how="left", validate="one_to_one")
    assert d.sample_token.notna().all(), "unmapped frames"

    # Planner C's per-frame costs, if computed
    pc = sorted(subs.glob(f"planC_{args.variant}_n*c*.csv"))
    if pc and args.planner_c:
        g = pd.concat([pd.read_csv(x) for x in pc], ignore_index=True)
        g = g.drop_duplicates("sample_token", keep="first")
        keep = ["sample_token", "JC_cheap", "JC_full", "dJC", "dJC_soft"]
        d = d.merge(g[[c for c in keep if c in g.columns]], on="sample_token", how="left")
        cov = float(d.JC_cheap.notna().mean())
        print(f"  Planner C: {len(g)} samples, coverage {cov:.3f}")
        if cov > 0:
            TASKS.update(PLANNER_C_TASK)
    elif args.planner_c:
        print("  Planner C: no CSV yet -- skipped")

    if args.planner_c_truth is None:
        sfx = "" if args.variant == "oracle" else f"_{args.variant}"
        args.planner_c_truth = str(CACHE / "planner_d" / f"planC_vs_truth{sfx}.csv")
    truth = Path(args.planner_c_truth)
    print(f"  Planner C truth-referenced costs: {truth.name}")
    if args.planner_c and truth.exists():
        g = pd.read_csv(truth).drop_duplicates("sample_token", keep="first")
        keep = ["sample_token", "JC_ade_cheap", "JC_ade_full", "JC_fde_cheap", "JC_fde_full"]
        d = d.merge(g[keep], on="sample_token", how="left")
        cov = float(d.JC_ade_cheap.notna().mean())
        print(f"  Planner C vs truth: {len(g)} samples, coverage {cov:.3f}")
        if cov > 0:
            TASKS.update(PLANNER_C_TRUTH_TASK)

    have = {}
    for m in ("pkl", "tip"):
        g = load_metric(subs, m, args.variant)
        if g is None:
            print(f"  {m.upper()}: no CSV yet -- skipped")
            continue
        col = f"G_{m.upper()}"
        d = d.merge(g[["sample_token", col]], on="sample_token", how="left")
        cov = float(d[col].notna().mean())
        print(f"  {m.upper()}: {len(g)} samples, coverage {cov:.3f}")
        have[m] = col

    # Two coverage rules, both reported because they answer different questions.
    #   intersect  -- every signal on the frames where *all* published metrics have a score,
    #                 which is the only like-for-like comparison between PKL and TIP.
    #   per_metric -- every signal on all frames, each published metric on the frames it
    #                 covers, which uses all the data actually computed.
    if have:
        inter = np.ones(len(d), bool)
        for col in have.values():
            inter &= d[col].notna().to_numpy()
        print(f"  coverage: intersection {int(inter.sum())}/{len(d)} frames; "
              + ", ".join(f"{m.upper()} {int(d[c].notna().sum())}" for m, c in have.items()))
        if args.coverage == "intersect":
            d = d[inter].reset_index(drop=True)

    for tname, col in TASKS.items():
        d[f"_dJ_{tname}"] = (d[col[0]] - d[col[1]]).to_numpy()
    dmask = {t: (d[c[0]].notna() & d[c[1]].notna()).to_numpy() for t, c in TASKS.items()}

    # Wiring validation.  Before believing a low eta, check the metrics are connected the
    # way their papers define them: the per-frame *level* must rise with that frame's error
    # count.  If it does and the cheap-minus-full *difference* still fails to rank frames by
    # dJ, the failure is in the signal, not in the plumbing.
    wire = []
    for m, gcol in have.items():
        for a, b, why in [(f"{m}_cheap", "cheap_fn", "more missed objects -> worse score"),
                          (f"{m}_full", "full_fn", "more missed objects -> worse score"),
                          (f"{m}_cheap", "n_gt", "more objects present -> more to get wrong"),
                          (f"{m}_cheap", "cheap_fp", "more false positives -> worse score"),
                          (gcol, "dE", "does the gain track the exact perception gain"),
                          (gcol, "crit_sum", "does the gain track downstream criticality")]:
            if a in d.columns and b in d.columns:
                wire.append({"metric": m, "x": a, "y": b, "expect": why,
                             "spearman": float(stats.spearmanr(d[a], d[b]).correlation)})
    if wire:
        w = pd.DataFrame(wire)
        w.to_csv(run / "wiring_validation.csv", index=False)
        print("\n=== wiring validation (levels must track error counts) ===")
        print(w[["x", "y", "spearman", "expect"]].to_string(
            index=False, float_format=lambda v: f"{v:+.3f}"))

    # How much signal there is to recover at all: dJ is zero wherever the cheap and the
    # expensive mode lead to the same action and the same cost.
    print("\n=== how much is there to allocate (eta's denominator, in absolute terms) ===")
    stakes = []
    for tname, col in TASKS.items():
        dsel = d[dmask[tname]]
        dj = (dsel[col[0]] - dsel[col[1]]).to_numpy()
        act = ("same_action" if tname == "longitudinal" else
               "lat_same_action" if tname == "lateral" else None)
        allc, allf = float(dsel[col[0]].sum()), float(dsel[col[1]].sum())
        orc = budget.total_risk(dsel, budget.select_pooled(dj, 0.20), col)
        st = {"task": tname, "J_all_cheap": allc, "J_all_full": allf,
              "J_oracle_at_20pct": orc,
              "oracle_reduction_frac": (allc - orc) / max(allc, 1e-9),
              "all_full_reduction_frac": (allc - allf) / max(allc, 1e-9),
              "frames_dJ_nonzero": int((np.abs(dj) > 1e-9).sum()),
              "frames_dJ_pos": int((dj > 1e-9).sum()), "frames_dJ_neg": int((dj < -1e-9).sum()),
              "action_changes": (int((dsel[act] == 0).sum()) if act else -1),
              "n_frames": len(dsel)}
        stakes.append(st)
        print(f"  {tname:13s} oracle@20% cuts {100 * st['oracle_reduction_frac']:5.2f}% of the"
              f" all-cheap cost; all-full at 100% compute cuts"
              f" {100 * st['all_full_reduction_frac']:5.2f}%")
        print(f"                |dJ|>0 on {st['frames_dJ_nonzero']}/{len(dsel)} frames"
              f"  (+{st['frames_dJ_pos']} / -{st['frames_dJ_neg']}),"
              f" action changes {st['action_changes']}")
    pd.DataFrame(stakes).to_csv(run / "allocation_stakes.csv", index=False)

    rows, curves = [], []
    for tname, col in TASKS.items():
        dj = d[f"_dJ_{tname}"].to_numpy()
        best_metric = max(PM.METRICS, key=lambda m: eta(d, d[f"dE_{m}"], col))

        signals = {
            "random": None,                                   # averaged over seeds below
            "uncertainty": d.unc_sum.to_numpy(),
            "criticality": d.crit_sum.to_numpy(),
            "perception_oracle_dE": d["dE"].to_numpy(),
            f"best_dE_metric ({best_metric})": d[f"dE_{best_metric}"].to_numpy(),
        }
        if not args.skip_multimetric:
            # Fit only where this task has a target.  The Planner C tasks cover 2,655 of the
            # 3,376 frames (the rest have no 4 s future), so fitting on the whole table fed the
            # regressor 721 NaN targets and it raised "Input y contains NaN" after ~18 minutes
            # of work.  Predictions are written back into a full-length array with NaN outside
            # the covered set, which the per-signal coverage mask already handles.
            feats = _pm.oracle_features(d)
            cov = dmask[tname]
            dd2 = d[cov].copy().reset_index(drop=True)
            for c in feats:
                dd2[c] = pd.to_numeric(dd2[c], errors="coerce").fillna(0.0)
            dd2["_dJ"] = dj[cov]
            pred = np.full(len(d), np.nan)
            pred[cov] = predict.loso(dd2, feats, "gbm", "_dJ", "reg", checker=NOCHK).pred
            signals["multimetric_dE_oracle"] = pred
        for m, gcol in have.items():
            signals[f"{m.upper()}_gain"] = d[gcol].to_numpy()
        signals["decision_oracle_dJ"] = dj

        base = dict(signals)
        for name, sc in base.items():
            if sc is None or name == "decision_oracle_dJ":
                continue
            signals[f"{name} [scene-z]"] = scene_z(d, sc)

        for name, sc in signals.items():
            # under per_metric coverage a published metric is scored on its own frames only,
            # and so is the oracle it is compared against, so the ratio stays well defined
            sub = d[col[0]].notna().to_numpy() & d[col[1]].notna().to_numpy()
            for m, gcol in have.items():
                if name.startswith(m.upper()):
                    sub = sub & d[gcol].notna().to_numpy()
            dd = d if sub.all() else d[sub].reset_index(drop=True)
            ss = None if sc is None else np.asarray(sc, float)[sub]
            r = {"task": tname, "variant": args.variant, "signal": name,
                 "scene_normalised": "[scene-z]" in name,
                 "n_frames": len(dd), "n_scenes": int(dd.seq.nunique())}
            for q in QUOTAS:
                if ss is None:
                    vs = [eta_parts(dd, np.random.default_rng(k).random(len(dd)), col, q)
                          for k in range(16)]
                    got = float(np.mean([g for g, _, _ in vs])); prize = vs[0][1]; allc = vs[0][2]
                else:
                    got, prize, allc = eta_parts(dd, ss, col, q)
                v = got / prize if prize > 1e-12 else np.nan
                r[f"eta_{int(q * 100)}"] = v
                r[f"reduction_frac_{int(q * 100)}"] = got / allc if allc > 1e-12 else np.nan
                r[f"tie_frac_{int(q * 100)}"] = (np.nan if ss is None
                                                else budget.tie_fraction(ss, q))
                lo, hi, drop = eta_boot(dd, ss, col, q, nboot=args.nboot)
                r[f"eta_{int(q * 100)}_lo"], r[f"eta_{int(q * 100)}_hi"] = lo, hi
                r[f"eta_{int(q * 100)}_boot_dropped"] = drop
                curves.append({"task": tname, "variant": args.variant, "signal": name,
                               "quota": q, "eta": v, "eta_lo": lo, "eta_hi": hi,
                               "reduction_frac": r[f"reduction_frac_{int(q * 100)}"]})
            if ss is not None:
                dsub = dj[sub]
                r["spearman_vs_dJ"] = float(stats.spearmanr(ss, dsub).correlation)
                r["kendall_vs_dJ"] = float(stats.kendalltau(ss, dsub).correlation)
                r["inversion_rate_vs_dJ"] = inversion_rate(ss, dsub, rng)
                r["top10_overlap_vs_dJ"] = topk_overlap(ss, dsub, 0.10)
                r["top20_overlap_vs_dJ"] = topk_overlap(ss, dsub, 0.20)
            rows.append(r)

    out = Path(RESULTS) / "final"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(out / "phase0f_planning_metric_eta.csv", index=False)
    m.to_csv(run / "phase0f_planning_metric_eta.csv", index=False)
    pd.DataFrame(curves).to_csv(run / "phase0f_regret_curves.csv", index=False)
    d.to_pickle(run / "joined_frames.pkl")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    for tname in TASKS:
        s = m[m.task == tname]
        if s.empty:
            continue
        print(f"\n=== {tname} ({args.variant}) ===")
        s = s.copy()
        s["eta_20_ci"] = [f"[{a:+.2f},{b:+.2f}]" if np.isfinite(a) else "--"
                          for a, b in zip(s.eta_20_lo, s.eta_20_hi)]
        print(s[["signal", "n_frames", "eta_10", "eta_20", "eta_20_ci",
                 "reduction_frac_20", "eta_30", "eta_50", "spearman_vs_dJ",
                 "inversion_rate_vs_dJ", "top20_overlap_vs_dJ"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # kill test 1: a published planning-aware metric recovering >=0.8 at 20% quota
    kill = m[m.signal.str.contains("PKL|TIP")][["task", "signal", "eta_20"]]
    if len(kill):
        kill = m[m.signal.str.contains("PKL|TIP")][
            ["task", "signal", "eta_10", "eta_20", "eta_20_lo", "eta_20_hi",
             "eta_30", "eta_50"]]
        worst = kill.eta_20.max()
        print(f"\nKILL TEST 1  max eta@20 over published planning-aware metrics: {worst:+.3f}"
              f"  -> {'STOP (metrics already solve it)' if worst >= 0.8 else 'gap survives'}")
        (run / "kill_test_1.json").write_text(json.dumps(
            {"max_eta20_published": float(worst), "threshold": 0.8,
             "verdict": "STOP" if worst >= 0.8 else "PROCEED",
             "rows": kill.to_dict("records")}, indent=2))
    print("\nwrote", run)


if __name__ == "__main__":
    main()

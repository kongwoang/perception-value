#!/usr/bin/env python
"""Phase 0G: do two downstream systems disagree about WHICH inputs deserve extra compute?

This is the score-free form of the planner-conditionality question.  It involves no allocation
metric, no PKL, no TIP and no perception feature: only the two decision values themselves.

For downstream systems q1 and q2 evaluated on the same frames:

    rho          Spearman between V^q1 and V^q2, over all frames and over the frames where
                 *both* systems respond (both values non-zero), since a frame neither system
                 reacts to contributes agreement for free
    overlap@k    share of q1's top-k oracle set that is also in q2's, against the chance level
                 k/N that two unrelated rankings would give
    cross-eta    eta_{q1->q2}(k) = sum of V^q2 over q1's top-k, divided by the same sum over
                 q2's own top-k: 1.0 if the systems agree about what is valuable, and about the
                 random level if they are unrelated

Every quantity is bootstrapped over **scenes**, and for every quantity its baseline is
recomputed inside the same draw so a **paired difference** can be reported.  Comparing two
separately-computed intervals for overlap tests a much weaker claim than asking whether the
paired difference excludes zero, which is what "the systems disagree" actually means.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rap import runmeta                                                        # noqa: E402
from rap.paths import CACHE, RESULTS                                           # noqa: E402

QUOTAS = [0.10, 0.20, 0.30]


def topk_mask(v: np.ndarray, q: float) -> np.ndarray:
    k = max(int(round(q * len(v))), 1)
    sel = np.zeros(len(v), bool)
    sel[np.argsort(-np.asarray(v, float), kind="stable")[:k]] = True
    return sel


def cross_eta(v_from: np.ndarray, v_to: np.ndarray, q: float) -> float:
    best = float(np.asarray(v_to, float)[topk_mask(v_to, q)].sum())
    got = float(np.asarray(v_to, float)[topk_mask(v_from, q)].sum())
    return got / best if abs(best) > 1e-12 else np.nan


def random_eta(v_to: np.ndarray, q: float, rng, reps: int = 8) -> float:
    best = float(np.asarray(v_to, float)[topk_mask(v_to, q)].sum())
    if abs(best) <= 1e-12:
        return np.nan
    vals = [float(np.asarray(v_to, float)[topk_mask(rng.random(len(v_to)), q)].sum()) / best
            for _ in range(reps)]
    return float(np.mean(vals))


def scene_boot(scenes: np.ndarray, nboot: int, rng):
    uniq = np.unique(scenes)
    idx = {u: np.flatnonzero(scenes == u) for u in uniq}
    for _ in range(nboot):
        yield np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq), replace=True)])


def ci(vals):
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 20:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brake_table", required=True,
                    help="pickled decision table with J_cheap/J_full for the braking controller")
    ap.add_argument("--plan_csv", required=True,
                    help="planC_vs_truth CSV with JC_ade_cheap/JC_ade_full")
    ap.add_argument("--token_map", default=str(CACHE / "nusc_token_map.csv"))
    ap.add_argument("--label", default="mono")
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--tag", default="cross_target")
    args = ap.parse_args()
    run = runmeta.new_run(args.tag, vars(args))
    rng = np.random.default_rng(0)

    b = pd.read_pickle(args.brake_table)
    tm = pd.read_csv(args.token_map)
    b = b.merge(tm, on=["seq", "frame"], how="left", validate="one_to_one")
    b["V_brake"] = b.J_cheap - b.J_full
    p = pd.read_csv(args.plan_csv).drop_duplicates("sample_token")
    p["V_plan"] = p.JC_ade_cheap - p.JC_ade_full

    d = b[["sample_token", "seq", "V_brake"]].merge(
        p[["sample_token", "V_plan"]], on="sample_token", how="inner")
    d = d.rename(columns={"seq": "scene"}).dropna()
    va, vb = d.V_brake.to_numpy(float), d.V_plan.to_numpy(float)
    scenes = d.scene.to_numpy()
    both = (np.abs(va) > 1e-9) & (np.abs(vb) > 1e-9)
    print(f"  {args.label}: {len(d)} common frames, {len(np.unique(scenes))} scenes; "
          f"brake responds on {int((np.abs(va)>1e-9).sum())}, plan on "
          f"{int((np.abs(vb)>1e-9).sum())}, both on {int(both.sum())}")

    rows = []
    # ---- rank agreement -----------------------------------------------------------------
    for name, mask in (("all frames", np.ones(len(d), bool)), ("both respond", both)):
        r = float(stats.spearmanr(va[mask], vb[mask]).correlation)
        k = float(stats.kendalltau(va[mask], vb[mask]).correlation)
        boots = []
        for take in scene_boot(scenes, args.nboot, rng):
            m = mask[take]
            if m.sum() > 30:
                boots.append(float(stats.spearmanr(va[take][m], vb[take][m]).correlation))
        lo, hi = ci(boots)
        rows.append({"quantity": f"spearman ({name})", "value": r, "baseline": 0.0,
                     "paired_diff": r, "lo": lo, "hi": hi, "n": int(mask.sum())})
        print(f"  spearman, {name:13s} {r:+.3f} [{lo:+.3f},{hi:+.3f}]   kendall {k:+.3f}")

    # ---- oracle-set overlap against chance ---------------------------------------------
    for q in QUOTAS:
        chance = max(int(round(q * len(d))), 1) / len(d)
        ov = float((topk_mask(va, q) & topk_mask(vb, q)).sum() / topk_mask(va, q).sum())
        diffs, vals = [], []
        for take in scene_boot(scenes, args.nboot, rng):
            x, y = va[take], vb[take]
            sa, sb = topk_mask(x, q), topk_mask(y, q)
            o = float((sa & sb).sum() / sa.sum())
            c = sa.sum() / len(x)
            vals.append(o); diffs.append(o - c)
        lo, hi = ci(diffs)
        rows.append({"quantity": f"overlap@{int(q*100)}", "value": ov, "baseline": chance,
                     "paired_diff": ov - chance, "lo": lo, "hi": hi, "n": len(d)})
        print(f"  overlap@{int(q*100):<3d} {ov:.3f}  chance {chance:.3f}  "
              f"paired diff {ov-chance:+.3f} [{lo:+.3f},{hi:+.3f}]")

    # ---- cross-target eta against a random ranking -------------------------------------
    for lab, x, y in (("brake->plan", va, vb), ("plan->brake", vb, va)):
        for q in QUOTAS:
            e = cross_eta(x, y, q)
            r0 = random_eta(y, q, rng)
            diffs = []
            for take in scene_boot(scenes, args.nboot, rng):
                ee = cross_eta(x[take], y[take], q)
                rr = random_eta(y[take], q, rng)
                if np.isfinite(ee) and np.isfinite(rr):
                    diffs.append(ee - rr)
            lo, hi = ci(diffs)
            rows.append({"quantity": f"cross_eta {lab} @{int(q*100)}", "value": e,
                         "baseline": r0, "paired_diff": e - r0, "lo": lo, "hi": hi,
                         "n": len(d)})
            print(f"  cross-eta {lab} @{int(q*100):<3d} {e:+.3f}  random {r0:+.3f}  "
                  f"paired diff {e-r0:+.3f} [{lo:+.3f},{hi:+.3f}]")

    m = pd.DataFrame(rows)
    m.insert(0, "geometry", args.label)
    out = Path(RESULTS) / "final" / f"phase0g_cross_target_{args.label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False); m.to_csv(run / out.name, index=False)

    sig = m[(m.quantity.str.startswith(("overlap", "cross_eta")))
            & ~((m.lo < 0) & (m.hi > 0))]
    verdict = {"geometry": args.label, "n_frames": len(d),
               "quantities_whose_paired_difference_excludes_zero": sig.quantity.tolist()}
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n  paired differences excluding zero: "
          f"{sig.quantity.tolist() if len(sig) else 'none'}")
    print("  wrote", out)


if __name__ == "__main__":
    main()

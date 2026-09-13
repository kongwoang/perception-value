#!/usr/bin/env python
"""Phase 0G Track B: is disagreement driven by the objective, the planner, or neither?

Brake-vs-plan on nuScenes differs in both objective and planner family, and PDM-vs-IDM on nuPlan
shares both objective and an IDM core, so neither comparison alone separates the two factors.
Costs were recorded per component for each planner, which gives the full 2x2 on the same states:
same planner under two objectives, two planners under one objective, and both varied.

Outcome, recorded when this was first run: every cell agrees positively (gamma +0.34 to +0.78),
including the same planner under two objectives. So on nuPlan neither factor produces the
disagreement seen on nuScenes -- the values co-vary, most plausibly through a shared "did the
intervention remove a relevant object" factor that any cost on either planner responds to.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import RESULTS                                                   # noqa: E402
from rap.transfer import strict_pairs                                           # noqa: E402

out = Path(RESULTS) / "final"
I = pd.read_csv(out / "phase0g_external_idm_raw.csv")
P = pd.read_csv(out / "phase0g_external_pdm_closed_raw.csv")
j = I.merge(P, on=["scenario", "iteration"], suffixes=("_i", "_p"))

def V(sfx, obj):
    def c(m):
        coll = j[f"collision_{m}{sfx}"]; sh = np.clip(1 - j[f"min_clearance_{m}{sfx}"], 0, 2) ** 2
        return {"safety": 10 * coll + 1.5 * sh,
                "progress_dev": j[f"log_deviation_mean_{m}{sfx}"]}[obj]
    return (c("cheap") - c("full")).to_numpy()

vals = {(p, o): V(s, o) for p, s in (("IDM", "_i"), ("PDM", "_p"))
        for o in ("safety", "progress_dev")}
g = j.scenario.to_numpy(); u = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in u}
cells = [("same planner, different objective", ("IDM", "safety"), ("IDM", "progress_dev")),
         ("same planner, different objective", ("PDM", "safety"), ("PDM", "progress_dev")),
         ("different planner, same objective", ("IDM", "safety"), ("PDM", "safety")),
         ("different planner, same objective", ("IDM", "progress_dev"), ("PDM", "progress_dev")),
         ("both differ", ("IDM", "safety"), ("PDM", "progress_dev")),
         ("both differ", ("PDM", "safety"), ("IDM", "progress_dev"))]
rows = []
for label, k1, k2 in cells:
    a, b = vals[k1], vals[k2]
    sp = strict_pairs(a, b)
    both = (np.abs(a) > 1e-9) & (np.abs(b) > 1e-9)
    rng = np.random.default_rng(0)
    boots = [strict_pairs(a[t], b[t])["gamma"] for t in
             (np.concatenate([idx[x] for x in rng.choice(u, len(u))]) for _ in range(300))]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    rows.append({"contrast": label, "a": "-".join(k1), "b": "-".join(k2),
                 "both_responsive": int(both.sum()),
                 "same_sign_both_responsive": float((np.sign(a[both]) == np.sign(b[both])).mean()),
                 "n_pairs_strict": sp["n_pairs_strict"], "disagreement": sp["disagreement"],
                 "gamma": sp["gamma"], "gamma_lo": lo, "gamma_hi": hi})
d = pd.DataFrame(rows); d.to_csv(out / "phase0g_external_2x2.csv", index=False)
print(d.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

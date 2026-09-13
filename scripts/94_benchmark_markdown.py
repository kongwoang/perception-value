#!/usr/bin/env python
"""Render the benchmark CSVs as markdown tables, so no number in the docs is copied by hand.

Reads results/final/benchmark_{table,cells,self_agreement,budget_two_level,budget_multifidelity}.csv
and writes docs/benchmark_tables.md.  Bold marks a deployable signal whose paired difference to
random has a lower bound above zero; a dash is a signal that cannot exist in that cell (the reason is
in the `na_reason` column of benchmark_table.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "results" / "final"
KEY = ["track", "geometry", "system", "target"]
COLS = [("random", "random"), ("uncertainty", "unc."), ("criticality_cheap", "crit. cheap"),
        ("gate_ridge", "gate ridge"), ("gate_gbm", "gate GBM"), ("criticality_gt", "crit. GT"),
        ("dE_exact", "ΔE exact"), ("dE_E6_risk_weighted", "ΔE E6"), ("PKL", "PKL"), ("TIP", "TIP")]
DEPLOY = {"random", "uncertainty", "criticality_cheap", "gate_ridge", "gate_gbm"}


def f3(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def label(r):
    g = "" if r["geometry"] == "n/a" else f" {r['geometry']}"
    return f"{r['track']}{g} · {r['system']}" + (f" · {r['target']}" if r["track"] == "nuPlan" else "")


def cells_table(c, split):
    c = c[c.split == split]
    out = ["| cell | units | frames | affected | harmed of affected | D | all-FULL | oracle@20 |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in c.iterrows():
        out.append(f"| {label(r)} | {r.n_units} | {r.n_frames} | {r.affected_share:.1%} | "
                   f"{r.harmed_among_affected:.1%} | {abs(r.destroyed_benefit_D):.2f} | "
                   f"{-r.all_full_reduction:+.2%} | {-r.oracle20_reduction:+.2%} |")
    return "\n".join(out)


def eta_table(t, q, split="test"):
    t = t[(t.split == split)]
    head = "| cell | " + " | ".join(n for _, n in COLS) + " |"
    out = [head, "|" + "---|" * (len(COLS) + 1)]
    for key, g in t.groupby(KEY, sort=False):
        r0 = g.iloc[0]
        cells = []
        for s, _ in COLS:
            x = g[(g.signal == s)]
            if x.empty or not bool(x.iloc[0].available):
                cells.append("—")
                continue
            x = x[np.isclose(x.quota, q)].iloc[0]
            txt = f3(x.eta)
            if s in DEPLOY and s != "random" and np.isfinite(x.minus_random_lo) and x.minus_random_lo > 0:
                txt = f"**{txt}**"
            cells.append(txt)
        out.append(f"| {label(r0)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def paired_table(t, q):
    t = t[(t.split == "test") & t.available.astype(bool) & np.isclose(t.quota.fillna(-1), q)]
    out = ["| cell | " + " | ".join(n for s, n in COLS if s in DEPLOY and s != "random") + " |",
           "|" + "---|" * 5]
    for key, g in t.groupby(KEY, sort=False):
        cells = []
        for s, _ in COLS:
            if s not in DEPLOY or s == "random":
                continue
            x = g[g.signal == s]
            cells.append("—" if x.empty else
                         f"{f3(x.iloc[0].minus_random)} [{f3(x.iloc[0].minus_random_lo)}, {f3(x.iloc[0].minus_random_hi)}]")
        out.append(f"| {label(g.iloc[0])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def self_table(s):
    out = ["| cell | split | metric | η self @20 | η truth @20 | S @20 [95% CI] | S @10 | S @30 | S @50 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (geom, system, split, metric), g in s.groupby(["geometry", "system", "split", "metric"], sort=False):
        byq = {round(q, 2): r for q, r in zip(g.quota, g.itertuples())}
        r = byq[0.2]
        out.append(f"| nuScenes {geom} · {system} | {split} | {metric} | {r.eta_self:.3f} | {r.eta_truth:.3f} | "
                   f"{r.S:.2f} [{f3(r.S_lo)}, {f3(r.S_hi)}] | {byq[0.1].S:.2f} | {byq[0.3].S:.2f} | {byq[0.5].S:.2f} |")
    return "\n".join(out)


def budget_tables(two, multi):
    out = []
    if two is not None:
        b = two[np.isclose(two.budget_level, 0.2) & two.feasible.astype(bool)]
        out += ["| cell | unit | budget / frame | signal | overhead | escalated | η | η 95% CI |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in b[b.signal.isin(["random", "uncertainty", "gate_ridge", "gate_gbm"])].iterrows():
            out.append(f"| {label(r)} | {r.unit} | {r.budget_per_frame:.2f} | {r.signal} | {r.overhead:.3f} | "
                       f"{r.escalated_frac:.1%} | {f3(r.eta)} | [{f3(r.eta_lo)}, {f3(r.eta_hi)}] |")
    if multi is not None:
        m = multi[np.isclose(multi.budget_level, 0.2)]
        out += ["", "| system | unit | signal | η | share 384 | share 512 | share 640 | other-unit spend / budget |",
                "|---|---|---|---|---|---|---|---|"]
        for _, r in m.iterrows():
            other = "mJ" if r.unit == "ms" else "ms"
            sp, bu = r.get(f"extra_{other}_per_frame", np.nan), r.get(f"extra_{other}_budget_at_same_level", np.nan)
            ratio = f"{sp / bu:.2f}" if np.isfinite(sp) and np.isfinite(bu) and bu > 0 else "—"
            pct = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.1%}"          # noqa: E731
            out.append(f"| {r.system} | {r.unit} | {r.signal} | {f3(r.eta)} | {pct(r.get('share_384', np.nan))} | "
                       f"{pct(r.get('share_512', np.nan))} | {pct(r.share_640)} | {ratio} |")
    return "\n".join(out)


def read(name):
    """pandas parses the nuPlan geometry label "n/a" as a missing value; put it back."""
    f = FINAL / f"{name}.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    if "geometry" in d:
        d["geometry"] = d.geometry.fillna("n/a")
    return d


def main():
    t, c, s = read("benchmark_table"), read("benchmark_cells"), read("benchmark_self_agreement")
    two, multi = read("benchmark_budget_two_level"), read("benchmark_budget_multifidelity")
    doc = ["# Benchmark tables (generated by scripts/94_benchmark_markdown.py — do not edit by hand)", "",
           "Sources: `results/final/benchmark_*.csv`. Bold = deployable signal whose paired difference to "
           "random excludes zero. — = signal cannot exist in the cell (reason in `na_reason`).", "",
           "## Cells, test split", "", cells_table(c, "test"), "",
           "## Cells, all units", "", cells_table(c, "all"), ""]
    for q in (0.1, 0.2, 0.3, 0.5):
        doc += [f"## η@{int(q * 100)}, test split", "", eta_table(t, q), ""]
    doc += ["## Deployable signals minus random, η@20, test split (paired, 95% CI)", "", paired_table(t, 0.2), "",
            "## η@20, all units (non-learned signals; continuity with earlier reports)", "", eta_table(t, 0.2, "all"), "",
            "## Self-agreement share S = 1 − η_truth / η_self", "", self_table(s), ""]
    if two is not None or multi is not None:
        doc += ["## Allocation under measured cost, budget level 20%", "", budget_tables(two, multi), ""]
    out = ROOT / "docs" / "benchmark_tables.md"
    out.write_text("\n".join(doc))
    print("wrote", out)


if __name__ == "__main__":
    main()

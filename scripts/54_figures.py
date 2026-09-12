#!/usr/bin/env python
"""Phase 0E Stage 18: the data figures. No paper assembly."""
from __future__ import annotations

import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from rap import budget                              # noqa: E402
from rap.paths import RAW, RESULTS                  # noqa: E402
from rap.viz import (GRID, INK, INK2, MUTED, SERIES, SURFACE, _tidy)  # noqa: E402

OUT = Path(RESULTS) / "final" / "figures"
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def eta(df, score, col, quota=0.20):
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def short(row):
    d = "K" if row.dataset == "KITTI" else "nuSc"
    det = "Y8" if row.detector == "YOLOv8s" else "RT"
    gap = row.cheap_mode.replace("cheap_", "").replace("ns_", "").replace("rt_", "") \
          + "→" + row.full_mode.replace("full_", "").replace("ns_", "").replace("rt_", "")
    g = "" if row.geometry == "mono" else " ·orc"
    return f"{d}/{det} {gap}{g}"


def main():
    cm = pd.read_csv(Path(RESULTS) / "final" / "core_matrix.csv")
    cmr = Path(sorted(glob.glob(str(RAW / "*_core_matrix")))[-1])
    tables = {p.stem: pd.read_pickle(p).reset_index(drop=True) for p in sorted(cmr.glob("*.pkl"))}
    cm["label"] = cm.apply(short, axis=1)

    # 01 perception vs decision scatter
    pick = ["KITTI__YOLOv8s__cheap_320tofull_640__mono",
            "KITTI__RT-DETR-l__rt_cheap_320tort_full_640__mono",
            "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__mono"]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), sharex="col")
    for j, key in enumerate(pick):
        d = tables.get(key)
        if d is None:
            continue
        for i, (tname, col) in enumerate(TASKS.items()):
            ax = axes[i, j]
            x = d["dE"].to_numpy(); y = (d[col[0]] - d[col[1]]).to_numpy()
            ax.scatter(x, y, s=5, alpha=0.09, color=SERIES[0], linewidths=0)
            q = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
            idx = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
            bx = [x[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 40]
            by = [y[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 40]
            ax.plot(bx, by, "-o", ms=6, color=INK2)
            ax.axhline(0, color=MUTED, lw=1)
            lo, hi = np.percentile(y, [1, 99]); pad = 0.15 * max(hi - lo, 1e-6)
            ax.set_ylim(lo - pad, hi + pad)
            xlo, xhi = np.percentile(x, [0.5, 99.5])
            ax.set_xlim(xlo - 0.5, xhi + 0.5)
            r = stats.spearmanr(x, y).correlation
            ax.set_title(f"{key.split('__')[0]}/{key.split('__')[1]} · {tname}  ρ={r:+.3f}",
                         fontsize=10)
            if j == 0:
                ax.set_ylabel(f"ΔJ ({tname})")
            if i == 1:
                ax.set_xlabel("perception gain ΔE")
            _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout(); save(fig, "01_perception_vs_decision_scatter")

    # 02 oracle gap across configurations
    for task in TASKS:
        sub = cm[cm.task == task].sort_values("eta_perception_oracle_20")
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(sub) + 1.6))
        y = np.arange(len(sub))[::-1]
        ax.barh(y, sub.eta_perception_oracle_20, 0.55, color=SERIES[0], zorder=3,
                label="perception-gain oracle")
        ax.scatter(sub.eta_multimetric_perception_oracle_20, y, s=48, color=SERIES[1],
                   zorder=5, label="multi-metric perception oracle")
        ax.axvline(1.0, color=MUTED, lw=1.6, ls="--")
        ax.annotate("decision-value oracle = 1.0", (1.0, y.max()), xytext=(-6, 4),
                    textcoords="offset points", ha="right", fontsize=9, color=INK2)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(y); ax.set_yticklabels(sub.label, fontsize=9)
        ax.set_xlabel("η at a 20% FULL-compute quota")
        ax.set_title(f"How much of the decision oracle perception knowledge buys — {task}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, axis="x"); ax.set_axisbelow(True)
        fig.tight_layout(); save(fig, f"02_oracle_gap_{task}")

    # 03 perception-metric robustness
    mr = Path(sorted(glob.glob(str(RAW / "*_percep_metrics")))[-1]) / "metric_robustness.csv"
    if mr.exists():
        m = pd.read_csv(mr)
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharey=True)
        for ax, task in zip(axes, TASKS):
            sub = m[m.task == task]
            cfgs = sorted(sub.config.unique())
            mets = list(sub.metric.unique())
            wdt = 0.8 / len(cfgs)
            for i, c in enumerate(cfgs):
                v = [float(sub[(sub.config == c) & (sub.metric == mm)]["eta@20"].iloc[0])
                     for mm in mets]
                ax.bar(np.arange(len(mets)) + i * wdt - 0.4 + wdt / 2, v, wdt,
                       color=SERIES[i], label=c, zorder=3)
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_xticks(np.arange(len(mets)))
            ax.set_xticklabels([mm.replace("_", " ") for mm in mets], rotation=28,
                               ha="right", fontsize=8)
            ax.set_title(task); _tidy(ax)
        axes[0].set_ylabel("η@20")
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle("No definition of perception improvement ranks frames for the decision",
                     fontsize=12, x=0.02, ha="left")
        fig.tight_layout(); save(fig, "03_perception_metric_robustness")

    # 04 + 05 + 06 task conditionality
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    d = tables[pick[0]]
    jl = (d.J_cheap - d.J_full).to_numpy(); jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
    ax = axes[0]
    ax.scatter(jl, jt, s=5, alpha=0.12, color=SERIES[0], linewidths=0)
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    for a, p in ((jl, "x"), (jt, "y")):
        pass
    lo, hi = np.percentile(jl, [1, 99]); ax.set_xlim(lo, hi)
    lo, hi = np.percentile(jt, [1, 99]); ax.set_ylim(lo, hi)
    ax.set_xlabel("ΔJ longitudinal"); ax.set_ylabel("ΔJ lateral")
    ax.set_title(f"A  Same frames, two objectives (ρ={stats.spearmanr(jl,jt).correlation:+.3f})")
    _tidy(ax); ax.grid(True, axis="x")

    ax = axes[1]
    labs, o10, o20, o30 = [], [], [], []
    for key, t in tables.items():
        a = (t.J_cheap - t.J_full).to_numpy(); b = (t.Jlat_cheap - t.Jlat_full).to_numpy()
        def ov(f):
            k = int(round(f * len(a)))
            return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k])) / max(k, 1)
        p = key.split("__")
        labs.append(f"{p[0][:4]}/{p[1][:2]} {p[-1][:4]}")
        o10.append(ov(0.10)); o20.append(ov(0.20)); o30.append(ov(0.30))
    x = np.arange(len(labs))
    ax.bar(x - 0.26, o10, 0.25, color=SERIES[0], label="top 10%", zorder=3)
    ax.bar(x, o20, 0.25, color=SERIES[1], label="top 20%", zorder=3)
    ax.bar(x + 0.26, o30, 0.25, color=SERIES[2], label="top 30%", zorder=3)
    ax.axhline(0.8, color=SERIES[7], lw=1.6, ls="--")
    ax.annotate("F6 warning threshold", (len(labs) - 0.5, 0.81), ha="right", fontsize=8,
                color=SERIES[7])
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("overlap of the two tasks' top frames")
    ax.set_title("B  Optimal allocations barely overlap")
    ax.legend(fontsize=8); _tidy(ax)

    ax = axes[2]
    reg = []
    for key, t in tables.items():
        a = (t.J_cheap - t.J_full).to_numpy(); b = (t.Jlat_cheap - t.Jlat_full).to_numpy()
        reg.append((eta(t, b, TASKS["longitudinal"]), eta(t, a, TASKS["lateral"])))
    reg = np.array(reg)
    x = np.arange(len(reg))
    ax.bar(x - 0.2, reg[:, 0], 0.38, color=SERIES[0], label="long. cost, lateral ranking",
           zorder=3)
    ax.bar(x + 0.2, reg[:, 1], 0.38, color=SERIES[1], label="lateral cost, long. ranking",
           zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("η when the wrong task's ranking is used")
    ax.set_title("C  Cross-applying a ranking is harmful")
    ax.legend(fontsize=8); _tidy(ax)
    fig.tight_layout(); save(fig, "04_task_conditionality")

    # 07 fidelity, 08 detector, 09 dataset — one grouped view
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    groups = [
        ("07 fidelity gap (KITTI/YOLOv8s)",
         cm[(cm.dataset == "KITTI") & (cm.detector == "YOLOv8s") & (cm.geometry == "mono")]),
        ("08 detector (KITTI, mono)",
         cm[(cm.dataset == "KITTI") & (cm.geometry == "mono")]),
        ("09 dataset (YOLOv8s, mono)",
         cm[(cm.detector == "YOLOv8s") & (cm.geometry == "mono")]),
    ]
    for ax, (title, sub) in zip(axes, groups):
        labs = sub[sub.task == "longitudinal"].label.tolist()
        lo = sub[sub.task == "longitudinal"].eta_perception_oracle_20.to_numpy()
        la = sub[sub.task == "lateral"].eta_perception_oracle_20.to_numpy()
        x = np.arange(len(labs))
        ax.bar(x - 0.2, lo, 0.38, color=SERIES[0], label="longitudinal", zorder=3)
        ax.bar(x + 0.2, la, 0.38, color=SERIES[1], label="lateral", zorder=3)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(1.0, color=MUTED, lw=1.4, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); _tidy(ax)
    axes[0].set_ylabel("η of the perception-gain oracle @20")
    axes[0].legend(fontsize=9)
    fig.tight_layout(); save(fig, "07_09_generalization")

    # 10 geometry control
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    g = cm[cm.geometry.isin(["mono", "oracle"])]
    labs = sorted(set(g.apply(lambda r: f"{r.dataset}/{r.task}", axis=1)))
    x = np.arange(len(labs))
    for i, geo in enumerate(("mono", "oracle")):
        v = [float(g[(g.geometry == geo) &
                     (g.apply(lambda r: f"{r.dataset}/{r.task}", axis=1) == l)]
                   .eta_perception_oracle_20.mean()) for l in labs]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=SERIES[i], zorder=3,
               label="deployed monocular" if geo == "mono" else "oracle range")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("η perception oracle @20")
    ax.set_title("Better geometry narrows the gap but does not close it")
    ax.legend(); _tidy(ax)
    fig.tight_layout(); save(fig, "10_geometry_control")

    # 11 temporal replay
    tr = Path(sorted(glob.glob(str(RAW / "*_temporal_kitti")))[-1]) / "temporal.csv"
    if tr.exists():
        t = pd.read_csv(tr)
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharex=True)
        pols = ["random", "uncertainty", "criticality", "ego speed",
                "PERCEPTION-GAIN ORACLE", "DECISION-VALUE ORACLE (long)"]
        for ax, (colname, title) in zip(axes, [("eta_long", "A  longitudinal sequence cost"),
                                               ("eta_lat", "B  lateral sequence cost")]):
            for i, p in enumerate(pols):
                s = t[t.policy == p].sort_values("quota")
                if not len(s):
                    continue
                colr = MUTED if "ORACLE (long)" in p else SERIES[i]
                ax.plot(s.quota * 100, s[colname], "--o" if "ORACLE" in p else "-o",
                        ms=6, color=colr, label=p.lower())
            ax.axhline(0, color=MUTED, lw=1)
            ax.set_xlabel("FULL-compute quota (%)"); ax.set_title(title); _tidy(ax)
        axes[0].set_ylabel("η under temporal replay")
        axes[0].legend(fontsize=8, loc="upper left")
        fig.tight_layout(); save(fig, "11_temporal_replay")

    # 12 trivial heuristics
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    sub = cm.sort_values(["task", "best_trivial_eta20"])
    x = np.arange(len(sub))
    ax.bar(x, sub.best_trivial_eta20, 0.6,
           color=[SERIES[0] if t == "longitudinal" else SERIES[1] for t in sub.task], zorder=3)
    ax.axhline(0.9, color=SERIES[7], lw=1.6, ls="--")
    ax.annotate("F7 threshold 0.90", (len(sub) - 0.5, 0.91), ha="right", fontsize=9,
                color=SERIES[7])
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.label}\n{r.best_trivial_heuristic}" for _, r in sub.iterrows()],
                       rotation=32, ha="right", fontsize=7)
    ax.set_ylabel("η@20 of the best trivial heuristic")
    ax.set_title("The best 1–2 variable heuristic, and it is a different one each time")
    h = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=SERIES[1])]
    ax.legend(h, ["longitudinal", "lateral"], fontsize=9)
    _tidy(ax)
    fig.tight_layout(); save(fig, "12_trivial_heuristics")

    # 13 Jetson trade-off
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    sub = cm[(cm.task == "longitudinal") & cm.compute_ratio.notna()].drop_duplicates("label")
    for ax, (xc, xl) in zip(axes, [("compute_ratio", "GPU inference ratio  FULL / CHEAP"),
                                   ("energy_ratio", "GPU energy ratio  FULL / CHEAP")]):
        ax.scatter(sub[xc], sub.eta_perception_oracle_20, s=70, color=SERIES[0], zorder=4)
        for _, r in sub.iterrows():
            ax.annotate(r.label, (r[xc], r.eta_perception_oracle_20),
                        textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK2)
        ax.axhline(1.0, color=MUTED, ls="--", lw=1.4)
        ax.set_xlabel(xl); _tidy(ax); ax.grid(True, axis="x")
    axes[0].set_ylabel("η perception oracle @20")
    fig.suptitle("Real hardware cost is at stake, and perception knowledge does not direct it",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout(); save(fig, "13_jetson_tradeoff")

    # 14 counterexamples
    d = tables[pick[0]]
    jl = (d.J_cheap - d.J_full).to_numpy(); jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
    dE = d.dE.to_numpy()
    cats = {
        "large ΔE\nzero ΔJ": (dE >= 2) & (np.abs(jl) < 1e-9),
        "ΔE ≤ 0\nΔJ > 0": (dE <= 0) & (jl > 1e-9),
        "better perception\nworse decision": (dE > 0) & (jl < -1e-9),
        "long-only\nvalue": (jl > 1e-9) & (np.abs(jt) < 1e-9),
        "lateral-only\nvalue": (jt > 1e-9) & (np.abs(jl) < 1e-9),
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    v = [100 * m.mean() for m in cats.values()]
    ax.bar(np.arange(len(v)), v, 0.58, color=SERIES[:len(v)], zorder=3)
    ax.set_xticks(np.arange(len(v))); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("% of KITTI frames")
    ax.set_title("Every counterexample type is common, not anecdotal")
    for i, q in enumerate(v):
        ax.annotate(f"{q:.1f}%", (i, q), ha="center", va="bottom", fontsize=10, color=INK2)
    _tidy(ax)
    fig.tight_layout(); save(fig, "14_counterexamples")

    print(f"\nwrote figures to {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Phase 0F Stage 1 figures: how far published planning-aware metrics get.

Reads the eta table and regret curves written by 62_planning_metric_eta.py, plus the
joined per-frame table it saves, and draws
  fig0f1  eta vs compute quota, every allocation signal, both tasks, bootstrap bands
  fig0f2  each signal against the decision gain dJ, with rank statistics
  fig0f3  top-20% frame-set overlap between signals
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from rap.paths import RESULTS                       # noqa: E402
from rap.viz import GRID, INK, INK2, MUTED, SERIES, _tidy   # noqa: E402

OUT = Path(RESULTS) / "final" / "figures"
TASKS = ["longitudinal", "lateral"]

# one stable colour per signal family, so the three figures read together
STYLE = {
    "random":                 (MUTED,     "--", 1.4),
    "uncertainty":            (SERIES[4], "-",  1.6),
    "criticality":            (SERIES[3], "-",  1.6),
    "perception_oracle_dE":   (SERIES[2], "-",  1.8),
    "multimetric_dE_oracle":  (SERIES[5], "-",  1.8),
    "PKL_gain":               (SERIES[0], "-",  2.4),
    "TIP_gain":               (SERIES[1], "-",  2.4),
    "decision_oracle_dJ":     (INK,       ":",  2.0),
}


def style_for(name):
    for k, v in STYLE.items():
        if name.startswith(k) or k in name:
            return v
    return (SERIES[7], "-", 1.6)


def label_for(name):
    return {"perception_oracle_dE": "perception oracle ΔE",
            "multimetric_dE_oracle": "multi-metric ΔE oracle",
            "PKL_gain": "PKL gain (CVPR'20)",
            "TIP_gain": "TIP gain (ICML'23)",
            "decision_oracle_dJ": "decision oracle ΔJ"}.get(name, name)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def fig_eta_vs_quota(cur):
    fig, axes = plt.subplots(1, len(TASKS), figsize=(11.4, 4.3), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), TASKS):
        s = cur[cur.task == t]
        if s.eta.notna().sum() == 0:
            ax.text(0.5, 0.5, f"{t}: oracle gain is zero on this subset",
                    ha="center", va="center", color=MUTED, transform=ax.transAxes)
            _tidy(ax); ax.set_title(t); continue
        for name, g in s.groupby("signal", sort=False):
            g = g.sort_values("quota")
            c, ls, lw = style_for(name)
            q = g.quota * 100
            ax.plot(q, g.eta, color=c, ls=ls, lw=lw, label=label_for(name),
                    marker="o", ms=3.5)
            if {"eta_lo", "eta_hi"} <= set(g.columns) and g.eta_lo.notna().any():
                ax.fill_between(q, g.eta_lo, g.eta_hi, color=c, alpha=0.10, lw=0)
        ax.axhline(0.8, color=INK2, lw=1, ls="--")
        ax.annotate("kill-test threshold 0.8", xy=(12, 0.8), xytext=(12, 0.84),
                    color=INK2, fontsize=8.5)
        ax.axhline(0.6, color=GRID, lw=1)
        ax.set_xlabel("compute quota (% of frames run at full fidelity)")
        ax.set_title(t)
        _tidy(ax)
    np.atleast_1d(axes)[0].set_ylabel("η  (share of decision-oracle gain captured)")
    np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8.5, loc="upper left",
                                   bbox_to_anchor=(1.02, 1.0))
    save(fig, "fig0f1_eta_vs_quota")


def fig_signal_vs_dj(d, signals, task):
    dj = d[f"_dJ_{task}"].to_numpy()
    cols = [c for c in signals if c in d.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        x = d[c].to_numpy()
        ax.scatter(x, dj, s=6, alpha=0.30, color=style_for(c)[0], lw=0)
        rho = stats.spearmanr(x, dj).correlation
        tau = stats.kendalltau(x, dj).correlation
        ax.axhline(0, color=MUTED, lw=0.9, ls="--")
        ax.axvline(0, color=MUTED, lw=0.9, ls="--")
        ax.set_title(f"{label_for(c)}\nSpearman {rho:+.3f}   Kendall {tau:+.3f}",
                     fontsize=9.5)
        ax.set_xlabel(label_for(c))
        _tidy(ax)
    np.atleast_1d(axes)[0].set_ylabel(f"ΔJ, {task}")
    save(fig, f"fig0f2_signal_vs_dJ_{task}")


def fig_overlap(d, signals, task, frac=0.20):
    cols = [c for c in signals if c in d.columns] + [f"_dJ_{task}"]
    k = int(round(frac * len(d)))
    top = {c: set(np.argsort(-d[c].to_numpy(), kind="stable")[:k]) for c in cols}
    M = np.array([[len(top[a] & top[b]) / max(k, 1) for b in cols] for a in cols])
    fig, ax = plt.subplots(figsize=(0.72 * len(cols) + 3.2, 0.72 * len(cols) + 2.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    names = [label_for(c) if not c.startswith("_dJ") else "decision oracle ΔJ" for c in cols]
    ax.set_xticks(range(len(cols)), names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols)), names, fontsize=8.5)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color=INK if M[i, j] < 0.6 else "white")
    ax.set_title(f"overlap of the top {int(frac*100)}% frames — {task}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    save(fig, f"fig0f3_top{int(frac*100)}_overlap_{task}")


def main():
    run = sorted(Path(RESULTS).parent.glob("results/raw/*phase0f_eta*"))[-1]
    cur = pd.read_csv(run / "phase0f_regret_curves.csv")
    d = pd.read_pickle(run / "joined_frames.pkl")
    print(f"reading {run.name}: {len(d)} frames")
    signals = [c for c in ("unc_sum", "crit_sum", "dE", "PKL_gain", "TIP_gain")
               if c in d.columns]
    d = d.rename(columns={"unc_sum": "uncertainty", "crit_sum": "criticality",
                          "dE": "perception_oracle_dE"})
    signals = ["uncertainty", "criticality", "perception_oracle_dE",
               "PKL_gain", "TIP_gain"]
    signals = [s for s in signals if s in d.columns]
    fig_eta_vs_quota(cur)
    for t in TASKS:
        if f"_dJ_{t}" not in d.columns:
            continue
        fig_signal_vs_dj(d, signals, t)
        fig_overlap(d, signals, t)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

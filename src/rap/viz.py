"""Figures for the Phase-0 report.

Palette slots are assigned in fixed order and never cycled; forms that compare
across all pairs (scatter) stay within the first three slots.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ("#2a78d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": INK2,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "axes.titlesize": 11, "axes.titleweight": "600", "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "lines.linewidth": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})


def _tidy(ax, ygrid=True):
    ax.grid(ygrid, axis="y", zorder=0)
    ax.set_axisbelow(True)


def value_heterogeneity(df, path, value_col="value_task", label="Value_task"):
    v = np.sort(df[value_col].to_numpy())[::-1]
    n = len(v)
    frac = np.arange(1, n + 1) / n
    pos = np.clip(v, 0, None)
    cum = np.cumsum(pos) / max(pos.sum(), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    ax.plot(frac * 100, v, color=SERIES[0])
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("frames, ranked by value of running FULL (%)")
    ax.set_ylabel(f"{label} (risk reduction)")
    ax.set_title(f"A  Most frames gain nothing from FULL")
    _tidy(ax)
    share_zero = float(np.mean(np.abs(df[value_col]) <= 1e-9)) * 100
    ax.annotate(f"{share_zero:.0f}% of frames: exactly zero",
                xy=(0.97, 0.9), xycoords="axes fraction", ha="right",
                color=INK2, fontsize=9)

    ax = axes[1]
    ax.plot(frac * 100, cum * 100, color=SERIES[0], label="actual")
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.5, ls="--", label="if value were uniform")
    for p in (10, 20):
        y = cum[max(int(round(p / 100 * n)) - 1, 0)] * 100
        ax.plot([p], [y], "o", ms=8, color=SERIES[1], zorder=5)
        ax.annotate(f"top {p}% of frames\ncarry {y:.0f}% of the gain",
                    xy=(p, y), xytext=(p + 8, y - 18), color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("frames, ranked (%)")
    ax.set_ylabel("cumulative share of total risk reduction (%)")
    ax.set_title("B  The gain is concentrated")
    ax.legend(loc="lower right")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def compute_profile(summary: dict, path, order=None):
    names = order or list(summary)
    px = [summary[n]["pixels"] / 1000 for n in names]
    lat = [summary[n]["lat_e2e_ms_median"] for n in names]
    p95 = [summary[n]["lat_e2e_ms_p95"] for n in names]
    pwr = [summary[n].get("GPU_mw_over_idle", np.nan) / 1000 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    x = np.arange(len(names))
    ax = axes[0]
    ax.bar(x - 0.19, lat, 0.36, color=SERIES[0], label="median", zorder=3)
    ax.bar(x + 0.19, p95, 0.36, color=SERIES[1], label="p95", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("A  Latency per frame")
    ax.legend(); _tidy(ax)
    for xi, v in zip(x, lat):
        ax.annotate(f"{v:.1f}", (xi - 0.19, v), ha="center", va="bottom", fontsize=8, color=INK2)

    ax = axes[1]
    ax.bar(x, pwr, 0.55, color=SERIES[0], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("GPU rail power over idle (W)")
    ax.set_title("B  Power draw")
    _tidy(ax)

    ax = axes[2]
    ax.plot(px, lat, "o-", ms=8, color=SERIES[0])
    for p, l, n in zip(px, lat, names):
        ax.annotate(n, (p, l), textcoords="offset points", xytext=(6, -3),
                    fontsize=8, color=INK2)
    ax.set_xlabel("network input (kilopixels)")
    ax.set_ylabel("median latency (ms)")
    ax.set_title("C  Latency scales with input")
    _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def predictability(pred_df, path, model="gbm", target="value_task", metric="spearman",
                   per_seq=None, ylabel=None):
    d = pred_df[(pred_df.model == model) & (pred_df.target == target)]
    arms = list(d.arm)
    vals = list(d[metric])
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    colors = [SERIES[3] if a.startswith(("A_", "B_", "C_", "F_")) else SERIES[0] for a in arms]
    x = np.arange(len(arms))
    ax.bar(x, vals, 0.6, color=colors, zorder=3)
    if per_seq is not None:
        for xi, a in zip(x, arms):
            pts = np.array([v for v in per_seq.get(a, {}).values() if np.isfinite(v)])
            if len(pts):
                ax.scatter(np.full(len(pts), xi) + np.random.default_rng(0).normal(0, .06, len(pts)),
                           pts, s=14, color=INK2, alpha=0.45, zorder=4, linewidths=0)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([a.replace("_", " ") for a in arms], rotation=18, ha="right")
    ax.set_ylabel(ylabel or f"out-of-fold {metric} (Value_task)")
    ax.set_title(f"Predicting where FULL pays off — {model}, leave-one-sequence-out")
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:+.3f}", (xi, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[3]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[0])]
    ax.legend(handles, ["visual information only", "includes downstream criticality"],
              loc="upper left")
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def budget_curves(bud, path, policies, mode="pooled", ycol="eta",
                  ylabel="fraction of the oracle's risk reduction captured"):
    assert len(policies) <= len(SERIES), "hues are assigned in fixed order, never cycled"
    d = bud[bud["mode"] == mode]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, p in enumerate(policies):
        sub = d[d.policy == p].sort_values("quota")
        if not len(sub):
            continue
        color = MUTED if p == "oracle" else SERIES[i]
        ls = "--" if p in ("oracle", "random") else "-"
        ax.plot(sub.quota * 100, sub[ycol], ls, marker="o", ms=7, color=color,
                label=p.replace("_", " "))
    ax.set_xlabel("FULL-compute quota (% of frames)")
    ax.set_ylabel(ylabel)
    ax.set_title("Same compute budget, different frames")
    ax.legend(loc="best", ncol=1)
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def matched_pairs(pairs_df, path, nbins=12):
    d = pairs_df
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.scatter(d.d_crit, d.d_value, s=6, color=SERIES[0], alpha=0.12, linewidths=0)
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Δ predicted criticality  (frame A − frame B)")
    ax.set_ylabel("Δ Value_task  (A − B)")
    ax.set_title("A  Uncertainty-matched pairs")
    _tidy(ax); ax.grid(True, axis="x")

    ax = axes[1]
    q = np.quantile(d.d_crit, np.linspace(0, 1, nbins + 1))
    q = np.unique(q)
    idx = np.clip(np.digitize(d.d_crit, q[1:-1]), 0, len(q) - 2)
    xs, ys, es = [], [], []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() < 30:
            continue
        xs.append(float(np.mean(d.d_crit[m])))
        ys.append(float(np.mean(d.d_value[m])))
        es.append(float(np.std(d.d_value[m]) / np.sqrt(m.sum())))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=8, color=SERIES[0], ecolor=MUTED,
                elinewidth=1.5, capsize=3)
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Δ predicted criticality (binned)")
    ax.set_ylabel("mean Δ Value_task ± s.e.")
    ax.set_title("B  Criticality still orders the pairs")
    _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def risk_per_ms(bud, path, policies, mode="pooled"):
    assert len(policies) <= len(SERIES), "hues are assigned in fixed order, never cycled"
    d = bud[(bud["mode"] == mode)]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, p in enumerate(policies):
        sub = d[d.policy == p].sort_values("quota")
        if not len(sub) or not np.isfinite(sub["risk_per_1000_extra_ms"]).any():
            continue
        color = MUTED if p == "oracle" else SERIES[i]
        ax.plot(sub.quota * 100, sub["risk_per_1000_extra_ms"], "--o" if p == "oracle" else "-o",
                ms=7, color=color, label=p.replace("_", " "))
    ax.set_xlabel("FULL-compute quota (% of frames)")
    ax.set_ylabel("risk reduction per extra second of GPU time")
    ax.set_title("What each millisecond of extra perception buys")
    ax.legend(); _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

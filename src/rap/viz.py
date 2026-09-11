"""Figures for the Phase-0 report.

Palette slots are assigned in fixed order and never cycled; forms that compare
across all pairs (scatter) stay within the first three slots.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

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
    cum = np.cumsum(v) / v.sum()      # same definition as gain_in_top_*

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
                    xy=(p, y), xytext=(p + 14, y - 26), color=INK2, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_xlabel("frames, ranked (%)")
    ax.set_ylabel("cumulative share of total risk reduction (%)")
    ax.set_ylim(0, 118)
    ax.set_title("B  The gain is concentrated")
    ax.legend(loc="lower right")
    _tidy(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def compute_profile(summary: dict, path, order=None):
    names = order or [k for k in summary if not k.startswith("_")]
    px = [summary[n]["pixels"] / 1000 for n in names]
    pre = [summary[n]["lat_pre_ms_median"] for n in names]
    fwd = [summary[n]["lat_fwd_ms_median"] for n in names]
    post = [summary[n]["lat_post_ms_median"] for n in names]
    e2e = [summary[n]["lat_e2e_ms_median"] for n in names]
    p95 = [summary[n]["lat_e2e_ms_p95"] for n in names]
    energy = [summary[n].get("energy_gpu_mj_per_frame", np.nan) for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    x = np.arange(len(names))

    # A: where the time goes. A 2px surface gap separates the stacked segments.
    ax = axes[0]
    bottom = np.zeros(len(names))
    for vals, lab, col in [(pre, "pre-process", SERIES[2]),
                           (fwd, "GPU inference", SERIES[0]),
                           (post, "post-process (NMS, host)", SERIES[3])]:
        ax.bar(x, vals, 0.58, bottom=bottom, color=col, label=lab, zorder=3,
               edgecolor=SURFACE, linewidth=2)
        bottom += np.asarray(vals)
    ax.plot(x, p95, "o", ms=7, color=INK2, zorder=5, label="p95 end-to-end")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("latency per frame (ms)")
    ax.set_title("A  Only GPU inference scales")
    ax.legend(fontsize=8, loc="upper left"); _tidy(ax)

    ax = axes[1]
    ax.bar(x, energy, 0.58, color=SERIES[0], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("GPU energy over idle (mJ / frame)")
    ax.set_title("B  Energy per frame")
    for xi, v in zip(x, energy):
        if np.isfinite(v):
            ax.annotate(f"{v:.0f}", (xi, v), ha="center", va="bottom", fontsize=8, color=INK2)
    _tidy(ax)

    ax = axes[2]
    ax.plot(px, fwd, "o-", ms=8, color=SERIES[0], label="GPU inference")
    ax.plot(px, e2e, "o--", ms=8, color=SERIES[3], label="end-to-end")
    for p_, l, n in zip(px, fwd, names):
        ax.annotate(n, (p_, l), textcoords="offset points", xytext=(6, -10),
                    fontsize=8, color=INK2)
    ax.set_xlabel("network input (kilopixels)")
    ax.set_ylabel("median latency (ms)")
    ax.set_title("C  Latency vs input size")
    ax.legend(fontsize=8); _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


VISUAL_ARMS = {"A_conf", "B_uncertainty", "C_complexity", "F_all_visual"}
STAKES_ARMS = {"H_visual_plus_stakes"}


def predictability(pred_df, path, model="gbm", target="value_task", metric="spearman",
                   per_seq=None, ylabel=None):
    d = pred_df[(pred_df.model == model) & (pred_df.target == target)]
    arms = list(d.arm)
    vals = list(d[metric])
    base = 0.5 if metric == "auc" else 0.0
    fig, ax = plt.subplots(figsize=(9.2, 4.4))

    def colour(a):
        if a in VISUAL_ARMS:
            return SERIES[3]
        return SERIES[2] if a in STAKES_ARMS else SERIES[0]

    x = np.arange(len(arms))
    ax.bar(x, np.asarray(vals) - base, 0.6, bottom=base,
           color=[colour(a) for a in arms], zorder=3)
    if per_seq is not None:
        rng = np.random.default_rng(0)
        for xi, a in zip(x, arms):
            pts = np.array([v for v in per_seq.get(a, {}).values() if np.isfinite(v)])
            if len(pts):
                ax.scatter(np.full(len(pts), xi) + rng.normal(0, .07, len(pts)), pts,
                           s=16, color=INK2, alpha=0.4, zorder=4, linewidths=0)
    ax.axhline(base, color=MUTED, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ") for a in arms], rotation=18, ha="right")
    ax.set_ylabel(ylabel or "out-of-fold Spearman with Value_task")
    ax.set_title(f"Predicting where FULL pays off — {model}, leave-one-sequence-out")
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:+.3f}" if metric != "auc" else f"{v:.3f}", (xi, v), ha="center",
                    va="bottom" if v >= base else "top", fontsize=9, color=INK2,
                    xytext=(0, 4 if v >= base else -12), textcoords="offset points")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[3]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[2]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[0])]
    ax.legend(handles, ["visual information only", "visual + total-stakes scalar",
                        "visual + full criticality"], loc="upper left", fontsize=9)
    _tidy(ax)
    if per_seq is not None:
        ax.annotate("dots: individual held-out sequences", xy=(0.99, 0.02),
                    xycoords="axes fraction", ha="right", fontsize=8, color=MUTED)
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
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
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
    dc, dv = d.d_crit.to_numpy(), d.d_value.to_numpy()
    tie = np.abs(dc) < 1e-9
    q = np.unique(np.quantile(dc[~tie], np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(dc[~tie], q[1:-1]), 0, len(q) - 2)
    xs, ys, es = [], [], []
    for b in range(len(q) - 1):
        m = idx == b
        if m.sum() < 30:
            continue
        xs.append(float(np.mean(dc[~tie][m])))
        ys.append(float(np.mean(dv[~tie][m])))
        es.append(float(np.std(dv[~tie][m]) / np.sqrt(m.sum())))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", ms=8, color=SERIES[0], ecolor=MUTED,
                elinewidth=1.5, capsize=3, label="pairs that differ in criticality")
    if tie.sum() >= 30:
        ax.errorbar([0], [float(np.mean(dv[tie]))],
                    yerr=[float(np.std(dv[tie]) / np.sqrt(tie.sum()))], fmt="s", ms=9,
                    color=MUTED, ecolor=MUTED, elinewidth=1.5, capsize=3,
                    label=f"equal criticality (n={int(tie.sum())})")
    ax.legend(fontsize=8, loc="upper left")
    ax.axhline(0, color=MUTED, lw=1); ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Δ predicted criticality (binned)")
    ax.set_ylabel("mean Δ Value_task ± s.e.")
    ax.set_title("B  Criticality still orders the pairs")
    _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def budget_contrast(bud, path, policies, mode="pooled"):
    """Same selections, two objectives. Uncertainty routing and criticality routing
    are not competing answers to one question — they answer different questions."""
    assert len(policies) <= len(SERIES), "hues are assigned in fixed order, never cycled"
    d = bud[bud["mode"] == mode]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    panels = [("eta", "share of the oracle's RISK reduction captured",
               "A  Objective: downstream risk"),
              ("std_err_captured", "share of the full-compute DETECTION gain captured",
               "B  Objective: standard detection error")]
    for ax, (col, ylab, title) in zip(axes, panels):
        if col not in d:
            continue
        for i, p in enumerate(policies):
            sub = d[d.policy == p].sort_values("quota")
            if not len(sub):
                continue
            color = MUTED if p == "oracle" else SERIES[i]
            ls = "--" if p in ("oracle", "random") else "-"
            ax.plot(sub.quota * 100, sub[col], ls, marker="o", ms=7, color=color,
                    label=p.replace("_", " "))
        ax.set_xlabel("FULL-compute quota (% of frames)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        _tidy(ax)
    axes[0].legend(loc="lower right", fontsize=9)
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


def mechanism(by_distance: pd.DataFrame, path):
    """The opposing gradients: extra compute recovers distant objects, criticality lives near.

    Panel B plots two quantities that are both shares of their own total, so they
    belong on one axis; mixing a probability with a mass would not.
    """
    labels = [str(i) for i in by_distance.index]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    ax.bar(x - 0.19, by_distance["recall_cheap"], 0.36, color=SERIES[3], label="CHEAP", zorder=3,
           edgecolor=SURFACE, linewidth=2)
    ax.bar(x + 0.19, by_distance["recall_full"], 0.36, color=SERIES[0], label="FULL", zorder=3,
           edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("ego distance (m)")
    ax.set_ylabel("recall of ground-truth objects")
    ax.set_title("A  Extra compute buys distant objects")
    ax.legend(); _tidy(ax)

    ax = axes[1]
    ax.plot(x, by_distance["crit_mass"], "-o", ms=8, color=SERIES[1],
            label="share of all criticality")
    ax.plot(x, by_distance["share_of_risk_recovered"], "-o", ms=8, color=SERIES[0],
            label="share of risk FULL recovers")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("ego distance (m)")
    ax.set_ylabel("share of total")
    ax.set_title("B  …but criticality lives close to the ego")
    ax.legend(); _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def sensitivity_dots(sens: pd.DataFrame, path, controls=("uniform", "proximity")):
    """Does the budget gain survive every alternative definition of risk?

    Two rows are negative controls by construction and are drawn apart from the rest:
    `uniform` makes criticality constant (so the target collapses to plain detection
    value) and `proximity` makes it a pure function of distance, which apparent object
    size — and therefore visual uncertainty — already encodes.
    """
    d = sens.copy()
    d["d_eta"] = d["eta20_learned_unc_crit"] - d["eta20_learned_uncertainty"]
    d["label"] = d["axis"].str.replace("_", " ") + " = " + d["value"].astype(str)
    d["is_control"] = d["value"].astype(str).isin(controls)
    d = d.sort_values(["is_control", "d_eta"])
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.6, 0.30 * len(d) + 1.6))
    ax.axvline(0, color=MUTED, lw=1.2)
    for mask, colour, lab in [(~d.is_control, SERIES[0], "criticality encodes ego-path / TTC"),
                              (d.is_control, SERIES[1], "negative control")]:
        m = mask.to_numpy()
        ax.scatter(d.d_eta[m], y[m], s=64, color=colour, zorder=4, linewidths=0, label=lab)
        ax.hlines(y[m], 0, d.d_eta[m], color=colour, lw=2, alpha=0.35, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label, fontsize=8)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlabel("Δη at a 20 % quota  (uncertainty + criticality  −  uncertainty alone)")
    ax.set_title("The budget gain under every alternative definition of risk")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fail_vs_recover(obj: pd.DataFrame, p_fail, p_recover, path, nbins=10):
    """Failure and recoverability are different questions — three ways of showing it."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    fail = obj["cheap_fail"].to_numpy(bool)

    # A: among objects cheap got wrong, does extra compute actually fix them? By distance.
    ax = axes[0]
    d = obj.loc[fail, "gt_dist"].to_numpy()
    ok = obj.loc[fail, "full_ok"].to_numpy(float)
    edges = np.array([0, 10, 20, 30, 45, 60, 200])
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 30:
            continue
        xs.append(f"{lo}–{hi if hi < 200 else '∞'}")
        ys.append(float(ok[m].mean()))
        ns.append(int(m.sum()))
    x = np.arange(len(xs))
    ax.bar(x, ys, 0.6, color=SERIES[0], zorder=3)
    ax.axhline(float(ok.mean()), color=SERIES[1], lw=2, ls="--",
               label=f"overall {ok.mean():.2f}")
    ax.set_xticks(x); ax.set_xticklabels(xs, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("ego distance (m)")
    ax.set_ylabel("P(FULL recovers | CHEAP failed)")
    ax.set_title("A  Recoverability depends strongly on range")
    ax.legend(); _tidy(ax)
    for xi, v, n in zip(x, ys, ns):
        ax.annotate(f"n={n:,}", (xi, 0.03), ha="center", fontsize=7.5, color=INK2)

    # B: the two predicted factors are correlated but far from interchangeable
    ax = axes[1]
    pf, pr = np.asarray(p_fail)[fail], np.asarray(p_recover)[fail]
    ax.scatter(pf, pr, s=5, alpha=0.08, color=SERIES[0], linewidths=0)
    q = np.quantile(pf, np.linspace(0, 1, nbins + 1))
    q = np.unique(q)
    idx = np.clip(np.digitize(pf, q[1:-1]), 0, len(q) - 2)
    bx = [pf[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 30]
    by = [pr[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 30]
    ax.plot(bx, by, "-o", ms=8, color=SERIES[1], label="binned mean")
    ax.set_xlabel("predicted P(cheap fails)")
    ax.set_ylabel("predicted P(full recovers | fails)")
    ax.set_title(f"B  Correlated, not the same (r = {np.corrcoef(pf, pr)[0,1]:+.2f})")
    ax.legend(); _tidy(ax); ax.grid(True, axis="x")

    # C: how well each factor can be predicted at all
    ax = axes[2]
    from sklearn.metrics import roc_auc_score
    names, aucs, bases = [], [], []
    for lab, y, p, m in [("P(cheap fails)", obj["cheap_fail"].to_numpy(float), p_fail, slice(None)),
                         ("P(recovers | fails)", obj["full_ok"].to_numpy(float), p_recover, fail),
                         ("P(compute helps)", (obj["gain"] > 0).to_numpy(float), p_fail, slice(None))]:
        yy, pp = np.asarray(y)[m], np.asarray(p)[m]
        if lab == "P(compute helps)":
            continue
        names.append(lab); aucs.append(roc_auc_score(yy, pp)); bases.append(float(yy.mean()))
    x = np.arange(len(names))
    ax.bar(x, np.array(aucs) - 0.5, 0.5, bottom=0.5, color=[SERIES[0], SERIES[3]], zorder=3)
    ax.axhline(0.5, color=MUTED, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("out-of-fold AUC")
    ax.set_title("C  Recoverability is barely predictable")
    for xi, v in zip(x, aucs):
        ax.annotate(f"{v:.3f}", (xi, v), ha="center", va="bottom", fontsize=10, color=INK2)
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def phase0b_budget(bud: pd.DataFrame, path, policies, labels=None, mode="pooled"):
    assert len(policies) <= len(SERIES), "hues are assigned in fixed order, never cycled"
    d = bud[bud["mode"] == mode]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, p in enumerate(policies):
        sub = d[d.policy == p].sort_values("quota")
        if not len(sub):
            continue
        color = MUTED if p == "oracle" else SERIES[i]
        ls = "--" if p in ("oracle", "random") else "-"
        ax.plot(sub.quota * 100, sub["eta"], ls, marker="o", ms=7, color=color,
                label=(labels or {}).get(p, p.replace("_", " ")))
    ax.set_xlabel("FULL-compute quota (% of frames)")
    ax.set_ylabel("share of the oracle's risk reduction captured")
    ax.set_title("Adding a recoverability factor changes nothing")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def decision_vs_signals(df, path):
    """None of the Phase-0 signals explain decision gain."""
    from scipy import stats as _st
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))
    panels = [("dE", "perception gain  (detections recovered)", SERIES[0], "A"),
              ("feat_crit_sum", "estimated criticality", SERIES[1], "B"),
              ("feat_binent_sum", "visual uncertainty", SERIES[3], "C")]
    for ax, (col, lab, colr, tag) in zip(axes, panels):
        x, y = df[col].to_numpy(float), df["dJ"].to_numpy(float)
        ax.scatter(x, y, s=5, alpha=0.10, color=colr, linewidths=0)
        q = np.unique(np.quantile(x, np.linspace(0, 1, 13)))
        idx = np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2)
        bx = [x[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 40]
        by = [y[idx == b].mean() for b in range(len(q) - 1) if (idx == b).sum() > 40]
        ax.plot(bx, by, "-o", ms=7, color=INK2, label="binned mean")
        ax.axhline(0, color=MUTED, lw=1)
        # a handful of extreme frames otherwise flatten the whole panel; clip the view
        # (all points are still used for the correlation and the binned means)
        lo, hi = np.percentile(y, [1, 99])
        pad = 0.15 * max(hi - lo, 1e-6)
        ax.set_ylim(lo - pad, hi + pad)
        r = _st.spearmanr(x, y).correlation
        xlo, xhi = np.percentile(x, [0.5, 99.5])
        ax.set_xlim(xlo - 0.05 * abs(xhi - xlo) - 1e-6, xhi + 0.05 * abs(xhi - xlo) + 1e-6)
        ax.set_xlabel(lab); ax.set_title(f"{tag}  ρ = {r:+.3f}")
        if tag == "A":
            ax.set_ylabel("decision gain  ΔJ")
        ax.legend(fontsize=8); _tidy(ax); ax.grid(True, axis="x")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def action_invariance(df, path):
    """How much perception improvement reaches the decision at all."""
    improved = (df["dE"] > 0).to_numpy()
    changed = (df["same_action"] == 0).to_numpy()
    ben = (df["dJ"] > 1e-9).to_numpy()
    harm = (df["dJ"] < -1e-9).to_numpy()
    n = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    ax = axes[0]
    cats = ["same action\n(no decision effect)", "action changes\nfor the better",
            "action changes\nfor the worse"]
    vals = [100 * (improved & ~changed).sum() / improved.sum(),
            100 * (improved & changed & ben).sum() / improved.sum(),
            100 * (improved & changed & harm).sum() / improved.sum()]
    ax.bar(np.arange(3), vals, 0.58, color=[SERIES[3], SERIES[2], SERIES[7]], zorder=3)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("% of frames where detection improved")
    ax.set_title("A  Most perception gain never reaches the decision")
    for xi, v in zip(np.arange(3), vals):
        ax.annotate(f"{v:.0f}%", (xi, v), ha="center", va="bottom", fontsize=10, color=INK2)
    _tidy(ax)

    ax = axes[1]
    cats2 = ["perception\ndiffers", "action\ndiffers", "action change\nbeneficial",
             "action change\nharmful"]
    vals2 = [100 * ((df["dE"].abs() > 1e-9) | (df.a_req_cheap - df.a_req_full).abs().gt(0.05)).mean(),
             100 * changed.mean(), 100 * (changed & ben).mean(), 100 * (changed & harm).mean()]
    ax.bar(np.arange(4), vals2, 0.58,
           color=[SERIES[0], SERIES[0], SERIES[2], SERIES[7]], zorder=3)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(cats2, fontsize=9)
    ax.set_ylabel("% of all frames")
    ax.set_title("B  Where the funnel narrows")
    for xi, v in zip(np.arange(4), vals2):
        ax.annotate(f"{v:.0f}%", (xi, v), ha="center", va="bottom", fontsize=10, color=INK2)
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def decision_budget(bud, path, policies, labels=None):
    assert len(policies) <= len(SERIES), "hues are assigned in fixed order, never cycled"
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, p in enumerate(policies):
        sub = bud[bud.policy == p].sort_values("quota")
        if not len(sub):
            continue
        oracle = "ORACLE" in p
        color = MUTED if p == "DECISION-VALUE ORACLE" else SERIES[i]
        ls = "--" if oracle or p == "random" else "-"
        ax.plot(sub.quota * 100, sub["eta"], ls, marker="o", ms=7, color=color,
                label=(labels or {}).get(p, p))
    ax.set_xlabel("FULL-compute quota (% of frames)")
    ax.set_ylabel("share of achievable decision-cost reduction captured")
    ax.set_title("Knowing where perception improves is not knowing where decisions improve")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    _tidy(ax)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def failed_mechanism(df, ablation: dict, path):
    """The proposed decision-boundary mechanism does not hold; report it plainly."""
    from scipy import stats as _st
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax = axes[0]
    b = df["feat_bdist"].to_numpy(); adj = df["dJ"].abs().to_numpy()
    edges = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 10]
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (b >= lo) & (b < hi)
        if m.sum() < 50:
            continue
        xs.append(f"{lo}–{hi if hi < 10 else '∞'}"); ys.append(adj[m].mean()); ns.append(int(m.sum()))
    x = np.arange(len(xs))
    ax.bar(x, ys, 0.58, color=SERIES[3], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(xs, rotation=12, ha="right")
    ax.set_xlabel("|required decel − nearest action threshold|")
    ax.set_ylabel("mean |ΔJ|")
    r = _st.spearmanr(b, adj).correlation
    ax.set_title(f"A  No concentration near the boundary (ρ = {r:+.3f})")
    for xi, v, nn in zip(x, ys, ns):
        ax.annotate(f"n={nn:,}", (xi, 0.05), ha="center", fontsize=7.5, color=INK2)
    _tidy(ax)

    ax = axes[1]
    names = list(ablation); vals = [ablation[k] for k in names]
    colors = [SERIES[7] if v < 0.05 else SERIES[0] for v in vals]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, 0.6, color=colors, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("η at a 20% quota")
    ax.set_title("B  Ego speed carries it, not decision margin")
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:+.3f}", (max(v, 0) + 0.02, yi), va="center", fontsize=9, color=INK2)
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

"""Budgeted selection: at a fixed FULL-compute quota, which frames should get it?

Every policy is scored identically — rank frames, spend the budget on the top of the
ranking — so the only thing that differs between arms is what the score knows.
Learned scores arrive already out-of-fold from `predict.loso`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tie_fraction(score: np.ndarray, quota: float) -> float:
    """Share of the selected set that a tie at the cut decides rather than the score.

    Count-valued signals (a false-negative count, an unweighted dE) take few distinct
    values, so the quota can fall deep inside a tie group: at a 20% quota on 3,376 nuScenes
    frames only 326 frames are strictly above dE's cut value while 366 share it, leaving 52%
    of the selection to the tie-break.  Reported so a ranking that is mostly arbitrary can
    never be read as a property of the signal.
    """
    n = len(score)
    k = int(round(quota * n))
    if k <= 0 or k >= n:
        return 0.0
    cut = np.sort(np.asarray(score, float))[::-1][k - 1]
    above = int((np.asarray(score, float) > cut).sum())
    return (k - above) / k


def select_pooled(score: np.ndarray, quota: float, seed: int | None = None) -> np.ndarray:
    """Top-quota frames over the whole dataset, with ties broken at random.

    A stable argsort breaks ties by row order, which is not a property of the score: two
    signals that tie on most of the selected set would still get a definite, reproducible and
    meaningless ranking, and two independent implementations sharing the convention would
    agree with each other while both being arbitrary.  With `seed` given, ties are broken by a
    seeded random key instead, so callers can average over seeds; `seed=None` keeps the old
    deterministic order for backward compatibility with published tables.
    """
    score = np.asarray(score, float)
    k = int(round(quota * len(score)))
    sel = np.zeros(len(score), dtype=bool)
    if k <= 0:
        return sel
    if seed is None:
        order = np.argsort(-score, kind="stable")
    else:
        jitter = np.random.default_rng(seed).random(len(score))
        order = np.lexsort((jitter, -score))
    sel[order[:k]] = True
    return sel


def select_per_sequence(score: np.ndarray, groups: np.ndarray, quota: float) -> np.ndarray:
    """Top-quota frames within each sequence — enforces the budget locally."""
    sel = np.zeros(len(score), dtype=bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        k = int(round(quota * len(idx)))
        if k > 0:
            sel[idx[np.argsort(-score[idx], kind="stable")[:k]]] = True
    return sel


def total_risk(df: pd.DataFrame, sel: np.ndarray, risk_col=("risk_cheap", "risk_full")) -> float:
    rc = df[risk_col[0]].to_numpy()
    rf = df[risk_col[1]].to_numpy()
    return float(np.sum(np.where(sel, rf, rc)))


def evaluate(df: pd.DataFrame, scores: dict[str, np.ndarray], quotas, seeds=64,
             mode: str = "pooled", risk_col=("risk_cheap", "risk_full"),
             extra_cols=("err_std_cheap", "err_std_full")) -> pd.DataFrame:
    groups = df["seq"].to_numpy()
    rc_all = float(df[risk_col[0]].sum())
    rf_all = float(df[risk_col[1]].sum())
    pick = (lambda s, q: select_pooled(s, q)) if mode == "pooled" else \
           (lambda s, q: select_per_sequence(s, groups, q))
    oracle_score = (df[risk_col[0]] - df[risk_col[1]]).to_numpy()

    ec, ef = (df[extra_cols[0]].to_numpy(), df[extra_cols[1]].to_numpy()) if extra_cols else (None, None)
    rows = []
    for q in quotas:
        sel_oracle = pick(oracle_score, q)
        r_oracle = total_risk(df, sel_oracle, risk_col)
        span = rc_all - r_oracle

        rng = np.random.default_rng(0)
        rand_sel = [pick(rng.random(len(df)), q) for _ in range(seeds)]
        rand_risk = [total_risk(df, s, risk_col) for s in rand_sel]

        entries = {"random": (float(np.mean(rand_risk)), float(np.std(rand_risk)), rand_sel)}
        for name, s in scores.items():
            sel = pick(np.asarray(s, float), q)
            entries[name] = (total_risk(df, sel, risk_col), 0.0, [sel])
        entries["oracle"] = (r_oracle, 0.0, [sel_oracle])

        for name, (r, sd, sels) in entries.items():
            row = {
                "quota": q, "policy": name, "mode": mode,
                "total_risk": r, "risk_sd": sd,
                "risk_all_cheap": rc_all, "risk_all_full": rf_all, "risk_oracle": r_oracle,
                "risk_reduction": rc_all - r,
                "eta": (rc_all - r) / span if span > 1e-12 else np.nan,
                "frac_of_full_gain": (rc_all - r) / (rc_all - rf_all) if rc_all - rf_all > 1e-12 else np.nan,
                "n_selected": int(round(q * len(df))),
            }
            if ec is not None:
                row["std_err_total"] = float(np.mean([np.sum(np.where(s, ef, ec)) for s in sels]))
            rows.append(row)
    return pd.DataFrame(rows)


def add_compute_columns(res: pd.DataFrame, lat_cheap_ms: float, lat_full_ms: float,
                        energy_cheap_mj: float = np.nan,
                        energy_full_mj: float = np.nan) -> pd.DataFrame:
    res = res.copy()
    n = res["n_selected"]
    total = res["n_selected"] / res["quota"].replace(0, np.nan)
    extra_ms = n * (lat_full_ms - lat_cheap_ms)
    res["mean_latency_ms"] = lat_cheap_ms + res["quota"] * (lat_full_ms - lat_cheap_ms)
    res["extra_compute_ms"] = extra_ms
    res["risk_per_extra_ms"] = res["risk_reduction"] / extra_ms.replace(0, np.nan)
    res["risk_per_1000_extra_ms"] = res["risk_per_extra_ms"] * 1000
    if np.isfinite(energy_cheap_mj) and np.isfinite(energy_full_mj):
        res["mean_energy_mj"] = energy_cheap_mj + res["quota"] * (energy_full_mj - energy_cheap_mj)
        res["extra_energy_j"] = n * (energy_full_mj - energy_cheap_mj) / 1000.0
        res["risk_per_extra_joule"] = res["risk_reduction"] / res["extra_energy_j"].replace(0, np.nan)
    res["total_frames"] = total
    return res

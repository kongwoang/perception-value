"""Budgeted selection: at a fixed FULL-compute quota, which frames should get it?

Every policy is scored identically — rank frames, spend the budget on the top of the
ranking — so the only thing that differs between arms is what the score knows.
Learned scores arrive already out-of-fold from `predict.loso`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_pooled(score: np.ndarray, quota: float) -> np.ndarray:
    """Top-quota frames over the whole dataset."""
    k = int(round(quota * len(score)))
    sel = np.zeros(len(score), dtype=bool)
    if k > 0:
        sel[np.argsort(-score, kind="stable")[:k]] = True
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

    rows = []
    for q in quotas:
        sel_oracle = pick(oracle_score, q)
        r_oracle = total_risk(df, sel_oracle, risk_col)
        span = rc_all - r_oracle

        rng = np.random.default_rng(0)
        rand = [total_risk(df, pick(rng.random(len(df)), q), risk_col) for _ in range(seeds)]

        entries = {"random": (float(np.mean(rand)), float(np.std(rand))), }
        for name, s in scores.items():
            entries[name] = (total_risk(df, pick(np.asarray(s, float), q), risk_col), 0.0)
        entries["oracle"] = (r_oracle, 0.0)

        for name, (r, sd) in entries.items():
            sel = (pick(oracle_score, q) if name == "oracle"
                   else (None if name == "random" else pick(np.asarray(scores[name], float), q)))
            row = {
                "quota": q, "policy": name, "mode": mode,
                "total_risk": r, "risk_sd": sd,
                "risk_all_cheap": rc_all, "risk_all_full": rf_all, "risk_oracle": r_oracle,
                "risk_reduction": rc_all - r,
                "eta": (rc_all - r) / span if span > 1e-12 else np.nan,
                "frac_of_full_gain": (rc_all - r) / (rc_all - rf_all) if rc_all - rf_all > 1e-12 else np.nan,
                "n_selected": int(round(q * len(df))),
            }
            if sel is not None and extra_cols:
                row["std_err_total"] = float(np.sum(np.where(sel, df[extra_cols[1]], df[extra_cols[0]])))
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

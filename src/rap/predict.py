"""Can a deployable predictor see, from CHEAP output alone, which frames FULL will help?

Evaluation is leave-one-sequence-out: consecutive frames of a driving sequence are
near-duplicates, so a random split would let the model memorise a scene it is then
tested on. Every number here is out-of-fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from scipy import stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .features import assert_no_leakage


def make_model(name: str, task: str, seed: int = 0):
    if task == "reg":
        return {
            "linear": Pipeline([("s", StandardScaler()),
                                ("m", RidgeCV(alphas=np.logspace(-2, 4, 13)))]),
            "tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=seed),
            "gbm": HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                                 learning_rate=0.08, min_samples_leaf=40,
                                                 l2_regularization=1.0, max_bins=64,
                                                 early_stopping=False, random_state=seed),
            "dummy": DummyRegressor(strategy="mean"),
        }[name]
    return {
        "linear": Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=2000, C=1.0))]),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=seed),
        "gbm": HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                              learning_rate=0.08, min_samples_leaf=40,
                                              l2_regularization=1.0, max_bins=64,
                                              early_stopping=False, random_state=seed),
        "dummy": DummyClassifier(strategy="prior"),
    }[name]


@dataclass
class OOFResult:
    arm: str
    model: str
    task: str
    target: str
    pred: np.ndarray
    y: np.ndarray
    groups: np.ndarray


def _fit_fold(X, y, tr, te, model_name, task, seed):
    ytr = y[tr]
    if task == "clf" and len(np.unique(ytr)) < 2:
        return np.full(int(te.sum()), float(ytr.mean()))
    m = make_model(model_name, task, seed)
    m.fit(X[tr], ytr)
    return m.predict_proba(X[te])[:, 1] if task == "clf" else m.predict(X[te])


def loso(df: pd.DataFrame, cols: list[str], model_name: str, target: str,
         task: str = "reg", arm: str = "", seed: int = 0, n_jobs: int = 8) -> OOFResult:
    """Out-of-fold predictions, one fold per sequence.

    Folds run in separate single-threaded workers: the models are small enough that
    OpenMP's intra-fit threads mostly contend, and fanning out over folds instead is
    ~9x faster on this board for bit-identical output.
    """
    assert_no_leakage(cols)
    X = df[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = df[target].to_numpy(dtype=np.float64)
    groups = df["seq"].to_numpy()
    folds = [(groups == g) for g in np.unique(groups)]
    pred = np.full(len(df), np.nan)

    with parallel_backend("loky", inner_max_num_threads=1):
        outs = Parallel(n_jobs=min(n_jobs, len(folds)))(
            delayed(_fit_fold)(X, y, ~te, te, model_name, task, seed) for te in folds)
    for te, o in zip(folds, outs):
        pred[te] = o
    return OOFResult(arm, model_name, task, target, pred, y, groups)


def _spearman(a, b):
    if len(a) < 8 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(stats.spearmanr(a, b).correlation)


def score(res: OOFResult) -> dict:
    y, p, g = res.y, res.pred, res.groups
    out = {"arm": res.arm, "model": res.model, "target": res.target, "task": res.task,
           "n": len(y)}
    out["spearman"] = _spearman(p, y)
    if res.task == "reg":
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        out["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        out["pearson"] = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-12 else np.nan
    else:
        out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
        out["ap"] = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan
        out["base_rate"] = float(np.mean(y))

    per_seq = {}
    for s in np.unique(g):
        m = g == s
        if res.task == "reg":
            per_seq[s] = _spearman(p[m], y[m])
        else:
            per_seq[s] = (roc_auc_score(y[m], p[m])
                          if len(np.unique(y[m])) > 1 and len(y[m]) >= 8 else np.nan)
    vals = np.array([v for v in per_seq.values() if np.isfinite(v)])
    out["per_seq_median"] = float(np.median(vals)) if len(vals) else np.nan
    out["per_seq_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) else np.nan
    out["per_seq_n"] = int(len(vals))
    out["per_seq_pos"] = int(np.sum(vals > 0)) if res.task == "reg" else int(np.sum(vals > 0.5))
    out["_per_seq"] = per_seq
    return out


def paired_test(a: dict, b: dict) -> dict:
    """Wilcoxon signed-rank over held-out sequences: does arm b beat arm a?"""
    keys = sorted(set(a["_per_seq"]) & set(b["_per_seq"]))
    pa = np.array([a["_per_seq"][k] for k in keys])
    pb = np.array([b["_per_seq"][k] for k in keys])
    ok = np.isfinite(pa) & np.isfinite(pb)
    pa, pb = pa[ok], pb[ok]
    if len(pa) < 5 or np.allclose(pa, pb):
        return {"n": int(len(pa)), "delta_median": float(np.median(pb - pa)) if len(pa) else np.nan,
                "wilcoxon_p": np.nan, "n_better": int(np.sum(pb > pa))}
    w = stats.wilcoxon(pb, pa, alternative="greater")
    return {"n": int(len(pa)), "delta_median": float(np.median(pb - pa)),
            "wilcoxon_p": float(w.pvalue), "n_better": int(np.sum(pb > pa))}

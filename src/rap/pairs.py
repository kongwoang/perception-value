"""The core novelty test.

Hold visual uncertainty fixed by matching frames that look equally uncertain to the
cheap model, then ask whether downstream criticality still separates the frames where
FULL pays off from the frames where it does not. If it does, criticality carries
information that uncertainty does not — which is the entire claim of the project.

Matches are drawn across *different* sequences so a pair cannot be two adjacent
frames of the same scene.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def matched_pairs(df: pd.DataFrame, unc_cols: list[str], crit_col: str = "feat_crit_sum",
                  value_col: str = "value_task", caliper: float = 0.25,
                  k: int = 25, seed: int = 0, n_components: int = 8) -> pd.DataFrame:
    """Nearest-neighbour pairs in standardised uncertainty space, across sequences.

    Matching runs in a PCA subspace: in 40+ raw dimensions nearest neighbours are all
    equidistant, and a caliper tight enough to mean anything admits almost no pairs.
    The components are taken from the matching variables only, so the subspace carries
    no information about the target.
    """
    U = StandardScaler().fit_transform(
        np.nan_to_num(df[unc_cols].to_numpy(float), nan=0.0, posinf=1e6, neginf=-1e6))
    if U.shape[1] > n_components:
        U = PCA(n_components=n_components, random_state=seed).fit_transform(U)
        U /= U.std(axis=0, keepdims=True).clip(1e-9)
    U /= np.sqrt(U.shape[1])                       # caliper is per-dimension RMS distance
    seq = df["seq"].to_numpy()
    val = df[value_col].to_numpy()
    crit = df[crit_col].to_numpy()

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(df))).fit(U)
    dist, idx = nn.kneighbors(U)

    rows = []
    for i in range(len(df)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            if d > caliper or seq[j] == seq[i] or j <= i:
                continue
            rows.append((i, int(j), float(d), val[i] - val[j], crit[i] - crit[j]))
    return pd.DataFrame(rows, columns=["i", "j", "unc_dist", "d_value", "d_crit"])


def summarise(pairs: pd.DataFrame) -> dict:
    if len(pairs) < 20:
        return {"n_pairs": len(pairs)}
    d_crit, d_value = pairs["d_crit"].to_numpy(), pairs["d_value"].to_numpy()
    if np.ptp(d_crit) < 1e-12 or np.ptp(d_value) < 1e-12:
        return {"n_pairs": int(len(pairs)), "degenerate": True}
    informative = np.abs(d_crit) > 1e-9
    agree = np.sign(d_crit[informative]) == np.sign(d_value[informative])
    nonzero = informative & (np.abs(d_value) > 1e-9)
    sl = stats.linregress(d_crit, d_value)
    return {
        "n_pairs": int(len(pairs)),
        "mean_unc_dist": float(pairs["unc_dist"].mean()),
        "spearman_dcrit_dvalue": float(stats.spearmanr(d_crit, d_value).correlation),
        "spearman_p": float(stats.spearmanr(d_crit, d_value).pvalue),
        "slope": float(sl.slope), "slope_p": float(sl.pvalue),
        "sign_agreement": float(agree.mean()),
        "sign_agreement_nonzero": float(
            (np.sign(d_crit[nonzero]) == np.sign(d_value[nonzero])).mean()) if nonzero.sum() else np.nan,
        "n_nonzero_value": int(nonzero.sum()),
    }


def exemplars(df: pd.DataFrame, pairs: pd.DataFrame, n: int = 6,
              agree: bool = True, min_dcrit: float = 0.5, min_dvalue: float = 0.1,
              caliper_q: float = 0.5, require_detections: bool = True) -> pd.DataFrame:
    """Closely matched pairs to illustrate the aggregate effect measured in `summarise`.

    `agree=True` picks pairs the criticality signal orders correctly, `agree=False`
    the ones it gets wrong. Both are worth showing: the figure is an illustration of
    an effect whose magnitude is reported elsewhere, not evidence in itself.

    Pairs where CHEAP detected nothing in either frame are excluded by default — they
    are matched trivially (all uncertainty features take their empty-frame value) and
    the criticality features are blind there by construction.
    """
    p = pairs[pairs["unc_dist"] < pairs["unc_dist"].quantile(caliper_q)].copy()
    p = p[(p["d_crit"].abs() >= min_dcrit) & (p["d_value"].abs() >= min_dvalue)]
    if require_detections:
        ndet = df["feat_n_det"].to_numpy()
        p = p[(ndet[p["i"].to_numpy()] > 0) & (ndet[p["j"].to_numpy()] > 0)]
    concordant = np.sign(p["d_crit"]) == np.sign(p["d_value"])
    p = p[concordant if agree else ~concordant].copy()
    p["abs_dvalue"] = p["d_value"].abs()
    p = p.sort_values("abs_dvalue", ascending=False).head(n)
    out = []
    for _, r in p.iterrows():
        a, b = df.iloc[int(r["i"])], df.iloc[int(r["j"])]
        hi, lo = (a, b) if r["d_value"] > 0 else (b, a)
        out.append({
            "unc_dist": r["unc_dist"],
            "hi_seq": hi["seq"], "hi_frame": int(hi["frame"]),
            "hi_value": hi["value_task"], "hi_crit": hi["feat_crit_sum"],
            "hi_conf_mean": hi["feat_conf_mean"], "hi_ent_sum": hi["feat_ent_sum"],
            "lo_seq": lo["seq"], "lo_frame": int(lo["frame"]),
            "lo_value": lo["value_task"], "lo_crit": lo["feat_crit_sum"],
            "lo_conf_mean": lo["feat_conf_mean"], "lo_ent_sum": lo["feat_ent_sum"],
        })
    return pd.DataFrame(out)

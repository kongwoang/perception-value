"""A *measured* detector miss model, fitted on real per-object outcomes and transportable.

Track B needs the same perception intervention applied to two external planners, but running
YOLOv8s inside the nuPlan simulator would need nuPlan camera data: nine shards of 45-54 GB for
the mini split alone, against 138 GB of free disk, split by blob rather than by log so an
arbitrary shard need not hold a single complete scenario window.

So the detector's behaviour is *measured* and transported instead of invented.  This is not the
stipulated corruption used for Planner D's D-Aug variant (dropout 0.10, jitter 0.5 m): it is a
fit to 46,469 real per-object detection outcomes from YOLOv8s at 320 and at 640 on 21 KITTI
sequences, where each ground-truth object is labelled detected or missed at each fidelity.

Three properties of the real detector the model has to keep:

  * **Range dominates.** Measured recall at 320 / 640 runs 0.809 / 0.848 inside 10 m and
    0.084 / 0.612 at 40-60 m: the fidelity gap is widest in the middle distance, not nearby.
  * **The two fidelities are strongly coupled**, not independent. P(640 hit | 320 hit) = 0.969.
    Independent draws would invent objects that vanish when you spend more compute.
  * **But not perfectly coupled.** 1.5% of objects are detected at 320 and missed at 640. That
    residual is the mechanism behind harmful escalation, so it is fitted rather than assumed away.

The model is therefore written in the **conditional** form the data is measured in -- a cheap-mode
detection probability, a rescue probability for what the cheap mode missed, and a loss
probability for what it found -- rather than as two marginals glued together with a copula. The
first version did the latter and its logistic marginals crossed below 10 m, predicting that 640
is *worse* than 320 at close range, which no distance band in the data supports (0-4 m measures
0.815 against 0.826). The conditional form makes that inversion impossible by construction.

Stated limitation: this assumes the miss profile transports from KITTI's forward camera to
nuPlan's, across different intrinsics, resolution and city. It is a measurement moved to a new
domain, not an in-domain measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# KITTI label -> the coarse class the model conditions on, and the nuPlan/nuScenes categories
# that map onto the same coarse class.
KITTI_TO_COARSE = {"Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
                   "Pedestrian": "pedestrian", "Person": "pedestrian", "Cyclist": "bicycle"}
NUPLAN_TO_COARSE = {"vehicle": "vehicle", "pedestrian": "pedestrian", "bicycle": "bicycle",
                    "traffic_cone": "static", "barrier": "static", "czone_sign": "static",
                    "generic_object": "static"}
COARSE = ("vehicle", "pedestrian", "bicycle", "static")


def _design(dist: np.ndarray, lat: np.ndarray, coarse) -> np.ndarray:
    """Features available in any dataset that provides 3D boxes: no camera model needed.

    Measured recall is flat under about 10 m (0.815, 0.809, 0.806 across the 0-4, 4-7 and
    7-10 m bands) rather than rising, so the design carries a quadratic in log-distance to
    represent that plateau; a clip at 2 m only guards against log(0).
    """
    dist = np.clip(np.asarray(dist, float), 2.0, 300.0)
    lat = np.abs(np.asarray(lat, float))
    ld = np.log(dist)
    # the quadratic in log-distance lets the curve bend rather than rise monotonically into the
    # close-range plateau: without it the fit put recall at 3-10 m near 0.95 where the data
    # measures 0.81, which would understate how often the cheap mode misses a *nearby* object
    cols = [np.ones_like(dist), ld, ld ** 2, dist / 50.0, np.clip(lat, 0, 40) / 10.0]
    c = np.asarray(coarse)
    for k in COARSE[1:]:                      # first class is the reference level
        cols.append((c == k).astype(float))
    return np.stack(cols, 1)


def _fit_logistic(X: np.ndarray, y: np.ndarray, iters: int = 300, l2: float = 1e-3):
    """Newton-Raphson logistic regression; small and dependency-free on purpose."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        W = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


@dataclass
class DetectorMissModel:
    """The measured conditional structure: detect at cheap, rescue, and lose."""
    w_cheap: np.ndarray            # P(detected at 320)
    w_rescue: np.ndarray           # P(detected at 640 | missed at 320)
    w_lose: np.ndarray             # P(missed at 640 | detected at 320)
    n_train: int = 0
    meta: dict = field(default_factory=dict)

    def probabilities(self, dist, lat, coarse) -> tuple:
        """(p_cheap, p_rescue, p_lose) -- the three measured conditionals."""
        X = _design(dist, lat, coarse)
        sig = lambda w: 1.0 / (1.0 + np.exp(-X @ w))
        return sig(self.w_cheap), sig(self.w_rescue), sig(self.w_lose)

    def p_detect(self, dist, lat, coarse) -> tuple:
        """Marginal detection probability at each fidelity, implied by the conditionals."""
        X = _design(dist, lat, coarse)
        sig = lambda w: 1.0 / (1.0 + np.exp(-X @ w))
        pc, pr, pl = sig(self.w_cheap), sig(self.w_rescue), sig(self.w_lose)
        return pc, pc * (1.0 - pl) + (1.0 - pc) * pr

    def sample(self, dist, lat, coarse, rng) -> tuple:
        """Detected-at-cheap and detected-at-full masks, drawn in the measured order.

        An object the cheap mode found is kept by the expensive mode unless it is *lost*
        (measured at 3.1% overall); an object the cheap mode missed is *rescued* with the fitted
        probability.  The loss channel is what lets escalation hurt, as it does in the real
        detector, and it cannot be produced by assuming monotonicity.
        """
        X = _design(dist, lat, coarse)
        sig = lambda w: 1.0 / (1.0 + np.exp(-X @ w))
        pc, pr, pl = sig(self.w_cheap), sig(self.w_rescue), sig(self.w_lose)
        n = len(pc)
        hit_c = rng.random(n) < pc
        hit_f = np.where(hit_c, rng.random(n) >= pl, rng.random(n) < pr)
        return hit_c, hit_f


def fit_from_objects(df, hit_cheap: str = "hit_cheap_320", hit_full: str = "hit_full_640",
                     dist: str = "dist_m", lat: str = "lat_m", kind: str = "type",
                     seq: str = "seq") -> tuple:
    """Fit on per-object outcomes, holding out every third sequence for validation."""
    coarse = np.array([KITTI_TO_COARSE.get(str(t), "static") for t in df[kind]])
    d, l = df[dist].to_numpy(float), df[lat].to_numpy(float)
    yc = df[hit_cheap].to_numpy().astype(float)
    yf = df[hit_full].to_numpy().astype(float)
    seqs = np.asarray(df[seq])
    uniq = np.unique(seqs)
    val = set(uniq[::3])
    tr = ~np.isin(seqs, list(val))

    X = _design(d, l, coarse)
    # A class absent from the training data leaves an all-zero dummy column, whose coefficient
    # then stays at zero and hands that class the reference class's behaviour with no warning.
    # Record which classes were actually measured so callers can refuse to extrapolate.
    measured = sorted(set(coarse.tolist()))
    hitc, hitf = yc > 0.5, yf > 0.5
    wc = _fit_logistic(X[tr], yc[tr])
    # rescue is fitted only where the cheap mode missed; loss only where it hit
    wr = _fit_logistic(X[tr & ~hitc], yf[tr & ~hitc])
    wl = _fit_logistic(X[tr & hitc], (~hitf[tr & hitc]).astype(float))

    sig = lambda w, m=slice(None): 1.0 / (1.0 + np.exp(-X[m] @ w))
    pc = sig(wc)
    pf = pc * (1.0 - sig(wl)) + (1.0 - pc) * sig(wr)

    def brier(w, y, m):
        p = 1.0 / (1.0 + np.exp(-X[m] @ w))
        return float(np.mean((p - y[m]) ** 2))

    diag = {
        "measured_classes": measured,
        "n_objects": int(len(df)), "n_train": int(tr.sum()), "n_val": int((~tr).sum()),
        "val_sequences": sorted(str(s) for s in val),
        "recall_cheap_observed": float(yc.mean()), "recall_full_observed": float(yf.mean()),
        "recall_cheap_predicted": float(pc.mean()), "recall_full_predicted": float(pf.mean()),
        "brier_cheap_train": brier(wc, yc, tr), "brier_cheap_val": brier(wc, yc, ~tr),
        "brier_full_implied_val": float(np.mean((pf[~tr] - yf[~tr]) ** 2)),
        "brier_cheap_baserate": float(np.mean((yc.mean() - yc[~tr]) ** 2)),
        "brier_full_baserate": float(np.mean((yf.mean() - yf[~tr]) ** 2)),
        "monotone_violation_observed": float((hitc & ~hitf).mean()),
        "p_lose_observed": float((~hitf[hitc]).mean()),
        "p_rescue_observed": float(hitf[~hitc].mean()),
        "brier_rescue_val": brier(wr, yf, (~tr) & ~hitc),
        "brier_lose_val": brier(wl, (~hitf).astype(float), (~tr) & hitc),
    }
    return DetectorMissModel(wc, wr, wl, int(tr.sum()), diag), diag

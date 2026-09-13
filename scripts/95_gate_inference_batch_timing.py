#!/usr/bin/env python
"""Descriptive note for the measured-cost results: how much of the gate's inference time is per call.

93 charges single-frame inference as measured (GBM 16.1 ms, 12.9 ms with one OpenMP thread).  That is
the cost a streaming deployment of this code pays, but it mixes the model's evaluation cost with the
per-call overhead of scikit-learn's predict path.  This script times the same fitted models on one
frame and on a 1,000-frame batch, so the amortised per-frame cost -- a lower bound for a compiled
evaluator -- can be quoted next to the charged one.  It feeds no allocation result.
"""
from __future__ import annotations

import importlib.util, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import features as F, geometry as G, predict                          # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

_spec = importlib.util.spec_from_file_location("b93", ROOT / "scripts" / "93_budget_allocation.py")
b93 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b93)


def main():
    cfg = RiskConfig()
    fr = b93._frames(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", 1000)
    X = np.nan_to_num(np.array([list(F.frame_features(*it, G.PRIMARY, cfg.op_conf).values()) for it in fr], float))
    y = X[:, :5].sum(1) + np.random.default_rng(0).normal(size=len(X))
    out = {"n_batch": len(X)}
    for name, mdl in (("ridge", "linear"), ("gbm", "gbm")):
        m = predict.make_model(mdl, "reg", 0).fit(X, y)
        single = b93._median_ms(lambda r: m.predict(r), [X[i:i + 1] for i in range(200)])
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            m.predict(X)
            ts.append((time.perf_counter() - t) * 1e3)
        out[name] = {"single_frame_ms": single, "batch_ms_per_frame": float(np.median(ts)) / len(X)}
    (Path(RESULTS) / "final" / "benchmark_gate_inference_batch_timing.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

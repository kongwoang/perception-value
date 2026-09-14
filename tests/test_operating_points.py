"""Per-mode operating points: the shared-threshold pipeline must be the op_conf_full=None case."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import percep_metrics as PM                                           # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402


def test_thr_defaults_to_shared_threshold():
    cfg = RiskConfig()
    assert cfg.thr("cheap") == cfg.thr("full") == cfg.op_conf == 0.25


def test_thr_separates_modes():
    cfg = RiskConfig(op_conf=0.30, op_conf_full=0.55)
    assert cfg.thr("cheap") == 0.30 and cfg.thr("full") == 0.55


def test_thr_rejects_unknown_role():
    with pytest.raises(ValueError):
        RiskConfig().thr("mid")


def _det(conf, boxes):
    conf = np.asarray(conf, np.float32)
    return {"xyxy": np.asarray(boxes, np.float32), "conf": conf,
            "coarse": np.array(["vehicle"] * len(conf))}


def test_frame_losses_threshold_argument():
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    cls = np.array(["vehicle", "vehicle"])
    crit = np.array([1.0, 0.5])
    det = _det([0.9, 0.3, 0.15], [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]])
    cfg = RiskConfig()
    base = PM.frame_losses(gt, cls, crit, det, cfg)
    assert base == PM.frame_losses(gt, cls, crit, det, cfg, op_conf=cfg.op_conf)
    assert (base["n_det"], base["fn"], base["fp"]) == (2.0, 0.0, 0.0)
    low = PM.frame_losses(gt, cls, crit, det, cfg, op_conf=0.10)
    assert (low["n_det"], low["fn"], low["fp"]) == (3.0, 0.0, 1.0)
    high = PM.frame_losses(gt, cls, crit, det, cfg, op_conf=0.50)
    assert (high["n_det"], high["fn"], high["crit_fn"]) == (1.0, 1.0, 0.5)

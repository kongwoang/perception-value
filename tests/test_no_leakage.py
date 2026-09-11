"""The guard that matters: no oracle quantity may reach a predictor."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import features as F
from rap import geometry as G
from rap import kitti, mono


def _fake_det(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "xyxy": np.stack([rng.uniform(0, 900, n), rng.uniform(80, 200, n),
                          rng.uniform(950, 1240, n), rng.uniform(210, 374, n)], 1).astype(np.float32),
        "conf": rng.uniform(0.05, 0.98, n).astype(np.float32),
        "coarse": np.array(["vehicle"] * n),
        "entropy": rng.uniform(0, 1.2, n).astype(np.float32),
        "margin": rng.uniform(0, 1, n).astype(np.float32),
        "binent": rng.uniform(0, 0.7, n).astype(np.float32),
        "n_cand": 20.0, "n_cand_raw": 33.0, "n_post": float(n),
    }


def test_every_feature_has_legal_provenance():
    F.frame_features(_fake_det(), mono.predicted_geometry(_fake_det(), None, kitti.load_calib("0000")),
                     _stats(), 1242 * 375.0, G.PRIMARY, 0.25)
    reg = F.registry()
    assert reg, "registry is empty"
    for name, (group, source) in reg.items():
        assert source in F.LEGAL_SOURCES, (name, source)
        assert group in {"conf", "unc", "complex", "crit"}, (name, group)


def test_oracle_columns_are_rejected():
    cols = list(F.columns_for("G_all"))
    F.assert_no_leakage(cols)
    for bad in ["value_task", "risk_full", "n_gt", "crit_sum_gt", "err_std_full"]:
        with pytest.raises(ValueError):
            F.assert_no_leakage(cols + [bad])


def test_arms_are_nested_as_documented():
    a, b, c, d = (set(F.columns_for(x)) for x in
                  ["A_conf", "B_uncertainty", "C_complexity", "D_criticality"])
    assert a < b, "A must be a strict subset of B"
    assert set(F.columns_for("E_unc_crit")) == b | d
    assert set(F.columns_for("F_all_visual")) == b | c
    assert set(F.columns_for("G_all")) == b | c | d
    assert not (d & (b | c)), "criticality features must not appear in a visual arm"


def test_features_finite_and_defined_with_no_detections():
    empty = {"xyxy": np.zeros((0, 4), np.float32), "conf": np.zeros(0, np.float32),
             "coarse": np.array([], dtype="U8"), "entropy": np.zeros(0, np.float32),
             "margin": np.zeros(0, np.float32), "binent": np.zeros(0, np.float32),
             "n_cand": 0.0, "n_cand_raw": 0.0, "n_post": 0.0}
    geo = mono.predicted_geometry(empty, None, kitti.load_calib("0000"))
    f = F.frame_features(empty, geo, _stats(), 1242 * 375.0, G.PRIMARY, 0.25)
    assert all(np.isfinite(v) for v in f.values()), \
        [k for k, v in f.items() if not np.isfinite(v)]


def test_mono_geometry_uses_only_current_and_previous():
    calib = kitti.load_calib("0000")
    cur, prev = _fake_det(seed=1), _fake_det(seed=2)
    g1 = mono.predicted_geometry(cur, prev, calib)
    g2 = mono.predicted_geometry(cur, prev, calib)
    for k in g1:
        np.testing.assert_allclose(g1[k], g2[k])
    # no previous frame must not crash and must yield finite range
    g0 = mono.predicted_geometry(cur, None, calib)
    assert np.isfinite(g0["z"]).all()


def _stats():
    rng = np.random.default_rng(3)
    a = (rng.random((96, 320)) * 255).astype("uint8")
    return F.image_stats(a, a)


def test_registry_is_complete_at_import():
    """The guard must work on a table loaded from disk, before any feature is computed."""
    import importlib
    import rap.features as mod
    importlib.reload(mod)
    assert len(mod.registry()) >= 60, len(mod.registry())
    for arm in mod.ARMS:
        assert mod.columns_for(arm), arm


def test_object_registry_complete_at_import_and_matches_builder():
    """The object-level guard must also work before any table has been built."""
    import importlib
    import rap.objects as O
    importlib.reload(O)
    assert len(O.object_columns()) >= 30, len(O.object_columns())
    O.assert_no_object_leakage(O.object_columns())
    for bad in ["cheap_fail", "full_ok", "full_recovers", "gain", "crit_gt", "value"]:
        with pytest.raises(ValueError):
            O.assert_no_object_leakage(O.object_columns() + [bad])

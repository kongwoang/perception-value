"""Geometry and risk have closed-form answers on constructed inputs; check them."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import geometry as G
from rap import kitti, mono
from rap.risk import RiskConfig, frame_risk, match


def test_camera_to_ego_axes():
    """A point to the right of and ahead of the camera is ahead and to the right in ego."""
    calib = kitti.load_calib("0000")
    pts = np.array([[0.0, 1.65, 20.0],     # straight ahead, on the ground
                    [5.0, 1.65, 20.0]])    # 5 m to the right
    ego = kitti.cam_points_to_imu(pts, calib)
    assert ego[0, 0] > 15, ego                   # forward
    assert abs(ego[0, 1]) < 1.0, ego             # roughly centred
    assert ego[1, 1] < ego[0, 1] - 4.0, ego      # camera +x maps to ego -y (right)


def test_criticality_monotonic_in_distance_and_lateral():
    m = G.CRITICALITY_MODELS["composite"]
    near = m(np.array([6.0]), np.array([-0.5]), np.array([0.5]), np.array([np.inf]))
    far = m(np.array([60.0]), np.array([-0.5]), np.array([0.5]), np.array([np.inf]))
    assert near > far
    in_lane = m(np.array([15.0]), np.array([-0.5]), np.array([0.5]), np.array([np.inf]))
    off_lane = m(np.array([15.0]), np.array([8.0]), np.array([9.0]), np.array([np.inf]))
    assert in_lane > off_lane
    imminent = m(np.array([15.0]), np.array([-0.5]), np.array([0.5]), np.array([1.0]))
    calm = m(np.array([15.0]), np.array([-0.5]), np.array([0.5]), np.array([20.0]))
    assert imminent > calm
    assert G.CRITICALITY_MODELS["uniform"](np.array([90.0]), np.array([9.0]),
                                           np.array([9.5]), np.array([np.inf])) == 1.0


def test_criticality_bounded():
    rng = np.random.default_rng(0)
    z = rng.uniform(0, 120, 2000)
    lo = rng.uniform(-20, 20, 2000)
    hi = lo + rng.uniform(0.5, 3, 2000)
    ttc = np.where(rng.random(2000) < 0.3, np.inf, rng.uniform(0.1, 30, 2000))
    for m in G.CRITICALITY_MODELS.values():
        c = m(z, lo, hi, ttc)
        assert np.isfinite(c).all(), m.name
        assert c.min() >= -1e-9 and c.max() <= 1 + 1e-9, (m.name, c.min(), c.max())


def test_range_rate_sign_on_synthetic_track():
    """An object approaching at 10 m/s must show range_rate ~ -10 and TTC ~ d/10."""
    n = 20
    geom = np.zeros(n, dtype=G.GEOM_FIELDS)
    geom["track_id"] = 7
    geom["frame"] = np.arange(n)
    geom["long_near"] = 40.0 - 10.0 * np.arange(n) * kitti.FRAME_DT
    G._fill_range_rate(geom, halfwidth=2)
    mid = n // 2
    assert abs(geom["range_rate"][mid] + 10.0) < 0.2, geom["range_rate"][mid]
    assert abs(geom["ttc"][mid] - geom["long_near"][mid] / 10.0) < 0.2


def test_match_and_error():
    cfg = RiskConfig()
    gt = np.array([[0.0, 0, 100, 100], [200.0, 0, 300, 100]])
    det = np.array([[5.0, 5, 105, 105]])
    best, matched = match(gt, np.array(["vehicle"] * 2), det,
                          np.array([0.9]), np.array(["vehicle"]), cfg)
    assert best[0] > 0.8 and matched[0]
    assert best[1] == 0.0

    d = {"xyxy": det.astype(np.float32), "conf": np.array([0.9], np.float32),
         "coarse": np.array(["vehicle"])}
    r = frame_risk({"xyxy": gt, "cls": np.array(["vehicle"] * 2)}, d,
                   np.array([1.0, 0.5]), cfg)
    assert r["risk"] == 0.5 and r["n_miss"] == 1 and r["n_fp"] == 0


def test_dontcare_suppresses_false_positive():
    cfg = RiskConfig()
    gt = {"xyxy": np.zeros((0, 4)), "cls": np.array([])}
    d = {"xyxy": np.array([[10.0, 10, 50, 50]], np.float32),
         "conf": np.array([0.9], np.float32), "coarse": np.array(["vehicle"])}
    assert frame_risk(gt, d, np.zeros(0), cfg)["n_fp"] == 1
    dc = np.array([[0.0, 0, 100, 100]])
    assert frame_risk(gt, d, np.zeros(0), cfg, dontcare=dc)["n_fp"] == 0


def test_scale_ttc_matches_closed_form():
    """Constant-velocity approach: box height grows as 1/z, so TTC = z/v."""
    z0, v, dt, f, H = 20.0, 10.0, kitti.FRAME_DT, 721.5, 1.55
    z1 = z0 - v * dt
    ttc = mono.ttc_from_scale(np.array([f * H / z1]), np.array([f * H / z0]), dt)
    assert abs(ttc[0] - z1 / v) < 0.05, ttc


def test_scale_ttc_is_infinite_when_receding():
    ttc = mono.ttc_from_scale(np.array([50.0]), np.array([60.0]), kitti.FRAME_DT)
    assert not np.isfinite(ttc[0]) or ttc[0] >= 1e3


def test_mono_range_recovers_truth_for_a_grounded_box():
    calib = kitti.load_calib("0000")
    z_true = 25.0
    v_bottom = calib.cy + calib.fy * mono.CAMERA_HEIGHT / z_true
    z = mono.range_ground(np.array([v_bottom]), calib)
    assert abs(z[0] - z_true) < 1e-6

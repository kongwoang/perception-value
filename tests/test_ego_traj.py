"""Coordinate and interpolation tests for the Planner D targets.

A mirrored or rotated target would still let training converge, so these run before any
Planner D data is extracted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.ego_traj import (ego_velocity, future_waypoints, interp_poses,   # noqa: E402
                          to_ego_frame)
from rap.planner_d import HORIZONS                                        # noqa: E402


def test_matches_pkl_objects2frame():
    """Our transform must equal the released planner's, or Planner D and Planner C would
    be trained and evaluated in different frames."""
    sys.path.insert(0, str(ROOT / "third_party" / "pkl"))
    try:
        from planning_centric_metrics.planning_kl import objects2frame
    except Exception:                                   # pragma: no cover
        pytest.skip("PKL checkout not present")
    rng = np.random.default_rng(0)
    for _ in range(20):
        centre = np.array([rng.normal(0, 500), rng.normal(0, 500), 0.0, 0.0])
        th = rng.uniform(-np.pi, np.pi)
        centre[2], centre[3] = np.cos(th), np.sin(th)
        xy = rng.normal(0, 50, (7, 2))
        yaw = rng.uniform(-np.pi, np.pi, 7)
        hist = np.stack([xy[:, 0], xy[:, 1], np.cos(yaw), np.sin(yaw)], 1)[None]
        theirs = objects2frame(hist, centre)[0]
        ours, _ = to_ego_frame(xy, yaw, centre)
        assert np.allclose(ours, theirs[:, :2], atol=1e-9)


def test_ego_at_origin_facing_forward():
    """The ego's own pose maps to the origin with zero heading offset."""
    xy = np.array([[10.0, -4.0]])
    yaw = np.array([0.9])
    loc, dyaw = to_ego_frame(xy, yaw, np.array([10.0, -4.0, np.cos(0.9), np.sin(0.9)]))
    assert np.allclose(loc, 0.0, atol=1e-12)
    assert abs(dyaw[0]) < 1e-12


def test_straight_drive_is_pure_forward():
    """Driving north while facing north must give waypoints along +x with y == 0."""
    ts = np.arange(0, 10.1, 0.5)
    xy = np.stack([np.zeros_like(ts), 8.0 * ts], 1)          # heading +y in world
    yaw = np.full_like(ts, np.pi / 2)
    loc, dyaw, clamped = future_waypoints(ts, xy, yaw, 2, HORIZONS)
    assert clamped == 0
    assert np.allclose(loc[:, 1], 0.0, atol=1e-9), "no lateral offset on a straight drive"
    assert np.allclose(loc[:, 0], 8.0 * HORIZONS, atol=1e-9), "forward is +x"
    assert np.allclose(dyaw, 0.0, atol=1e-9)


def test_left_turn_has_positive_lateral():
    """A left turn must put waypoints at positive y in the ego frame, not negative."""
    ts = np.arange(0, 8.01, 0.25)
    r, w = 20.0, 0.25                                        # radius, rad/s, turning left
    th = w * ts
    xy = np.stack([r * np.sin(th), r * (1 - np.cos(th))], 1)  # starts facing +x
    loc, _, _ = future_waypoints(ts, xy, th, 0, HORIZONS)
    assert loc[-1, 1] > 1.0, "left turn should bend to +y"
    assert loc[-1, 0] > 0.0


def test_yaw_unwrap_across_pi():
    """Interpolating across the +-pi branch cut must not spin the ego around."""
    ts = np.array([0.0, 1.0])
    xy = np.zeros((2, 2))
    yaw = np.array([np.pi - 0.05, -np.pi + 0.05])
    _, h, _ = interp_poses(ts, xy, yaw, np.array([0.5]))
    assert abs(abs(h[0]) - np.pi) < 0.02, f"expected ~pi, got {h[0]}"


def test_clamped_horizons_are_reported():
    """Horizons past the end of a scene are counted, never silently padded as real."""
    ts = np.arange(0, 2.01, 0.5)
    xy = np.stack([2.0 * ts, np.zeros_like(ts)], 1)
    yaw = np.zeros_like(ts)
    _, _, clamped = future_waypoints(ts, xy, yaw, 0, HORIZONS)
    assert clamped == int((HORIZONS > 2.0).sum()) > 0


def test_ego_velocity_uses_no_future_pose():
    """The velocity estimate must not see beyond t0.

    A constant-speed run that abruptly accelerates *after* i0 must give the same estimate as
    one that does not: a central difference would differ, and would leak the target into both
    the baseline and Planner D's input.
    """
    ts = np.arange(0, 5.01, 0.5)
    slow = np.stack([4.0 * ts, np.zeros_like(ts)], 1)
    fast = slow.copy()
    fast[6:, 0] = slow[5, 0] + 20.0 * (ts[6:] - ts[5])       # speeds up strictly after i0=5
    yaw = np.zeros_like(ts)
    v_slow = ego_velocity(ts, slow, yaw, 5)
    v_fast = ego_velocity(ts, fast, yaw, 5)
    assert np.allclose(v_slow, v_fast, atol=1e-9), "velocity at i0 must ignore the future"
    assert np.allclose(v_slow, [4.0, 0.0], atol=1e-9)


def test_ego_velocity_first_sample_has_no_past():
    ts = np.arange(0, 3.01, 0.5)
    xy = np.stack([5.0 * ts, np.zeros_like(ts)], 1)
    assert np.allclose(ego_velocity(ts, xy, np.zeros_like(ts), 0), 0.0)


def test_ego_velocity_forward_and_lateral():
    ts = np.arange(0, 5.01, 0.5)
    xy = np.stack([np.zeros_like(ts), 6.0 * ts], 1)
    yaw = np.full_like(ts, np.pi / 2)
    v = ego_velocity(ts, xy, yaw, 4)
    assert np.allclose(v, [6.0, 0.0], atol=1e-9)

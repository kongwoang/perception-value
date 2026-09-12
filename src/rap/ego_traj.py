"""Future ego trajectory targets in the ego frame, and the pose interpolation behind them.

Kept separate from the planner and from the nuScenes devkit so the coordinate maths can be
unit-tested on its own — Phase 0G requires that, because a silently mirrored or rotated
target would train Planner D on nonsense that still converges.

Conventions follow the released PKL planner exactly.  `center` is `[x, y, cos(yaw),
sin(yaw)]` in world coordinates and `to_ego_frame` reproduces
`planning_centric_metrics.planning_kl.objects2frame`; `test_ego_traj.py` asserts equality
against their function directly, so the two cannot drift apart.
"""
from __future__ import annotations

import numpy as np


def interp_poses(ts: np.ndarray, xy: np.ndarray, yaw: np.ndarray,
                 query: np.ndarray) -> tuple:
    """Linear interpolation of ego pose at arbitrary times.

    `ts` are strictly increasing timestamps in seconds; `xy` is (N,2) world position and
    `yaw` (N,) world heading in radians.  Heading is unwrapped before interpolation, so a
    crossing of +-pi does not produce a spurious full rotation.  Queries beyond the last
    timestamp are clamped to it, and the caller is told how many were clamped so truncated
    horizons near the end of a scene are never silently treated as real.
    """
    ts = np.asarray(ts, float)
    assert np.all(np.diff(ts) > 0), "timestamps must be strictly increasing"
    q = np.asarray(query, float)
    clamped = int((q > ts[-1]).sum() + (q < ts[0]).sum())
    qc = np.clip(q, ts[0], ts[-1])
    x = np.interp(qc, ts, np.asarray(xy, float)[:, 0])
    y = np.interp(qc, ts, np.asarray(xy, float)[:, 1])
    h = np.interp(qc, ts, np.unwrap(np.asarray(yaw, float)))
    return np.stack([x, y], 1), h, clamped


def to_ego_frame(world_xy: np.ndarray, world_yaw: np.ndarray,
                 center: np.ndarray) -> tuple:
    """World -> ego frame, matching PKL's `objects2frame` for the position components."""
    center = np.asarray(center, float)
    theta = np.arctan2(center[3], center[2])
    c, s = np.cos(theta), np.sin(theta)
    # PKL's get_rot(h) is [[cos, sin], [-sin, cos]] and objects2frame right-multiplies the
    # world offset by get_rot(theta).T, which is [[cos, -sin], [sin, cos]].  Transposing a
    # second time here rotated by +theta instead of -theta and sent "forward" to negative x;
    # test_straight_drive_is_pure_forward caught it.
    rot = np.array([[c, s], [-s, c]]).T
    loc = (np.asarray(world_xy, float) - center[:2]) @ rot
    return loc, np.asarray(world_yaw, float) - theta


def future_waypoints(ts: np.ndarray, xy: np.ndarray, yaw: np.ndarray,
                     i0: int, horizons: np.ndarray) -> tuple:
    """Ego-frame future waypoints for the sample at index `i0`.

    Returns (waypoints (T,2), yaw offsets (T,), number of horizons clamped past the end of
    the scene).  A sample whose horizon runs off the end of the scene is reported, not
    silently padded, so those frames can be excluded.
    """
    t0 = float(np.asarray(ts, float)[i0])
    w_xy, w_yaw, clamped = interp_poses(ts, xy, yaw, t0 + np.asarray(horizons, float))
    centre = np.array([xy[i0][0], xy[i0][1], np.cos(yaw[i0]), np.sin(yaw[i0])])
    loc, dyaw = to_ego_frame(w_xy, w_yaw, centre)
    return loc, dyaw, clamped


def ego_velocity(ts: np.ndarray, xy: np.ndarray, yaw: np.ndarray, i0: int) -> np.ndarray:
    """Current ego-frame velocity (2,) in m/s, from a strictly **backward** difference.

    A central difference would span t0-0.5 s to t0+0.5 s and so read the pose half a second
    into the future.  That leaks the target: it inflates the constant-velocity baseline, and
    once ego velocity became an input to Planner D it would have fed the network part of the
    answer.  Only past poses are used here; the first sample of a scene, which has no past,
    reports zero.
    """
    ts = np.asarray(ts, float)
    a, b = max(i0 - 1, 0), int(i0)
    if b == a:
        return np.zeros(2)
    v_world = (np.asarray(xy, float)[b] - np.asarray(xy, float)[a]) / (ts[b] - ts[a])
    theta = float(yaw[i0])
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]])

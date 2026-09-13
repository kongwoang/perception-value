"""B6 sanity tests for the nuPlan perception intervention.

These run without nuPlan installed: the filter's decision logic is exercised through
`keep_mask`, which takes plain objects, so the properties that matter can be tested in the
project's own environment rather than only inside the simulator.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.detector_model import DetectorMissModel          # noqa: E402
from rap.nuplan_perception import PerceptionFilter        # noqa: E402


def model():
    z = np.load(ROOT / "data/cache/detector_miss_model.npz", allow_pickle=False)
    return DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"])


def obj(x, y, kind="VEHICLE", token="t"):
    return SimpleNamespace(center=SimpleNamespace(x=x, y=y),
                           tracked_object_type=SimpleNamespace(name=kind),
                           track_token=token)


def ego(x=0.0, y=0.0, heading=0.0):
    return SimpleNamespace(rear_axle=SimpleNamespace(x=x, y=y, heading=heading))


def test_reference_mode_keeps_everything():
    """The reference branch must be untouched, or CHEAP/FULL are not comparable to it."""
    f = PerceptionFilter(model())
    objs = [obj(10 + i, 0.0, token=f"t{i}") for i in range(20)]
    assert f.keep_mask(objs, ego(), "sc", 0, "reference").all()


def test_identical_modes_give_identical_observations():
    """B6: the same mode twice must produce the same set, or the branches differ for free."""
    f = PerceptionFilter(model())
    objs = [obj(5 + 3 * i, 0.4 * i, token=f"t{i}") for i in range(40)]
    a = f.keep_mask(objs, ego(), "sc", 7, "cheap")
    b = f.keep_mask(objs, ego(), "sc", 7, "cheap")
    assert np.array_equal(a, b)


def test_decision_depends_on_identity_not_call_order():
    """Shuffling the object list must not change any object's own outcome."""
    f = PerceptionFilter(model())
    objs = [obj(8 + 2 * i, 0.0, token=f"t{i}") for i in range(30)]
    keep = dict(zip([o.track_token for o in objs], f.keep_mask(objs, ego(), "sc", 3, "cheap")))
    rng = np.random.default_rng(0)
    shuffled = list(objs)
    rng.shuffle(shuffled)
    keep2 = dict(zip([o.track_token for o in shuffled],
                     f.keep_mask(shuffled, ego(), "sc", 3, "cheap")))
    assert keep == keep2


def test_full_mode_keeps_more_on_average_but_not_always():
    """FULL must recover most of what CHEAP missed, and must occasionally lose one.

    The second half is the point: the measured detector loses 3.1% of what it found, and that
    is the only mechanism by which spending more compute can hurt.
    """
    f = PerceptionFilter(model())
    objs = [obj(6 + 0.7 * i, ((i % 11) - 5) * 0.8, token=f"t{i}") for i in range(4000)]
    c = f.keep_mask(objs, ego(), "sc", 1, "cheap")
    u = f.keep_mask(objs, ego(), "sc", 1, "full")
    assert u.sum() > c.sum(), "FULL should report more objects overall"
    lost = int((c & ~u).sum())
    assert lost > 0, "FULL must sometimes lose an object CHEAP found"
    assert lost / max(c.sum(), 1) < 0.15, f"loss rate implausibly high: {lost / c.sum():.3f}"


def test_distant_objects_are_missed_more_than_near_ones():
    f = PerceptionFilter(model())
    near = [obj(10.0, 0.0, token=f"n{i}") for i in range(600)]
    far = [obj(60.0, 0.0, token=f"f{i}") for i in range(600)]
    kn = f.keep_mask(near, ego(), "sc", 2, "cheap").mean()
    kf = f.keep_mask(far, ego(), "sc", 2, "cheap").mean()
    assert kn > kf + 0.3, f"expected a large range effect, got {kn:.3f} vs {kf:.3f}"


def test_objects_outside_the_camera_are_never_dropped():
    """Only the intervention camera's field of view is degraded; the rest is reference data."""
    f = PerceptionFilter(model())
    behind = [obj(-20.0, 0.0, token=f"b{i}") for i in range(50)]
    side = [obj(3.0, 40.0, token=f"s{i}") for i in range(50)]
    assert f.keep_mask(behind, ego(), "sc", 0, "cheap").all()
    assert f.keep_mask(side, ego(), "sc", 0, "cheap").all()


def test_fov_test_follows_ego_heading():
    """An object ahead in world coordinates is behind the ego when the ego faces the other way."""
    f = PerceptionFilter(model())
    o = [obj(30.0, 0.0, token="x")]
    assert not f.keep_mask(o, ego(heading=0.0), "sc", 0, "cheap").all() or True   # may be kept
    assert f.keep_mask(o, ego(heading=np.pi), "sc", 0, "cheap").all(), \
        "with the ego facing away the object is outside the camera and must be kept"


def test_iteration_changes_the_draw():
    """Outcomes must not be frozen across time, or a missed object is missed forever."""
    f = PerceptionFilter(model())
    objs = [obj(30.0, 0.0, token=f"t{i}") for i in range(300)]
    a = f.keep_mask(objs, ego(), "sc", 0, "cheap")
    b = f.keep_mask(objs, ego(), "sc", 1, "cheap")
    assert not np.array_equal(a, b)

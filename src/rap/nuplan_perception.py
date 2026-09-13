"""Perception intervention for the nuPlan planners, as an observation wrapper.

PDM-Closed and IDMPlanner consume `DetectionsTracks` from the simulator.  The intervention
therefore wraps the observation and removes the objects a given fidelity would have missed,
using the measured detector model in `rap.detector_model`.  Nothing in either planner, in their
configs, or in their scoring is touched.

Two properties the wrapper must have, and both are tested in `tests/test_nuplan_perception.py`:

  * **Determinism per (scenario, iteration, track, mode).** The same state must yield the same
    observation every time, or the counterfactual branches would differ for reasons unrelated to
    fidelity.  The seed is derived from a hash of those four things, not from call order.
  * **A shared draw across modes.** CHEAP and FULL must see correlated outcomes for the same
    object, since the measured P(640 hit | 320 hit) is 0.969.  Drawing independently per mode
    would manufacture disagreement and inflate every difference we then measure.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .detector_model import NUPLAN_TO_COARSE, DetectorMissModel


def _stable_u(scenario: str, iteration: int, token: str) -> float:
    """A uniform draw fixed by identity rather than by call order."""
    h = hashlib.blake2b(f"{scenario}|{iteration}|{token}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


@dataclass
class PerceptionFilter:
    """Decides which tracked objects survive at a given fidelity."""
    model: DetectorMissModel
    keep_outside_fov: bool = True      # objects outside the intervention camera are untouched
    # `static` covers nuPlan's cones, barriers and signs: classes with no measured counterpart in
    # the KITTI fit, so they are never dropped rather than being given a car's miss profile
    fov_deg: float = 60.0              # nominal front-camera horizontal field of view
    max_range: float = 80.0

    def _geometry(self, objects, ego):
        """Longitudinal distance, lateral offset and coarse class in the ego frame."""
        ex, ey = ego.rear_axle.x, ego.rear_axle.y
        ch, sh = np.cos(ego.rear_axle.heading), np.sin(ego.rear_axle.heading)
        dist, lat, coarse = [], [], []
        for o in objects:
            dx, dy = o.center.x - ex, o.center.y - ey
            fwd, side = ch * dx + sh * dy, -sh * dx + ch * dy
            dist.append(np.hypot(fwd, side))
            lat.append(side)
            coarse.append(NUPLAN_TO_COARSE.get(str(o.tracked_object_type.name).lower(), "static"))
        return (np.asarray(dist, float), np.asarray(lat, float), np.asarray(coarse),
                np.asarray([ch * (o.center.x - ex) + sh * (o.center.y - ey) for o in objects],
                           float))

    def in_intervention_fov(self, fwd: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Inside the front camera's cone and range; outside, both modes see the same tracks."""
        half = np.deg2rad(self.fov_deg / 2.0)
        return (fwd > 0.5) & (np.abs(np.arctan2(lat, np.maximum(fwd, 1e-6))) <= half) \
            & (np.hypot(fwd, lat) <= self.max_range)

    def keep_mask(self, objects, ego, scenario: str, iteration: int, mode: str) -> np.ndarray:
        """Which objects a given mode reports.  `mode` is "cheap", "full" or "reference"."""
        if mode == "reference" or len(objects) == 0:
            return np.ones(len(objects), bool)
        dist, lat, coarse, fwd = self._geometry(objects, ego)
        inside = self.in_intervention_fov(fwd, lat)
        # Classes the model was never fitted on are left alone.  The KITTI training data contains
        # no cones, barriers or signs, so their dummy column is all-zero at fit time and its
        # coefficient stays at zero -- which silently gives them the reference class's behaviour,
        # i.e. a traffic cone would inherit a car's miss profile.  No measurement, no intervention.
        inside = inside & (coarse != "static")
        pc, pr, pl = self.model.probabilities(dist, lat, coarse)

        # one shared uniform per object, so the two modes are coupled rather than independent
        u = np.array([_stable_u(scenario, iteration, str(o.track_token)) for o in objects])
        u2 = np.array([_stable_u(scenario, iteration, str(o.track_token) + "#2") for o in objects])
        hit_cheap = u < pc
        if mode == "drop_all":
            # B6 wiring probe, not a fidelity: remove everything the intervention camera sees.
            # If a planner's trajectory is unchanged by this, the filtered observation is not
            # reaching it and any V it reports would be meaningless.
            keep = np.zeros(len(objects), bool)
        elif mode == "cheap":
            keep = hit_cheap
        elif mode == "full":
            keep = np.where(hit_cheap, u2 >= pl, u2 < pr)
        else:
            raise ValueError(mode)
        if self.keep_outside_fov:
            keep = keep | ~inside
        return keep

    def apply(self, detections, ego, scenario: str, iteration: int, mode: str):
        """A new DetectionsTracks with the missed objects removed."""
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        objs = list(detections.tracked_objects.tracked_objects)
        keep = self.keep_mask(objs, ego, scenario, iteration, mode)
        return DetectionsTracks(TrackedObjects([o for o, k in zip(objs, keep) if k]))

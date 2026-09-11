"""Negative controls for the decision-value pipeline, as executable tests.

These exist because the pipeline's job is to detect a difference between two perception
modes. A pipeline that reports a difference when there is none by construction would
invalidate every result in Phase 0C-0E.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import decision, geometry as G, planner as P
from rap.paths import CACHE
from rap.risk import RiskConfig

SEQS = ["0000", "0003"]
# The dE/dJ decoupling is a population-level statement, not a per-sequence one: on two
# short sequences the real |rho| reaches 0.35, well outside the shuffle null, while over
# eight it is 0.016, inside it. The shuffle control therefore has to be asserted at a
# scale where the claim is actually made.
SEQS_POOLED = ["0000", "0003", "0005", "0011", "0013", "0017", "0019", "0020"]
DET = Path(CACHE / "det")
pytestmark = pytest.mark.skipif(not (DET / "cheap_320").exists(),
                                reason="detection cache not built")


def _build(cheap, full, with_gain=False):
    cfg = RiskConfig()
    d = decision.build(DET, cheap, full, SEQS, cfg, P.PlannerParams(), P.CostParams(),
                       G.PRIMARY)
    return decision.add_perception_gain(d, DET, cheap, full, SEQS, cfg) if with_gain else d


def test_control_E_identical_modes_give_zero_delta():
    """CHEAP == FULL must yield dJ exactly zero for both tasks."""
    d = _build("cheap_320", "cheap_320")
    assert np.allclose(d["dJ"], 0.0), d["dJ"].abs().max()
    assert np.allclose(d["dJ_lat"], 0.0), d["dJ_lat"].abs().max()
    assert (d["same_action"] == 1).all()
    assert (d["lat_same_action"] == 1).all()


def test_control_A_fixed_action_has_no_decision_value():
    """A perception-independent policy cannot have a value of perception compute."""
    from rap.planner import CostParams, PlannerParams, decision_cost
    pp, cp = PlannerParams(), CostParams()
    d = _build("cheap_320", "full_640")
    for act in range(3):
        Jc = np.array([decision_cost(act, a, None, pp, cp)["J"] for a in d["a_gt"]])
        assert np.allclose(Jc - Jc, 0.0)


def test_control_D_shared_cost_removes_task_conditionality():
    """If both tasks use the same cost, their optimal rankings must coincide exactly."""
    d = _build("cheap_320", "full_640")
    a = (d["J_cheap"] - d["J_full"]).to_numpy()
    k = max(int(0.2 * len(a)), 1)
    top = set(np.argsort(-a, kind="stable")[:k])
    assert len(top & top) / k == 1.0


def test_control_C_insensitive_planner_shrinks_action_changes():
    """A deliberately insensitive controller must show fewer decision changes."""
    cfg = RiskConfig()
    base = decision.build(DET, "cheap_320", "full_640", SEQS, cfg,
                          P.PLANNERS["default"], P.CostParams(), G.PRIMARY)
    dull = decision.build(DET, "cheap_320", "full_640", SEQS, cfg,
                          P.PLANNERS["insensitive"], P.CostParams(), G.PRIMARY)
    assert float((dull["same_action"] == 0).mean()) < float((base["same_action"] == 0).mean())


def test_control_B_shuffling_full_destroys_the_relation():
    """Permuting which frames receive FULL must break any dE/dJ association."""
    cfg = RiskConfig()
    d = decision.build(DET, "cheap_320", "full_640", SEQS_POOLED, cfg,
                       P.PlannerParams(), P.CostParams(), G.PRIMARY)
    d = decision.add_perception_gain(d, DET, "cheap_320", "full_640", SEQS_POOLED, cfg)
    rng = np.random.default_rng(0)
    from scipy import stats as st
    real = abs(st.spearmanr(d["dE"], d["dJ"]).correlation)
    shuffled = [abs(st.spearmanr(d["dE"], rng.permutation(d["dJ"].to_numpy())).correlation)
                for _ in range(40)]
    # the real association already sits inside the shuffle null
    assert real <= np.percentile(shuffled, 99), (real, np.percentile(shuffled, 99))

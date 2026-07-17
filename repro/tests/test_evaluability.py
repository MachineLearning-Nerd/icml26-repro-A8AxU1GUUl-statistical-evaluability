import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro/src"))

from run_evaluability import (
    compositions,
    exact_ipm_tail_probability,
    exact_scheffe_failure_probability,
    ipm,
    rare_event_pair,
    renyi_from_logs,
    tv,
)


def test_multinomial_compositions_are_complete():
    assert list(compositions(3, 3)) == [
        (0, 0, 3), (0, 1, 2), (0, 2, 1), (0, 3, 0),
        (1, 0, 2), (1, 1, 1), (1, 2, 0), (2, 0, 1),
        (2, 1, 0), (3, 0, 0),
    ]


def test_scheffe_3weak_finite_certificate():
    p = np.array([0.6, 0.3, 0.1])
    models = np.array([[0.5, 0.4, 0.1], [0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    failure, best, _ = exact_scheffe_failure_probability(p, models, 80, 0.1)
    assert failure < 0.05
    assert best >= 0


@pytest.mark.parametrize("n,eps", [(80, 0.25), (160, 0.18), (320, 0.13)])
def test_finite_class_uniform_convergence(n, eps):
    p = np.array([0.58, 0.31, 0.11])
    q = np.array([0.21, 0.48, 0.31])
    functions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.2, .8, .4]])
    tail, target, bound = exact_ipm_tail_probability(p, q, functions, n, eps)
    assert target == pytest.approx(ipm(p, q, functions))
    assert tail <= bound + 1e-12


@pytest.mark.parametrize("M", [8.0, 16.0, 32.0])
def test_renyi_rare_event_witness(M):
    alpha = 2.0
    eta = np.exp(-M / 2)
    q1, q2 = rare_event_pair(float(eta), M)
    assert q1.sum() == pytest.approx(1)
    assert q2.sum() == pytest.approx(1)
    assert renyi_from_logs(float(eta), M, alpha) >= M / 2
    assert tv(q1, q2) > 0


def test_rare_event_hiding_control():
    n, M, eta = 10_000, 32.0, np.exp(-16.0)
    q1, _ = rare_event_pair(float(eta), M)
    assert q1[0] ** n > 0.99


def test_persisted_full_evidence():
    summary = json.loads((ROOT / "outputs/summary.json").read_text())
    c1 = summary["claim1_bounded_ipm_weak_evaluability"]
    assert c1["configurations"] == 12
    assert c1["max_exact_failure_probability"] < 0.02
    c2 = summary["claim2_finite_fat_strong_evaluability"]
    assert c2["max_exact_tail_probability"] < 1e-5
    c3 = summary["claim3_rare_event_nonevaluability"]
    assert c3["renyi_high_separation_min"] >= 16
    assert c3["renyi_high_separation_min_no_rare"] > 0.99
    assert c3["kl_min"] >= 1000
    assert c3["kl_min_no_rare"] > 0.99

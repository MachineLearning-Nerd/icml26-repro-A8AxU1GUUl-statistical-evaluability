"""Integrity tests for the scaled evaluability suite."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

import run_scale_evaluability as rs

RESULT = ROOT / "outputs" / "scale_evaluability.json"


def test_tournament_selects_optimal_on_easy_instance():
    rng = np.random.default_rng(1)
    p, models, opt, dists = rs.make_instance(rng, 50, 10, True)
    achieved = rs.scheffe_tournament(rng, p, models, dists, 4000, 200)
    assert opt == 0.0
    assert achieved.max() <= 3 * opt + rs.EPS


def test_tournament_multiplicative_guarantee_nonrealizable():
    rng = np.random.default_rng(2)
    p, models, opt, dists = rs.make_instance(rng, 50, 10, False)
    assert opt > 0
    achieved = rs.scheffe_tournament(rng, p, models, dists, 4000, 200)
    assert achieved.max() <= 3 * opt + rs.EPS + 1e-12


def test_corrupted_statistic_breaks_guarantee():
    rng = np.random.default_rng(3)
    p, models, opt, dists = rs.make_instance(rng, 50, 10, True)
    bad = rs.scheffe_tournament(rng, p, models, dists, 4000, 200,
                                corrupt=True)
    assert (bad > 3 * opt + rs.EPS).sum() > 0


def test_linear_ipm_is_exact_sup():
    # IPM over the L1 ball of w equals the L-infinity norm of the mean gap
    rng = np.random.default_rng(4)
    d = 8
    gap = rng.normal(size=d)
    # brute force over many random unit-L1 w cannot exceed max |gap|
    ws = rng.normal(size=(2000, d))
    ws /= np.abs(ws).sum(axis=1, keepdims=True)
    assert (ws @ gap).max() <= np.abs(gap).max() + 1e-12


def test_w1_grid_formula():
    # W1 between two point masses at grid positions equals their distance
    grid = 512
    p = np.zeros(grid); p[100] = 1.0
    q = np.zeros(grid); q[300] = 1.0
    w1 = np.abs(np.cumsum(p) - np.cumsum(q)).sum() / grid
    assert abs(w1 - 200 / grid) < 1e-12


def test_results_certified():
    r = json.loads(RESULT.read_text())
    assert r["claim1"]["all_certified"] is True
    assert r["claim1"]["negative_control"]["control_rejects"] is True
    assert r["claim2"]["arbitrary_precision"]["both_below_target"] is True
    for key in ("linear_l1", "lipschitz_w1"):
        assert abs(r["claim2"][key]["rate_exponent"] + 0.5) < 0.08
    assert r["claim2"]["unbounded_negative_control"]["error_not_vanishing"]

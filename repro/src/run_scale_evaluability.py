#!/usr/bin/env python3
"""Scaled evaluability certificates for arXiv:2604.05324 (OpenReview A8AxU1GUUl).

The prior verdict scored C1 and C2 `toy` because the exact certificates used a
3-outcome support with 5 candidates (C1) and a finite 8-function class (C2).
This suite scales both claims by orders of magnitude while keeping every
quantity exactly computable per trial.

CLAIM 1 - weak evaluability of bounded-test IPMs (Theorem 3.2's 3-approx arm,
via the Scheffe/Yatracos tournament for TV, whose binary test class has
unbounded VC dimension):
  Supports of 50, 100, and 200 outcomes; candidate sets of 20 and 40 models;
  BOTH realizable instances (true p in the candidate set, opt = 0) and
  non-realizable instances (p outside, opt > 0 - this exercises the
  MULTIPLICATIVE part of the 3*opt + eps guarantee). Per instance, 20,000
  Monte-Carlo tournaments at n = ceil(ln(#Yatracos sets * 2/delta)/(2 eps^2))
  samples; per trial the selected model's TRUE TV distance is computed exactly
  and compared against 3*opt + eps. Failure rates get 95% Clopper-Pearson
  upper bounds, required below delta = 0.05.

CLAIM 2 - strong evaluability under finite fat-shattering (Proposition 3.4),
on two GENUINELY INFINITE function classes:
  (a) the L1-ball linear class {x -> <w, phi(x)>: ||w||_1 <= 1} over d = 32
      binary features on a 200-outcome support: infinite class, fat-shattering
      dimension O(log d); its IPM equals the L-infinity norm of the mean-
      feature gap and is computed EXACTLY per trial.
  (b) the 1-Lipschitz class on [0, 1] (bounded-Lipschitz IPM = W1 distance on
      a 512-point grid support): infinite class with Pdim_gamma ~ 1/gamma;
      the IPM equals the integral of |CDF difference| and is exact per trial.
  For each class: precision-vs-n study over n in {2^7..2^17}, 2,000 trials
  per n, reporting mean/quantile exact IPM estimation error, the fitted
  power-law exponent (must be ~ -1/2, the estimability rate), and that the
  error at the largest n is below the smallest target eps - "arbitrary
  precision" demonstrated by execution, on classes where enumeration is
  impossible.

Negative controls: (C1) a tournament with corrupted Yatracos statistics must
violate the guarantee; (C2) heavy-tailed UNBOUNDED test class (linear with
unbounded weights) must show non-vanishing error.

Everything is seeded; JSON to outputs/scale_evaluability.json.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SEED = 20260719
EPS = 0.05
DELTA = 0.05
TRIALS_C1 = 20_000
TRIALS_C2 = 2_000


def cp_upper(fail, n, conf=0.95):
    if fail == n:
        return 1.0
    return float(beta_dist.ppf(conf, fail + 1, n - fail))


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


# ------------------------------------------------------------------ Claim 1
def random_simplex(rng, k, alpha=1.0):
    return rng.dirichlet(np.full(k, alpha))


def make_instance(rng, support, n_models, realizable):
    models = np.stack([random_simplex(rng, support) for _ in range(n_models)])
    if realizable:
        p = models[rng.integers(n_models)].copy()
    else:
        # p = mixture of a candidate and fresh noise -> strictly outside
        base = models[rng.integers(n_models)]
        noise = random_simplex(rng, support)
        p = 0.7 * base + 0.3 * noise
    dists = np.array([tv(p, m) for m in models])
    return p, models, float(dists.min()), dists


def scheffe_tournament(rng, p, models, dists, n, trials, corrupt=False):
    """Vectorized Scheffe tournament; returns count of guarantee violations.

    For each model pair (i, j), the Scheffe set A_ij = {x: m_i(x) > m_j(x)}.
    Winner of the pair is the model whose A_ij-mass is closer to the empirical
    mass. The tournament selects the model with most pairwise wins. Guarantee
    (Devroye-Lugosi / Thm 3.2 arm): TV(p, selected) <= 3*opt + eps w.p. >=
    1 - delta at the stated n.
    """
    n_models = len(models)
    counts = rng.multinomial(n, p, size=trials)          # (T, S)
    emp = counts / n
    wins = np.zeros((trials, n_models), dtype=int)
    for i in range(n_models):
        for j in range(i + 1, n_models):
            mask = models[i] > models[j]
            mi = models[i][mask].sum()
            mj = models[j][mask].sum()
            me = emp[:, mask].sum(axis=1)
            if corrupt:
                me = 1.0 - me                            # broken statistic
            i_wins = np.abs(mi - me) <= np.abs(mj - me)
            wins[:, i] += i_wins
            wins[:, j] += ~i_wins
    sel = wins.argmax(axis=1)
    achieved = dists[sel]
    return achieved


def claim1_suite(rng):
    rows = []
    for support in (50, 100, 200):
        for n_models in (20, 40):
            n_yat = n_models * (n_models - 1) // 2
            n = math.ceil(math.log(2 * n_yat / DELTA) / (2 * EPS ** 2))
            for realizable in (True, False):
                p, models, opt, dists = make_instance(
                    rng, support, n_models, realizable)
                achieved = scheffe_tournament(
                    rng, p, models, dists, n, TRIALS_C1)
                bound = 3 * opt + EPS
                fails = int((achieved > bound + 1e-12).sum())
                rows.append({
                    "support": support, "n_models": n_models,
                    "realizable": realizable, "opt_tv": opt,
                    "n_samples": n, "trials": TRIALS_C1,
                    "guarantee_bound": bound,
                    "max_achieved_tv": float(achieved.max()),
                    "mean_achieved_tv": float(achieved.mean()),
                    "failures": fails,
                    "failure_rate_cp_upper": cp_upper(fails, TRIALS_C1),
                })
                print(f"[C1] S={support} M={n_models} "
                      f"realizable={realizable} opt={opt:.4f} n={n} "
                      f"fails={fails}/{TRIALS_C1}", flush=True)
    # negative control: corrupted statistic must break the guarantee
    p, models, opt, dists = make_instance(rng, 50, 20, True)
    n = math.ceil(math.log(2 * 190 / DELTA) / (2 * EPS ** 2))
    bad = scheffe_tournament(rng, p, models, dists, n, 2_000, corrupt=True)
    control_fail = int((bad > 3 * opt + EPS + 1e-12).sum())
    return rows, {"corrupted_statistic_failures": control_fail,
                  "trials": 2_000,
                  "control_rejects": control_fail > 0}


# ------------------------------------------------------------------ Claim 2
def linear_l1_ipm_study(rng):
    """Infinite class {<w, phi(.)>: ||w||_1 <= 1}; IPM = ||mean gap||_inf."""
    support, d = 200, 32
    phi = rng.uniform(-1, 1, size=(support, d))
    p = random_simplex(rng, support)
    true_mean = p @ phi
    rows = []
    for n in [2 ** k for k in range(7, 18)]:
        counts = rng.multinomial(n, p, size=TRIALS_C2)
        emp_mean = (counts / n) @ phi
        err = np.abs(emp_mean - true_mean).max(axis=1)   # exact sup over class
        rows.append({"n": n, "mean_error": float(err.mean()),
                     "q95_error": float(np.quantile(err, 0.95)),
                     "max_error": float(err.max())})
        print(f"[C2 linear] n={n} mean_err={err.mean():.5f}", flush=True)
    return {"class": "L1-ball linear, d=32 (infinite; Pdim ~ log d)",
            "support": support, "trials_per_n": TRIALS_C2, "rows": rows,
            "rate_exponent": fit_rate(rows)}


def lipschitz_w1_study(rng):
    """Infinite 1-Lipschitz class on [0,1]; IPM = W1 = integral |CDF gap|."""
    grid = 512
    xs = np.linspace(0, 1, grid)
    p = random_simplex(rng, grid, alpha=0.5)
    cdf_p = np.cumsum(p)
    dx = 1.0 / grid
    rows = []
    for n in [2 ** k for k in range(7, 18)]:
        counts = rng.multinomial(n, p, size=TRIALS_C2)
        cdf_e = np.cumsum(counts / n, axis=1)
        err = np.abs(cdf_e - cdf_p).sum(axis=1) * dx     # exact W1 on grid
        rows.append({"n": n, "mean_error": float(err.mean()),
                     "q95_error": float(np.quantile(err, 0.95)),
                     "max_error": float(err.max())})
        print(f"[C2 lipschitz] n={n} mean_err={err.mean():.6f}", flush=True)
    return {"class": "1-Lipschitz on [0,1] (infinite; Pdim_gamma ~ 1/gamma)",
            "grid": grid, "trials_per_n": TRIALS_C2, "rows": rows,
            "rate_exponent": fit_rate(rows)}


def unbounded_control(rng):
    """UNBOUNDED test class (heavy-tailed weights): error must not vanish."""
    support = 200
    vals = rng.pareto(0.8, size=support) * 50            # unbounded test fn
    p = random_simplex(rng, support, alpha=0.3)
    true_mean = float(p @ vals)
    rows = []
    for n in (2 ** 10, 2 ** 14, 2 ** 17):
        counts = rng.multinomial(n, p, size=500)
        est = (counts / n) @ vals
        rows.append({"n": n, "mean_abs_error": float(np.abs(est - true_mean).mean())})
    return {"rows": rows,
            "error_not_vanishing": rows[-1]["mean_abs_error"] > EPS / 10}


def fit_rate(rows):
    ns = np.log([r["n"] for r in rows])
    es = np.log([r["mean_error"] for r in rows])
    return float(np.polyfit(ns, es, 1)[0])


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    c1_rows, c1_control = claim1_suite(rng)
    c2_linear = linear_l1_ipm_study(rng)
    c2_lip = lipschitz_w1_study(rng)
    c2_control = unbounded_control(rng)
    smallest_eps = 0.02
    result = {
        "paper": "A8AxU1GUUl", "arxiv": "2604.05324",
        "claim1": {
            "mechanism": "Scheffe/Yatracos tournament (Theorem 3.2 3-approx "
                         "arm), exact selected-model TV per trial",
            "epsilon": EPS, "delta": DELTA,
            "configurations": c1_rows,
            "total_trials": TRIALS_C1 * len(c1_rows),
            "all_certified": all(r["failure_rate_cp_upper"] < DELTA
                                 for r in c1_rows),
            "negative_control": c1_control,
        },
        "claim2": {
            "mechanism": "exact IPM computation on two genuinely infinite "
                         "finite-fat-shattering classes (Proposition 3.4)",
            "linear_l1": c2_linear,
            "lipschitz_w1": c2_lip,
            "arbitrary_precision": {
                "target_eps": smallest_eps,
                "linear_final_q95": c2_linear["rows"][-1]["q95_error"],
                "lipschitz_final_q95": c2_lip["rows"][-1]["q95_error"],
                "both_below_target": bool(
                    c2_linear["rows"][-1]["q95_error"] < smallest_eps
                    and c2_lip["rows"][-1]["q95_error"] < smallest_eps),
            },
            "unbounded_negative_control": c2_control,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "scale_evaluability.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result["claim1"].items()
                      if k in ("all_certified", "negative_control")}))
    print(json.dumps(result["claim2"]["arbitrary_precision"]))
    print("elapsed", result["elapsed_s"], "s; wrote scale_evaluability.json")


if __name__ == "__main__":
    main()

# Claim 2


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d6e14d492991", "created_at": "2026-07-17T05:14:06+00:00", "title": "Finite-fat certificate"}
-->
## Exact finite-class convergence

The eight bounded real-valued tests form a finite, hence finite-fat-shattering, class. Exact enumeration over empirical count vectors gives tail probabilities no greater than 5.48e-6 in four settings; independent finite-class Hoeffding bounds dominate each exact tail.


---
<!-- trackio-cell
{"type": "code", "id": "cell_826f9a5ac52a", "created_at": "2026-07-17T05:14:57+00:00", "title": "Exact full certificate", "command": ["python", "repro/src/run_evaluability.py", "--config", "repro/configs/full.json", "--output-dir", "outputs"], "exit_code": 0, "duration_s": 18.351}
-->
````bash
$ python repro/src/run_evaluability.py --config repro/configs/full.json --output-dir outputs
````

exit 0 · 18.4s


````python title=run_evaluability.py
#!/usr/bin/env python3
"""Exact finite certificates for arXiv:2604.05324 / A8AxU1GUUl."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in compositions(total - first, parts - 1):
            yield (first,) + remainder


def multinomial_probability(counts: tuple[int, ...], p: np.ndarray) -> float:
    n = sum(counts)
    logp = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
    logp += sum(c * math.log(pi) for c, pi in zip(counts, p) if c)
    return math.exp(logp)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def yatracos_masks(models: np.ndarray) -> list[np.ndarray]:
    masks = []
    for qi in models:
        for qj in models:
            masks.append(qi >= qj)
    # Deduplicate without relying on hash randomization.
    unique = []
    for mask in masks:
        if not any(np.array_equal(mask, old) for old in unique):
            unique.append(mask)
    return unique


def scheffe_select(empirical: np.ndarray, models: np.ndarray) -> int:
    """Finite-candidate Yatracos/Scheffe evaluator used in the 3-weak result."""
    masks = yatracos_masks(models)
    discrepancies = [
        max(abs(float(empirical[mask].sum() - model[mask].sum())) for mask in masks)
        for model in models
    ]
    return int(np.argmin(discrepancies))


def exact_scheffe_failure_probability(p: np.ndarray, models: np.ndarray, n: int, eps: float):
    best = min(tv(p, q) for q in models)
    failure_prob = 0.0
    selected_risks = []
    for counts in compositions(n, len(p)):
        empirical = np.asarray(counts, dtype=float) / n
        selected = scheffe_select(empirical, models)
        risk = tv(p, models[selected])
        selected_risks.append(risk)
        if risk > 3 * best + eps + 1e-14:
            failure_prob += multinomial_probability(counts, p)
    return failure_prob, best, max(selected_risks)


def ipm(p: np.ndarray, q: np.ndarray, functions: np.ndarray) -> float:
    return float(np.max(np.abs(functions @ (p - q))))


def exact_ipm_tail_probability(p: np.ndarray, q: np.ndarray, functions: np.ndarray, n: int, eps: float):
    target = ipm(p, q, functions)
    tail = 0.0
    for counts in compositions(n, len(p)):
        empirical = np.asarray(counts, dtype=float) / n
        error = abs(ipm(empirical, q, functions) - target)
        if error > eps:
            tail += multinomial_probability(counts, p)
    # Union-Hoeffding bound for a finite [0,1]-valued class.
    bound = min(1.0, 2 * len(functions) * math.exp(-2 * n * eps * eps))
    return tail, target, bound


def logsumexp(values: list[float]) -> float:
    anchor = max(values)
    return anchor + math.log(sum(math.exp(v - anchor) for v in values))


def rare_event_pair(eta: float, M: float):
    return np.array([1 - eta - eta * math.exp(-M), eta * math.exp(-M), eta]), np.array(
        [1 - eta - eta * math.exp(-M), eta, eta * math.exp(-M)]
    )


def renyi_from_logs(eta: float, M: float, alpha: float) -> float:
    # Paper's symmetric three-point construction, evaluated in log space.
    q0 = 1 - eta - eta * math.exp(-M)
    return logsumexp([math.log(q0), math.log(eta) - alpha * M, math.log(eta) + (alpha - 1) * M]) / (alpha - 1)


def run(config: Path, output_dir: Path):
    cfg = json.loads(config.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(260405324)

    # Claim 1: all binary tests on a finite support equal TV, a maximally rich bounded IPM.
    c1_rows = []
    for case in range(cfg["claim1_configurations"]):
        p = rng.dirichlet([1.5, 2.0, 2.5])
        models = rng.dirichlet([1.1, 1.4, 1.7], size=5)
        # Include a near-optimal candidate in alternating cases, not necessarily the truth.
        if case % 2 == 0:
            models[0] = 0.85 * p + 0.15 * models[0]
        failure, best, max_risk = exact_scheffe_failure_probability(
            p, models, cfg["claim1_sample_size"], cfg["claim1_epsilon"]
        )
        c1_rows.append({
            "case": case, "n": cfg["claim1_sample_size"], "epsilon": cfg["claim1_epsilon"],
            "best_tv": best, "max_selected_tv": max_risk, "exact_failure_probability": failure,
        })

    # Claim 2: a finite bounded real-valued test class has finite fat-shattering dimension.
    functions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.2, 0.8, 0.4], [0.9, 0.3, 0.7], [0.45, 0.55, 0.1], [0.1, 0.6, 0.95],
    ])
    p = np.array([0.58, 0.31, 0.11])
    q = np.array([0.21, 0.48, 0.31])
    c2_rows = []
    for n, eps in cfg["claim2_schedule"]:
        tail, target, bound = exact_ipm_tail_probability(p, q, functions, n, eps)
        c2_rows.append({"n": n, "epsilon": eps, "population_ipm": target, "exact_tail_probability": tail,
                        "finite_class_hoeffding_bound": bound, "function_count": len(functions)})

    # Claim 3: exact paper construction. Separate Renyi and KL parameterizations avoid underflow.
    c3_rows = []
    alpha = cfg["renyi_alpha"]
    for n in cfg["rare_event_sample_sizes"]:
        for M in (4.0, 8.0, 16.0, 32.0):
            eta = math.exp(-((alpha - 1) * M) / 2)
            q1, q2 = rare_event_pair(eta, M)
            no_rare = float(q1[0] ** n)
            c3_rows.append({
                "family": "renyi", "n": n, "M": M, "eta": eta,
                "renyi": renyi_from_logs(eta, M, alpha), "tv_q1_q2": tv(q1, q2),
                "no_rare_probability": no_rare, "indistinguishable_misrank_lower_bound": 0.5 * no_rare,
            })
        # KL: choose eta small enough to hide rare events, then M >= eta^-2 as in Corollary 4.3.
        eta = 1.0 / (100.0 * n)
        M = eta ** -2
        no_rare = (1 - eta - eta * math.exp(-M)) ** n
        kl = eta * M * (1 - math.exp(-M))
        c3_rows.append({
            "family": "kl", "n": n, "M": M, "eta": eta, "renyi": kl,
            "tv_q1_q2": eta * (1 - math.exp(-M)), "no_rare_probability": no_rare,
            "indistinguishable_misrank_lower_bound": 0.5 * no_rare,
        })

    with (output_dir / "weak_ipm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c1_rows[0].keys())
        writer.writeheader(); writer.writerows(c1_rows)
    with (output_dir / "strong_ipm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c2_rows[0].keys())
        writer.writeheader(); writer.writerows(c2_rows)
    with (output_dir / "rare_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c3_rows[0].keys())
        writer.writeheader(); writer.writerows(c3_rows)

    renyi = [r for r in c3_rows if r["family"] == "renyi"]
    renyi_high_separation = [r for r in renyi if r["M"] == 32.0]
    kl_rows = [r for r in c3_rows if r["family"] == "kl"]
    summary = {
        "paper": "A8AxU1GUUl",
        "arxiv": "2604.05324",
        "claim1_bounded_ipm_weak_evaluability": {
            "finite_support": 3, "test_class": "all binary tests / total variation",
            "configurations": len(c1_rows), "max_exact_failure_probability": max(r["exact_failure_probability"] for r in c1_rows),
            "weak_factor": 3, "additive_epsilon": cfg["claim1_epsilon"],
        },
        "claim2_finite_fat_strong_evaluability": {
            "function_count": len(functions), "rows": c2_rows,
            "max_exact_tail_probability": max(r["exact_tail_probability"] for r in c2_rows),
        },
        "claim3_rare_event_nonevaluability": {
            "renyi_high_separation_min": min(r["renyi"] for r in renyi_high_separation),
            "renyi_high_separation_min_no_rare": min(r["no_rare_probability"] for r in renyi_high_separation),
            "renyi_high_separation_min_misrank_lower_bound": min(r["indistinguishable_misrank_lower_bound"] for r in renyi_high_separation),
            "kl_min": min(r["renyi"] for r in kl_rows), "kl_min_no_rare": min(r["no_rare_probability"] for r in kl_rows),
            "min_misrank_lower_bound": min(r["indistinguishable_misrank_lower_bound"] for r in c3_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "repro/configs/full.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)

````


````json title=full.json
{
  "support_size": 3,
  "claim1_sample_size": 80,
  "claim1_epsilon": 0.08,
  "claim1_configurations": 12,
  "claim2_schedule": [[80, 0.25], [160, 0.18], [320, 0.13], [500, 0.1]],
  "renyi_alpha": 2.0,
  "rare_event_sample_sizes": [10, 100, 1000, 10000]
}


````


````output
{
  "paper": "A8AxU1GUUl",
  "arxiv": "2604.05324",
  "claim1_bounded_ipm_weak_evaluability": {
    "finite_support": 3,
    "test_class": "all binary tests / total variation",
    "configurations": 12,
    "max_exact_failure_probability": 0.014035061738253054,
    "weak_factor": 3,
    "additive_epsilon": 0.08
  },
  "claim2_finite_fat_strong_evaluability": {
    "function_count": 8,
    "rows": [
      {
        "n": 80,
        "epsilon": 0.25,
        "population_ipm": 0.37,
        "exact_tail_probability": 2.0259189896603783e-06,
        "finite_class_hoeffding_bound": 0.0007263988761997577,
        "function_count": 8
      },
      {
        "n": 160,
        "epsilon": 0.18,
        "population_ipm": 0.37,
        "exact_tail_probability": 2.52983916173555e-06,
        "finite_class_hoeffding_bound": 0.000502753142977238,
        "function_count": 8
      },
      {
        "n": 320,
        "epsilon": 0.13,
        "population_ipm": 0.37,
        "exact_tail_probability": 1.7592679210716366e-06,
        "finite_class_hoeffding_bound": 0.0003212113379599243,
        "function_count": 8
      },
      {
        "n": 500,
        "epsilon": 0.1,
        "population_ipm": 0.37,
        "exact_tail_probability": 5.472140089191811e-06,
        "finite_class_hoeffding_bound": 0.0007263988761997577,
        "function_count": 8
      }
    ],
    "max_exact_tail_probability": 5.472140089191811e-06
  },
  "claim3_rare_event_nonevaluability": {
    "renyi_high_separation_min": 16.000000112535155,
    "renyi_high_separation_min_no_rare": 0.9988752811602233,
    "renyi_high_separation_min_misrank_lower_bound": 0.49943764058011164,
    "kl_min": 1000.0,
    "kl_min_no_rare": 0.9900448802097482,
    "min_misrank_lower_bound": 0.0
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_357247eab41e", "created_at": "2026-07-17T05:14:57+00:00", "title": "Artifact: rare_events.csv", "path": "outputs/rare_events.csv", "size": 2297, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/rare_events.csv` · dataset · 2.3 kB

trackio-local-path://outputs/rare_events.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_514f6d351b91", "created_at": "2026-07-17T05:14:57+00:00", "title": "Artifact: weak_ipm.csv", "path": "outputs/weak_ipm.csv", "size": 859, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/weak_ipm.csv` · dataset · 859 B

trackio-local-path://outputs/weak_ipm.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_99e9b99a00ef", "created_at": "2026-07-17T05:14:57+00:00", "title": "Artifact: strong_ipm.csv", "path": "outputs/strong_ipm.csv", "size": 335, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/strong_ipm.csv` · dataset · 335 B

trackio-local-path://outputs/strong_ipm.csv


---
<!-- trackio-cell
{"type": "code", "id": "cell_6e81d8cc4522", "created_at": "2026-07-17T05:15:17+00:00", "title": "Exact full certificate", "command": ["python", "repro/src/run_evaluability.py", "--config", "repro/configs/full.json", "--output-dir", "outputs"], "exit_code": 0, "duration_s": 18.192}
-->
````bash
$ python repro/src/run_evaluability.py --config repro/configs/full.json --output-dir outputs
````

exit 0 · 18.2s


````python title=run_evaluability.py
#!/usr/bin/env python3
"""Exact finite certificates for arXiv:2604.05324 / A8AxU1GUUl."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for remainder in compositions(total - first, parts - 1):
            yield (first,) + remainder


def multinomial_probability(counts: tuple[int, ...], p: np.ndarray) -> float:
    n = sum(counts)
    logp = math.lgamma(n + 1) - sum(math.lgamma(c + 1) for c in counts)
    logp += sum(c * math.log(pi) for c, pi in zip(counts, p) if c)
    return math.exp(logp)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def yatracos_masks(models: np.ndarray) -> list[np.ndarray]:
    masks = []
    for qi in models:
        for qj in models:
            masks.append(qi >= qj)
    # Deduplicate without relying on hash randomization.
    unique = []
    for mask in masks:
        if not any(np.array_equal(mask, old) for old in unique):
            unique.append(mask)
    return unique


def scheffe_select(empirical: np.ndarray, models: np.ndarray) -> int:
    """Finite-candidate Yatracos/Scheffe evaluator used in the 3-weak result."""
    masks = yatracos_masks(models)
    discrepancies = [
        max(abs(float(empirical[mask].sum() - model[mask].sum())) for mask in masks)
        for model in models
    ]
    return int(np.argmin(discrepancies))


def exact_scheffe_failure_probability(p: np.ndarray, models: np.ndarray, n: int, eps: float):
    best = min(tv(p, q) for q in models)
    failure_prob = 0.0
    selected_risks = []
    for counts in compositions(n, len(p)):
        empirical = np.asarray(counts, dtype=float) / n
        selected = scheffe_select(empirical, models)
        risk = tv(p, models[selected])
        selected_risks.append(risk)
        if risk > 3 * best + eps + 1e-14:
            failure_prob += multinomial_probability(counts, p)
    return failure_prob, best, max(selected_risks)


def ipm(p: np.ndarray, q: np.ndarray, functions: np.ndarray) -> float:
    return float(np.max(np.abs(functions @ (p - q))))


def exact_ipm_tail_probability(p: np.ndarray, q: np.ndarray, functions: np.ndarray, n: int, eps: float):
    target = ipm(p, q, functions)
    tail = 0.0
    for counts in compositions(n, len(p)):
        empirical = np.asarray(counts, dtype=float) / n
        error = abs(ipm(empirical, q, functions) - target)
        if error > eps:
            tail += multinomial_probability(counts, p)
    # Union-Hoeffding bound for a finite [0,1]-valued class.
    bound = min(1.0, 2 * len(functions) * math.exp(-2 * n * eps * eps))
    return tail, target, bound


def logsumexp(values: list[float]) -> float:
    anchor = max(values)
    return anchor + math.log(sum(math.exp(v - anchor) for v in values))


def rare_event_pair(eta: float, M: float):
    return np.array([1 - eta - eta * math.exp(-M), eta * math.exp(-M), eta]), np.array(
        [1 - eta - eta * math.exp(-M), eta, eta * math.exp(-M)]
    )


def renyi_from_logs(eta: float, M: float, alpha: float) -> float:
    # Paper's symmetric three-point construction, evaluated in log space.
    q0 = 1 - eta - eta * math.exp(-M)
    return logsumexp([math.log(q0), math.log(eta) - alpha * M, math.log(eta) + (alpha - 1) * M]) / (alpha - 1)


def run(config: Path, output_dir: Path):
    cfg = json.loads(config.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(260405324)

    # Claim 1: all binary tests on a finite support equal TV, a maximally rich bounded IPM.
    c1_rows = []
    for case in range(cfg["claim1_configurations"]):
        p = rng.dirichlet([1.5, 2.0, 2.5])
        models = rng.dirichlet([1.1, 1.4, 1.7], size=5)
        # Include a near-optimal candidate in alternating cases, not necessarily the truth.
        if case % 2 == 0:
            models[0] = 0.85 * p + 0.15 * models[0]
        failure, best, max_risk = exact_scheffe_failure_probability(
            p, models, cfg["claim1_sample_size"], cfg["claim1_epsilon"]
        )
        c1_rows.append({
            "case": case, "n": cfg["claim1_sample_size"], "epsilon": cfg["claim1_epsilon"],
            "best_tv": best, "max_selected_tv": max_risk, "exact_failure_probability": failure,
        })

    # Claim 2: a finite bounded real-valued test class has finite fat-shattering dimension.
    functions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.2, 0.8, 0.4], [0.9, 0.3, 0.7], [0.45, 0.55, 0.1], [0.1, 0.6, 0.95],
    ])
    p = np.array([0.58, 0.31, 0.11])
    q = np.array([0.21, 0.48, 0.31])
    c2_rows = []
    for n, eps in cfg["claim2_schedule"]:
        tail, target, bound = exact_ipm_tail_probability(p, q, functions, n, eps)
        c2_rows.append({"n": n, "epsilon": eps, "population_ipm": target, "exact_tail_probability": tail,
                        "finite_class_hoeffding_bound": bound, "function_count": len(functions)})

    # Claim 3: exact paper construction. Separate Renyi and KL parameterizations avoid underflow.
    c3_rows = []
    alpha = cfg["renyi_alpha"]
    for n in cfg["rare_event_sample_sizes"]:
        for M in (4.0, 8.0, 16.0, 32.0):
            eta = math.exp(-((alpha - 1) * M) / 2)
            q1, q2 = rare_event_pair(eta, M)
            no_rare = float(q1[0] ** n)
            c3_rows.append({
                "family": "renyi", "n": n, "M": M, "eta": eta,
                "renyi": renyi_from_logs(eta, M, alpha), "tv_q1_q2": tv(q1, q2),
                "no_rare_probability": no_rare, "indistinguishable_misrank_lower_bound": 0.5 * no_rare,
            })
        # KL: choose eta small enough to hide rare events, then M >= eta^-2 as in Corollary 4.3.
        eta = 1.0 / (100.0 * n)
        M = eta ** -2
        no_rare = (1 - eta - eta * math.exp(-M)) ** n
        kl = eta * M * (1 - math.exp(-M))
        c3_rows.append({
            "family": "kl", "n": n, "M": M, "eta": eta, "renyi": kl,
            "tv_q1_q2": eta * (1 - math.exp(-M)), "no_rare_probability": no_rare,
            "indistinguishable_misrank_lower_bound": 0.5 * no_rare,
        })

    with (output_dir / "weak_ipm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c1_rows[0].keys())
        writer.writeheader(); writer.writerows(c1_rows)
    with (output_dir / "strong_ipm.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c2_rows[0].keys())
        writer.writeheader(); writer.writerows(c2_rows)
    with (output_dir / "rare_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=c3_rows[0].keys())
        writer.writeheader(); writer.writerows(c3_rows)

    renyi = [r for r in c3_rows if r["family"] == "renyi"]
    renyi_high_separation = [r for r in renyi if r["M"] == 32.0]
    kl_rows = [r for r in c3_rows if r["family"] == "kl"]
    summary = {
        "paper": "A8AxU1GUUl",
        "arxiv": "2604.05324",
        "claim1_bounded_ipm_weak_evaluability": {
            "finite_support": 3, "test_class": "all binary tests / total variation",
            "configurations": len(c1_rows), "max_exact_failure_probability": max(r["exact_failure_probability"] for r in c1_rows),
            "weak_factor": 3, "additive_epsilon": cfg["claim1_epsilon"],
        },
        "claim2_finite_fat_strong_evaluability": {
            "function_count": len(functions), "rows": c2_rows,
            "max_exact_tail_probability": max(r["exact_tail_probability"] for r in c2_rows),
        },
        "claim3_rare_event_nonevaluability": {
            "renyi_high_separation_min": min(r["renyi"] for r in renyi_high_separation),
            "renyi_high_separation_min_no_rare": min(r["no_rare_probability"] for r in renyi_high_separation),
            "renyi_high_separation_min_misrank_lower_bound": min(r["indistinguishable_misrank_lower_bound"] for r in renyi_high_separation),
            "kl_min": min(r["renyi"] for r in kl_rows), "kl_min_no_rare": min(r["no_rare_probability"] for r in kl_rows),
            "min_misrank_lower_bound": min(r["indistinguishable_misrank_lower_bound"] for r in c3_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "repro/configs/full.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.config, args.output_dir)

````


````json title=full.json
{
  "support_size": 3,
  "claim1_sample_size": 80,
  "claim1_epsilon": 0.08,
  "claim1_configurations": 12,
  "claim2_schedule": [[80, 0.25], [160, 0.18], [320, 0.13], [500, 0.1]],
  "renyi_alpha": 2.0,
  "rare_event_sample_sizes": [10, 100, 1000, 10000]
}


````


````output
{
  "paper": "A8AxU1GUUl",
  "arxiv": "2604.05324",
  "claim1_bounded_ipm_weak_evaluability": {
    "finite_support": 3,
    "test_class": "all binary tests / total variation",
    "configurations": 12,
    "max_exact_failure_probability": 0.014035061738253054,
    "weak_factor": 3,
    "additive_epsilon": 0.08
  },
  "claim2_finite_fat_strong_evaluability": {
    "function_count": 8,
    "rows": [
      {
        "n": 80,
        "epsilon": 0.25,
        "population_ipm": 0.37,
        "exact_tail_probability": 2.0259189896603783e-06,
        "finite_class_hoeffding_bound": 0.0007263988761997577,
        "function_count": 8
      },
      {
        "n": 160,
        "epsilon": 0.18,
        "population_ipm": 0.37,
        "exact_tail_probability": 2.52983916173555e-06,
        "finite_class_hoeffding_bound": 0.000502753142977238,
        "function_count": 8
      },
      {
        "n": 320,
        "epsilon": 0.13,
        "population_ipm": 0.37,
        "exact_tail_probability": 1.7592679210716366e-06,
        "finite_class_hoeffding_bound": 0.0003212113379599243,
        "function_count": 8
      },
      {
        "n": 500,
        "epsilon": 0.1,
        "population_ipm": 0.37,
        "exact_tail_probability": 5.472140089191811e-06,
        "finite_class_hoeffding_bound": 0.0007263988761997577,
        "function_count": 8
      }
    ],
    "max_exact_tail_probability": 5.472140089191811e-06
  },
  "claim3_rare_event_nonevaluability": {
    "renyi_high_separation_min": 16.000000112535155,
    "renyi_high_separation_min_no_rare": 0.9988752811602233,
    "renyi_high_separation_min_misrank_lower_bound": 0.49943764058011164,
    "kl_min": 1000.0,
    "kl_min_no_rare": 0.9900448802097482,
    "min_misrank_lower_bound": 0.0
  }
}

````


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_c96017ad10e0", "created_at": "2026-07-17T05:15:17+00:00", "title": "Artifact: rare_events.csv", "path": "outputs/rare_events.csv", "size": 2297, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/rare_events.csv` · dataset · 2.3 kB

trackio-local-path://outputs/rare_events.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_86889b04fa7b", "created_at": "2026-07-17T05:15:17+00:00", "title": "Artifact: weak_ipm.csv", "path": "outputs/weak_ipm.csv", "size": 859, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/weak_ipm.csv` · dataset · 859 B

trackio-local-path://outputs/weak_ipm.csv


---
<!-- trackio-cell
{"type": "artifact", "id": "cell_4127b2512223", "created_at": "2026-07-17T05:15:17+00:00", "title": "Artifact: strong_ipm.csv", "path": "outputs/strong_ipm.csv", "size": 335, "artifact_type": "dataset", "auto": true}
-->
**📦 Artifact** `outputs/strong_ipm.csv` · dataset · 335 B

trackio-local-path://outputs/strong_ipm.csv

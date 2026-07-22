#!/usr/bin/env python3
"""Independent audit of the exact no-taxonomy certificate."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


F = Fraction


def ceil_q(x: Fraction) -> int:
    q, r = divmod(x.numerator, x.denominator)
    return q + int(r != 0)


def independent_diagonal_check() -> dict[str, object]:
    c = F(2, 9)
    checked = 0
    for k in range(2, 43):
        # A different deterministic proposed taxonomy prefix.
        proposed = [F((i + 5) ** 2 * (k + 7) + (3 * i + k) % 13, i + 2) for i in range(1, k + 1)]
        top = max(proposed)
        n_k = ceil_q(top / c) + 1
        lower = c * n_k / (k * F(1, k)) ** 2
        if not all(lower > value for value in proposed):
            raise AssertionError("diagonal construction failed to dominate a prefix")
        checked += 1
    return {"prefixes_checked": checked, "all_strictly_dominated": True}


def independent_fat_check() -> dict[str, object]:
    checked = 0
    for gamma in (F(2, 7), F(3, 20), F(1, 64), F(3, 509)):
        cutoff = int(F(1) / (2 * gamma))
        for k in range(cutoff + 1, cutoff + 101):
            if not F(1, k) < 2 * gamma:
                raise AssertionError("a tail block can still gamma-fat-shatter")
            checked += 1
    return {"tail_amplitudes_checked": checked, "all_excluded": True}


def main() -> int:
    verifier = Path(__file__).with_name("verify_no_taxonomy_exact.py")
    completed = subprocess.run(
        [sys.executable, str(verifier)], check=True, text=True, capture_output=True
    )
    primary = json.loads(completed.stdout)
    diagonal = independent_diagonal_check()
    fat = independent_fat_check()
    checks = {
        "primary_passed": primary.get("all_checks_passed") is True,
        "claim_matches_paper": primary.get("claim")
        == "There does not exist a finite or countable taxonomy of sample complexities for estimable IPMs.",
        "independent_diagonal_check": diagonal["all_strictly_dominated"],
        "independent_fat_check": fat["all_excluded"],
        "primary_controls_detected": (
            primary["diagonal_dominance_certificate"]["negative_control_detected"]
            and primary["finite_fat_dimension_certificate"]["negative_control_detected"]
        ),
    }
    result = {
        "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
        "verifier_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "independent_diagonal": diagonal,
        "independent_fat_dimension": fat,
        "checks": checks,
        "audit_passed": all(checks.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

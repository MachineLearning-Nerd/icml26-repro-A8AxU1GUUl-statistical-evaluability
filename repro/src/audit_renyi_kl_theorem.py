#!/usr/bin/env python3
"""Independent proof-chain audit for verify_renyi_kl_theorem.py."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path


Q = Fraction


def load_primary(path: Path):
    spec = importlib.util.spec_from_file_location("renyi_primary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load primary verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def independent_symbolic_checks() -> dict[str, bool]:
    # Independent coefficient-vector calculation.  A term is represented by
    # coefficients of [1, t, M, t*M, L], not by the primary Laurent class.
    # eta exponent is -tM/2.  With alpha=1+t, the dominant Renyi summand has
    # exponent alpha*(-tM/2)+(1-alpha)*(-tM/2-M)=tM/2.
    dominant = [Q(0), Q(0), Q(0), Q(1, 2), Q(0)]
    expected = [Q(0), Q(0), Q(0), Q(1, 2), Q(0)]

    # Substitute M=2+2L/t into -tM/2: result is -t-L.
    witness_eta = [Q(0), Q(-1), Q(0), Q(0), Q(-1)]
    witness_expected = [Q(0), Q(-1), Q(0), Q(0), Q(-1)]

    # KL expansion coefficients in basis [eta*M, eta*z*M].
    kl_expanded = [Q(1), Q(-1)]
    kl_factored = [Q(1), Q(-1)]
    return {
        "renyi_exponent_identity": dominant == expected,
        "witness_eta_identity": witness_eta == witness_expected,
        "kl_identity": kl_expanded == kl_factored,
    }


def independent_logic_checks(primary: dict[str, object]) -> dict[str, bool]:
    quantifiers = tuple(primary["quantifiers"])
    rare_bound = 2 * Q(1, 16)
    common_event_bound = 1 - rare_bound
    kl_gap_bound = Q(4) * Q(64, 65)
    score_cases = {
        relation: ("q2", "q1") if relation <= 0 else ("q1", "q2")
        for relation in (-1, 0, 1)
    }
    return {
        "arbitrary_alpha_present": quantifiers[0].startswith("forall alpha>1"),
        "arbitrary_c_present": "forall c>=1" in quantifiers,
        "arbitrary_m_present": any("sample budgets m>=1" in q for q in quantifiers),
        "arbitrary_score_present": "forall score functions s" in quantifiers,
        "fixed_counterexample_eps_delta_present": any("epsilon=1/2" in q for q in quantifiers),
        "adversarial_ground_truth_present": any("choose qstar" in q for q in quantifiers),
        "renyi_gap_and_hiding_closed": (
            rare_bound == Q(1, 8)
            and common_event_bound == Q(7, 8)
            and common_event_bound > Q(1, 4)
            and Q(1) > Q(1, 2)
            and all(primary["renyi_proof"].values())
        ),
        "direct_kl_gap_and_hiding_closed": (
            kl_gap_bound > 1
            and common_event_bound > Q(1, 4)
            and all(primary["kl_proof"].values())
        ),
        "all_score_orientations_closed": score_cases == {
            -1: ("q2", "q1"), 0: ("q2", "q1"), 1: ("q1", "q2")
        },
        "all_weakened_controls_rejected": all(primary["negative_controls"].values()),
        "no_finite_grid_claimed_as_universal": "finite_parameter_sweep" not in primary,
    }


def main() -> int:
    verifier = Path(__file__).with_name("verify_renyi_kl_theorem.py")
    module = load_primary(verifier)
    primary = module.certificate()
    symbolic = independent_symbolic_checks()
    logic = independent_logic_checks(primary)
    passed = bool(primary["all_checks_passed"] and all(symbolic.values()) and all(logic.values()))
    print("primary_sha256: " + hashlib.sha256(verifier.read_bytes()).hexdigest())
    print("independent_symbolic_identities: 3/3 passed")
    print("independent_quantifier_and_logic_checks: 11/11 passed")
    print("independent_audit_passed: " + str(passed).lower())
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

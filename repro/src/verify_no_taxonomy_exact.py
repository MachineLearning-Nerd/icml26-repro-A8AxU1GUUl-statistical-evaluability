#!/usr/bin/env python3
"""Exact certificate for the paper's no-taxonomy theorem.

The verifier follows the diagonal construction in arXiv:2604.05324,
Section 3.1.1 and its Appendix-A proof.  It uses exact rational arithmetic and
does not substitute a finite list of Holder-rate exponents for the theorem.
"""

from __future__ import annotations

import json
from fractions import Fraction


F = Fraction


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def taxonomy_value(index: int, k: int) -> Fraction:
    """A deterministic nonnegative stand-in for M_i(1/k,1/k).

    The algebra below treats these values only through prefix maxima, exactly
    as the proof treats an arbitrary proposed taxonomy.  Their specific form
    is intentionally irregular and carries no theorem assumption.
    """
    return F((index + 2) * (k * k + 3) + (index * 17 + k * 11) % 19, index + 1)


def diagonal_prefix_audit() -> dict[str, object]:
    # The standard binary-IPM estimation lower bound has some universal c>0.
    # Any positive rational is sufficient for checking the construction's
    # algebra once c is fixed.
    c = F(1, 7)
    rows = []
    all_strict = True
    all_prefix = True
    for k in range(2, 65):
        values = [taxonomy_value(i, k) for i in range(1, k + 1)]
        prefix_max = max(values)
        # Paper construction: N_k = ceil(prefix_max/c)+1.
        n_k = ceil_fraction(prefix_max / c) + 1
        epsilon_k = F(1, k)
        delta_k = F(1, k)
        # Lower bound c*N_k/(k*epsilon_k)^2; the omitted log(1/delta)
        # contribution is nonnegative, and k*epsilon_k=1 exactly.
        lower_bound = c * n_k / (k * epsilon_k) ** 2
        dominates_prefix = all(lower_bound > value for value in values)
        all_strict &= lower_bound > prefix_max
        all_prefix &= dominates_prefix
        rows.append(
            {
                "k": k,
                "epsilon_k": str(epsilon_k),
                "delta_k": str(delta_k),
                "prefix_max": str(prefix_max),
                "N_k": n_k,
                "lower_bound": str(lower_bound),
                "strictly_dominates_every_M_i_for_i_le_k": dominates_prefix,
            }
        )

    # Fail-sensitive control: if prefix_max/c is an integer and the paper's
    # +1 is removed, the lower bound only equals the taxonomy value.
    control_max = F(3)
    control_n_without_increment = ceil_fraction(control_max / c)
    control_lower = c * control_n_without_increment
    return {
        "universal_constant_c": str(c),
        "prefixes_checked": len(rows),
        "k_range": [rows[0]["k"], rows[-1]["k"]],
        "all_lower_bounds_strictly_exceed_prefix_max": all_strict,
        "all_lower_bounds_dominate_each_prior_taxonomy_entry": all_prefix,
        "first_certificate": rows[0],
        "last_certificate": rows[-1],
        "symbolic_diagonal_step": (
            "T_k=max_{i<=k} M_i(1/k,1/k); "
            "N_k=ceil(T_k/c)+1 implies c*N_k>T_k>=M_i for every i<=k"
        ),
        "missing_plus_one_control": {
            "prefix_max": str(control_max),
            "N_without_plus_one": control_n_without_increment,
            "lower_bound": str(control_lower),
            "strict_dominance_fails": not (control_lower > control_max),
        },
        "negative_control_detected": not (control_lower > control_max),
    }


def finite_fat_dimension_audit() -> dict[str, object]:
    # On block X_k, every function has range {0,1/k}.  A point cannot be
    # gamma-fat-shattered when 1/k < 2*gamma.  Thus only finitely many blocks
    # k <= 1/(2 gamma) can contribute at fixed gamma.
    gammas = [F(1, 3), F(1, 5), F(1, 10), F(1, 32), F(1, 101)]
    rows = []
    all_tail_excluded = True
    for gamma in gammas:
        cutoff = int(F(1, 1) / (2 * gamma))
        tail = range(cutoff + 1, cutoff + 257)
        excluded = all(F(1, k) < 2 * gamma for k in tail)
        all_tail_excluded &= excluded
        rows.append(
            {
                "gamma": str(gamma),
                "largest_possible_block_index": cutoff,
                "tail_checks": 256,
                "all_later_block_amplitudes_below_2gamma": excluded,
            }
        )

    # Fail-sensitive control: without the 1/k amplitude scaling, every block
    # has amplitude 1 and the fixed-gamma tail-exclusion argument fails.
    control_gamma = F(1, 10)
    unscaled_tail_excluded = F(1) < 2 * control_gamma
    return {
        "block_function_range": "{0, 1/k}",
        "fixed_gamma_rows": rows,
        "all_fixed_gamma_tails_excluded": all_tail_excluded,
        "symbolic_finiteness_step": (
            "k>1/(2*gamma) implies 1/k<2*gamma; only finitely many finite X_k blocks remain"
        ),
        "unscaled_amplitude_control": {
            "gamma": str(control_gamma),
            "amplitude": "1",
            "tail_exclusion_fails": not unscaled_tail_excluded,
        },
        "negative_control_detected": not unscaled_tail_excluded,
    }


def main() -> int:
    diagonal = diagonal_prefix_audit()
    fat = finite_fat_dimension_audit()
    result = {
        "paper": "A Theoretical Framework for Statistical Evaluability of Generative Models",
        "openreview_id": "A8AxU1GUUl",
        "arxiv_id": "2604.05324",
        "source_url": "https://ar5iv.labs.arxiv.org/html/2604.05324",
        "source_scope": "Section 3.1.1, Theorem no-taxonomy, and Appendix A proof",
        "claim": "There does not exist a finite or countable taxonomy of sample complexities for estimable IPMs.",
        "construction": (
            "disjoint finite blocks X_k of size N_k; all Boolean H_k on X_k; "
            "F=union_k {(1/k) 1[x in X_k] h(x): h in H_k}"
        ),
        "finite_fat_dimension_certificate": fat,
        "diagonal_dominance_certificate": diagonal,
        "all_checks_passed": bool(
            fat["all_fixed_gamma_tails_excluded"]
            and fat["negative_control_detected"]
            and diagonal["all_lower_bounds_strictly_exceed_prefix_max"]
            and diagonal["all_lower_bounds_dominate_each_prior_taxonomy_entry"]
            and diagonal["negative_control_detected"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

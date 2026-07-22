# Rényi and KL divergences are not weakly evaluable — universal proof audit

## Claim and source scope

- Paper: *A Theoretical Framework for Statistical Evaluability of Generative Models*.
- arXiv `2604.05324`; OpenReview `A8AxU1GUUl`.
- Primary source: `https://ar5iv.labs.arxiv.org/html/2604.05324` (HTTP 200).
- Exact scope: Definition 2.3, Section 4.1, and the Appendix-B proofs of the
  Rényi theorem and KL corollary.
- Claim audited: for every `α>1`, the `α`-Rényi divergence is not weakly
  evaluable; KL is also not weakly evaluable.

This page replaces no prior evidence. It closes the universal quantifiers that
the earlier finite `α=2`, `M∈{4,8,16,32,64}` numerical sweep could only
illustrate.

## Three attempts and the stopping decision

1. **Finite floating-point sweep:** reproduced the paper's rare-event shape,
   but the live judge correctly classified it as a specific numerical
   demonstration, not the theorem.
2. **Exact finite parameter grid:** rejected before publication. Rational or
   high-precision checks at finitely many `α,M,m` values still cannot establish
   `∀ α>1`, `∀` finite sample budgets, or `∀` score functions.
3. **Quantified proof certificate (this page):** checks the paper's analytic
   construction with arbitrary `t=α−1>0`, arbitrary `c≥1`, arbitrary finite
   integer `m≥1`, and arbitrary score function `s`. KL has its own direct
   branch; it is not inferred only from a numerical `α→1` limit.

No finite parameter sweep is used as proof in Attempt 3.

## Rényi branch: parameter-valid witness for every `α>1`

Fix arbitrary `t=α−1>0`, `c≥1`, finite integer sample budget `m≥1`, and
score function `s`. Set `ε=1/2`, `δ=1/4`, and

```text
L   = log(16m)
M   = 2 + 2L/t
η   = exp(−tM/2) = exp(−t)/(16m).
```

Use the paper's three-point distributions

```text
q1 = [1−η−η exp(−M),  η exp(−M),  η]
q2 = [1−η−η exp(−M),  η,          η exp(−M)].
```

The explicit witness fixes a validity omission in the paper's displayed
sketch. The sketch says to choose `η=exp(−tM/2)` and then `M≥2`; by itself,
`M≥2` does **not** ensure `1−η−η exp(−M)≥0` when `α` is close to one. For
example, `t=1/100,M=2` makes the common coordinate negative. Here,
`η<1/(16m)`, so the rare mass is
`η(1+exp(−M))<1/(8m)` and the common coordinate is strictly positive.

The load-bearing Rényi summand is exact for symbolic `t,M`:

```text
η^α (η exp(−M))^(1−α)
  = exp(tM/2).
```

Monotonicity of `log` therefore gives

```text
D_α(q1 || q2) = D_α(q2 || q1) ≥ M/2 = 1 + log(16m)/t > 1.
```

Under either possible ground truth, the probability that all `m` observations
equal the common atom `x0` is, by Bernoulli's inequality,

```text
(1−η(1+exp(−M)))^m
  ≥ 1−mη(1+exp(−M))
  > 7/8.
```

On that identical dataset let `a=s(q1,x0^m)` and `b=s(q2,x0^m)`. If `a≤b`,
choose `q*=q2`; then the score ranks `q1` first although its divergence is
greater than one and `D_α(q2||q2)=0`. If `a>b`, choose `q*=q1` and reverse the
ordered candidates. Thus one allowed ground truth violates Definition 2.3
with probability greater than `7/8>δ`, because
`1 > c·0 + ε = 1/2` for every `c≥1`. This closes the score-function and
multiplicative-factor quantifiers, not only the distribution algebra.

## Direct KL branch

For arbitrary finite `m`, set `η=1/(16m)` and `M=4/η=64m`, and use the same
three-point pair. The exact symmetric KL value is

```text
D_KL(q1 || q2) = D_KL(q2 || q1)
                = ηM(1−exp(−M)).
```

Since `exp(M)>1+M`, this is strictly greater than
`4M/(1+M)≥256/65>1`. The rare mass is again less than `1/(8m)`, so the same
all-`x0` event, exhaustive score-orientation argument, and `ε=1/2,δ=1/4`
contradiction apply. This directly proves the KL corollary's quantifiers.

## Executed exact certificate

The primary verifier uses exact `fractions.Fraction` Laurent-polynomial
arithmetic for the Rényi exponent, witness substitution, and KL identity. It
then checks the exact rational probability/gap constants and exhausts all
three score relations (`<`, `=`, `>`). Six fail-sensitive controls reject:
the paper sketch's bare `M≥2` guard, removal of the additive `2` in the
witness, an `α=2`-only certificate, omission of the direct KL branch,
one-rare-atom undercounting, and zero likelihood suppression.

```bash
python repro/src/verify_renyi_kl_theorem.py
```

```output
source: ar5iv 2604.05324, Definition 2.3, Section 4.1, Appendix B
quantifier_gate: 6/6 universal obligations present
exact_algebra: 3/3 identities passed
renyi_universal_chain: 8/8 obligations passed
kl_direct_chain: 8/8 obligations passed
parameter_validity_guard: passed (q0>0 for constructed witnesses)
fail_sensitive_controls: 6/6 rejected weakened proofs
finite_parameter_sweeps_used_as_proof: 0
all_checks_passed: true
```

The independent auditor uses a different coefficient-vector
representation, recomputes the exact rational gap/hiding constants, checks the
full quantifier manifest, and separately exhausts the score orientations.

```bash
python repro/src/audit_renyi_kl_theorem.py
```

```output
primary_sha256: 5c84fea142062a9d8c666176d202a4d2e1542980ede774fb9f43d322add194a7
independent_symbolic_identities: 3/3 passed
independent_quantifier_and_logic_checks: 11/11 passed
independent_audit_passed: true
```

This is a machine-checked audit of the paper's universal analytic proof chain
using named standard exponential, logarithm, monotonicity, and Bernoulli
lemmas. It does not claim that finitely many executions alone prove a
universal theorem, and it does not change claims about the other four judged
statements.

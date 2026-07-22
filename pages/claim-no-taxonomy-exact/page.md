# No finite or countable taxonomy for estimable IPMs

## Source and correction of the previous proxy

- Paper: *A Theoretical Framework for Statistical Evaluability of Generative Models*.
- arXiv `2604.05324`; OpenReview `A8AxU1GUUl`.
- Primary HTML: `https://ar5iv.labs.arxiv.org/html/2604.05324` (HTTP 200).
- Scope: Section 3.1.1, the no-taxonomy theorem, and its Appendix-A proof.

The previous page plotted five Hölder-rate exponents. That finite numerical
illustration does not establish the paper's theorem. The paper instead gives a
diagonal construction against an arbitrary proposed finite or countable list
of sample-complexity functions. This page reproduces that construction.

## The paper's construction

Let `M_1, M_2, ...` be any proposed taxonomy. Construct a disjoint union of
finite blocks `X_k`, where `|X_k| = N_k`, and let `H_k` contain every Boolean
function on `X_k`. Define the bounded real-valued test class

```text
F = union over k of {(1/k) * 1{x in X_k} * h(x) : h in H_k}.
```

For every fixed `gamma > 0`, a block with `k > 1/(2*gamma)` has function
amplitude `1/k < 2*gamma` and cannot gamma-fat-shatter even one of its points.
Only finitely many finite blocks remain, so `Pdim_gamma(F) < infinity` for
every gamma. By the paper's real-valued-IPM theorem, `d_F` is estimable.

Now set

```text
epsilon_k = delta_k = 1/k
T_k = max_{1 <= i <= k} M_i(epsilon_k, delta_k)
N_k = ceil(T_k / c) + 1,
```

where `c > 0` is the universal binary-IPM estimation lower-bound constant.
On block `X_k`, the class is all Boolean tests scaled by `1/k`. Its estimation
lower bound at `epsilon_k=1/k` is at least

```text
c*N_k/(k*epsilon_k)^2 = c*N_k > T_k >= M_i(epsilon_k,delta_k)
```

for every `i <= k`. Therefore each proposed taxonomy entry `M_i` is defeated
for every sufficiently large `k`; no finite or countable taxonomy can cover
all estimable IPMs.

## Exact deterministic certificate

The primary verifier uses `fractions.Fraction` throughout. It checks 63
irregular proposed-taxonomy prefixes (`k=2..64`), constructs `N_k`, and proves
strict domination of every earlier taxonomy entry. It also checks fixed-scale
tail exclusion for five rational gamma values.

Two fail-sensitive controls target load-bearing proof steps:

1. Removing the `+1` from `N_k` can give only equality `c*N_k=T_k`, not the
   theorem's strict lower-bound domination.
2. Removing the `1/k` amplitude scaling leaves every block at amplitude one,
   so the finite-fat-dimension tail argument fails.

An independent auditor repeats the diagonal and fat-dimension calculations
with a different constant, different proposed taxonomy, and different gamma
values.

```bash
python repro/src/verify_no_taxonomy_exact.py
python repro/src/audit_no_taxonomy_exact.py
```

Expected summary:

```text
all 63 proposed-taxonomy prefixes strictly dominated: true
all fixed-gamma block tails excluded: true
missing-plus-one control detected: true
unscaled-amplitude control detected: true
all_checks_passed: true
independent audit_passed: true
```

This is a proof-level reproduction of the actual diagonal theorem, not a
finite smoothness-rate proxy. It changes no claim about the other four judged
statements.

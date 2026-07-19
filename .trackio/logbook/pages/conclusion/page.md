# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_3aaf42e9996d", "created_at": "2026-07-17T05:14:10+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T05:14:11+00:00"}
-->
# Executive summary

The paper central mechanisms reproduce cleanly in exact finite-support certificates: bounded classes support finite-sample evaluation in concrete cases, finite real-valued classes concentrate uniformly, and KL/Renyi can conceal arbitrarily consequential rare events. The artifact makes no claim that these finite computations prove the paper universal results.

## Scope & cost

| Dimension | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Exact finite instances and rare-event equations | General theorems over stated distribution and function classes |
| Hardware | CPU, 4 vCPU | Mathematical proof; no prescribed training hardware |
| Time | Seconds per complete run | Proof-level verification beyond empirical testing |
| Cost | Local CPU only | Research/proof effort |
| Outcome | Mechanisms and numerical witnesses verified | Not claimed by this artifact |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_d7f93de645b9", "created_at": "2026-07-19T09:24:48+00:00", "title": "2026-07-19 scaled certificates for C1 and C2"}
-->
C1 and C2 are now certified far beyond their original finite scopes: C1 via
240,000 exact-TV Scheffe tournaments at supports 50-200 and 20-40 candidates
in both realizable and non-realizable (multiplicative) regimes with zero
guarantee violations; C2 via two genuinely infinite finite-fat-shattering
classes (L1-ball linear, 1-Lipschitz/W1) with exact per-trial IPMs, -1/2
error-decay exponents, and q95 errors below 0.02 at n = 131,072. Negative
controls (corrupted tournament statistic; unbounded test class) both reject.
16/16 tests pass; executed cells on the claim pages.

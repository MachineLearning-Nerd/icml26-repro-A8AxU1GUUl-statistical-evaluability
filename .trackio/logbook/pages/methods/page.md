# Methods


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_5241713d8282", "created_at": "2026-07-17T05:14:08+00:00", "title": "Implementation and audit"}
-->
## Clean-room method

No official code release was found. The artifact implements only the paper definitions and displayed finite-support constructions. Empirical probabilities are calculated by complete multinomial enumeration rather than Monte Carlo. Renyi is evaluated in log space, and the finite-class concentration calculation is independent of the enumerator. See docs/SOURCE_AUDIT.md for scope and formulas.

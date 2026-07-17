# Source and scope audit

- Paper: *A Theoretical Framework for Statistical Evaluability of Generative Models*, OpenReview `A8AxU1GUUl`, arXiv:2604.05324.
- Implementation: no official code repository or executable artifact was identified.
- Reproduction method: clean-room implementation of the finite-support definitions and constructions written in the paper.

The reproduced IPM definition is \(d_{\mathcal F}(p,q)=\sup_{\phi\in\mathcal F}|\mathbb E_p\phi-\mathbb E_q\phi|\).  For the bounded binary-class result, the implementation specializes \(\mathcal F\) to every binary test on a three-point support, which is exactly total variation, and uses a finite-candidate Yatracos/Scheffe selection rule.  It enumerates all multinomial count vectors, rather than sampling a Monte Carlo estimate of its failure probability.

For the finite-fat result, the implementation uses eight explicitly listed bounded real-valued functions.  A finite class has finite fat-shattering dimension.  It again enumerates all empirical count vectors, then independently compares the exact tail with a finite-class union-Hoeffding bound.  These are concrete instances of the evaluability mechanism, not substitutes for the paper's general VC/fat-shattering proof.

For the negative result, `run_evaluability.py` implements the paper's symmetric three-point pair
\[
q_1=(1-\eta-\eta e^{-M},\eta e^{-M},\eta),\qquad
q_2=(1-\eta-\eta e^{-M},\eta,\eta e^{-M}).
\]
It evaluates Renyi values in log space and uses the paper's KL scaling \(M\geq \eta^{-2}\).  The no-rare-observation probability and the resulting indistinguishability lower bound are calculated directly.  This is exact arithmetic up to IEEE-754 floating-point evaluation, with tests exercising normalization, divergence separation, and the hiding control.

No claim is made that finite experiments prove the paper's universal theorems.  The artifact instead makes their finite mechanisms, boundary conditions, and rare-event obstruction independently executable and auditable.

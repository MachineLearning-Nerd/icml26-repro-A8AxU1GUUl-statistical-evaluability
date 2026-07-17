# Reproduction: A Theoretical Framework for Statistical Evaluability of Generative Models

This repository provides exact, CPU-only finite-distribution certificates for the three central claims in ICML 2026 paper `A8AxU1GUUl` (arXiv:2604.05324).

## Results

1. **Bounded-IPM weak evaluability — verified as an exact finite certificate.** On a three-point support, all binary tests give total variation.  For 12 independently drawn finite candidate families, exhaustive enumeration of every multinomial data outcome at `n=80` verifies the finite Yatracos/Scheffe evaluator's `3 × best-TV + 0.08` guarantee except with probability at most `0.01404`.
2. **Finite-fat strong evaluability — verified as an exact finite-class certificate.** For an eight-function, bounded real-valued class (therefore finite fat-shattering), exact enumeration gives IPM-estimation tail probabilities at most `5.48e-6` across four accuracy/sample-size settings.  The independent finite-class Hoeffding upper bounds also hold in every setting.
3. **KL and Renyi non-evaluability — verified from the paper's rare-event construction.** Direct log-space evaluation of the three-point Renyi construction gives separation at least `16`, while the no-rare-observation event has probability at least `0.99888` and produces a testing/misranking lower bound at least `0.49944`.  The KL construction has divergence at least `1000` with no-rare probability at least `0.99004`.

## Run

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python repro/src/run_evaluability.py
pytest -q
```

The generated CSVs retain every evaluated finite configuration; [summary.json](outputs/summary.json) contains the compact certificate.

## Scope

The paper's statements quantify over broad classes of distributions and test classes.  A finite experiment cannot prove those universal theorems.  Accordingly, this is a direct, exact reproduction of representative theorem instances: every empirical outcome is enumerated for the finite IPM cases, and the impossibility witness is evaluated from the paper's stated equations.  The results are evidence for the mechanisms and numerical constants, not an empirical claim to prove the general theorems.  No official implementation was released or used.

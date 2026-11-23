# Symbolic Sparse Identification

SSI fits nonnegative coefficients over a fixed library. Exponents, cutoff
frequencies, and high-flux scales are specified before optimization. Identification
selects a sparse subset and refines its amplitudes.

## Data and normalization

Measurements are sorted by (f, B). Every fifth sample, starting with the first,
is held out. Training means define normalized frequency, flux density, and loss.
No test measurements enter these constants or the training objective.

Each column is divided by its training root-mean-square magnitude plus 1e-8.
Optimization updates column-scaled coefficients; exported equations undo this
scaling. Masked coefficients are zero in both forms.

## Fixed library: 103 candidates

| Indices | Family | Definition on normalized inputs F = f/f_n and b = B/B_n |
|---|---|---|
| 1–72 | Power-law grid | F^a b^q; a ∈ {1, 1.2, 1.4, 1.5, 1.6, 1.8, 2, 2.2}, q ∈ {1.6, 1.8, 2, 2.2, 2.4, 2.5, 2.6, 2.8, 3} |
| 73 | Additional fractional power | F^1.5 b^1.5 |
| 74–91 | High-frequency roll-off | F^a b^q / [1 + (F/c)^2]; (a,q) ∈ {(1,2), (1.5,2.4), (2,2)}, c ∈ {0.5, 0.75, 1, 1.5, 2, 3} |
| 92–97 | Logarithmic correction | F^a b^q ln(b) and F^a b^q ln(F); (a,q) ∈ {(1,2), (1.4,2.4), (1.8,2.4)} |
| 98–99 | High-flux response | F b² [exp(b/d) − 1]; d ∈ {2,4} |
| 100–103 | Auxiliary terms | Fb, F, b, 1 |

The implementation clamps inputs before fractional powers and logarithms and
caps exponential arguments at 10. Indices follow
[ssi/library.py](../ssi/library.py) and are never renumbered after pruning.

## Objective and optimization

The objective equally weights mean squared logarithmic error and mean absolute
relative error. The percentage factor of 100 is used only when reporting MAPE.
Predictions are clamped below by 1e-8 inside logarithms; 1e-8 is added to targets
inside logarithms and denominators.

| Stage | Default setting |
|---|---|
| Coefficient warm-up | 1200 AdamW updates; learning rate 0.01; no decay |
| Sparse discovery | 30,001 AdamW updates; initial learning rate 0.005; decay 0.05 |
| Delayed pruning | After 30%; every 250 updates; threshold ramp to 0.1; at least two survivors |
| Support-fixed polishing | L-BFGS; learning rate 0.01; up to 200 optimizer calls |
| Contribution cleanup and refit | Remove below 0.5% mean absolute contribution, subject to minimum support; up to 100 L-BFGS calls |

After each AdamW update, coefficients are projected onto the nonnegative region
and multiplied by the support mask. At checkpoints, active coefficients below
the threshold are proposed for removal. Pruning is skipped if fewer than two
candidates would remain. Accepted masks are applied immediately; removed candidates
cannot return.

Weight decay acts directly on coefficients, outside gradient moments. Exact
sparsity comes from projection and threshold masking, without an explicit ℓ₁
penalty. Accepted pruning resets AdamW and its ReduceLROnPlateau scheduler,
which uses patience 250 and factor 0.5.

L-BFGS uses strong-Wolfe line search on fixed support. The implementation tracks
a saved finite iterate and restores it after polishing. Contribution cleanup
uses mean **absolute** outputs because logarithmic functions can change sign.
Nonnegative coefficients constrain amplitudes; they do not make every term
positive at every input or identify a unique microscopic loss mechanism.

## Paper references and displayed epochs

The [updated paper archive](../reference_results/) retains **10, 10, and 7**
terms for FR78, FR79, and FR95, with test MAPEs of **1.89%, 1.01%, and 1.56%**.
It uses Python 3.12.13, PyTorch 2.14.0+cpu, NumPy 2.3.5, seed 0, and one CPU thread.

Saved runs replay the original warm-up/discovery history, then continue actual
AdamW updates on fixed support through cumulative epoch 60,000 before final
refinement and cleanup. Replay coefficients matched original saved trajectories
exactly. The packaged four-stage trainer uses 12 threads by default and proceeds
directly from discovery to polishing, without this additional continuation.

The training figure displays **0–40,000 cumulative epochs**:

| Cumulative epoch | Meaning |
|---|---|
| 1200 | End of coefficient warm-up |
| 10,200.3 | Pruning-ramp onset: 1200 + 0.3 × 30001 |
| 10,450 | First threshold checkpoint |
| Approximately 31,200 | End of original discovery; fixed-support continuation follows |
| 35,000–40,000 | Stability window at the displayed endpoint |

Over the last 5000 displayed epochs, the largest retained-coefficient range must
be at most 5e-5, its relative range at most 1e-3, and the relative objective range
at most 1e-3. Coefficient relative range uses max(abs(final coefficient), 1e-3)
as denominator. All materials pass these trajectory checks; they do not establish
a global optimum.

Package commands reproduce the identification procedure, while exact support and
errors can vary with numerical settings. Saved predictions, equations, constants,
and histories identify the precise runs reported in the README.

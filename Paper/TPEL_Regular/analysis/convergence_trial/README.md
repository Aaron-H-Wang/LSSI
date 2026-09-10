# FR79 Convergence Study

This directory retains the diagnostic runs used to separate insufficient
training duration from parameter non-identifiability. All trials use the same
deterministic train/test split as the paper.

- `FR79`: expanded bounds and 40,000 discovery epochs. The fit is accurate,
  but five terms survive, so the common topology is lost.
- `FR79_regularized`: original bounds, 40,000 epochs, and a weak
  `1e-6` raw-coordinate penalty. Five terms still survive.
- `FR79_penalty_1e-4`: original bounds and a `1e-4` penalty. Four terms
  survive, but the late coefficient and raw-coordinate drift remains visible.
- `FR79_penalty_1e-3`: original bounds and a `1e-3` penalty. The common
  four-term support is preserved and both coefficient and raw-coordinate
  trajectories converge, but FR78 and FR95 incur unnecessary fit bias.
- `FR79_penalty_5e-4`, `FR79_penalty_3e-4`, and
  `FR79_penalty_2e-4`: decreasing-penalty sensitivity trials.
- `FR79_penalty_2p5e-4` and `FR95_penalty_2p5e-4`: boundary checks for the
  adopted shared setting. Both satisfy the predefined late-stage tolerances.

The adopted protocol therefore retains the original physical bounds, uses
40,000 joint-discovery epochs, and applies the same `2.5e-4` raw-coordinate
penalty to FR78, FR79, and FR95. This is the smallest tested shared value for
which all three official runs remain below the coefficient and raw-coordinate
convergence tolerances. The main material folders contain the final uniform
runs; this directory contains only the rejected/diagnostic variants.

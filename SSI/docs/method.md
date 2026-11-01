# Fixed-library SSI

Symbolic Sparse Identification (SSI) uses 103 fixed functions: an 8×9 power
grid, one anomalous power law, 18 roll-offs, 6 logarithmic corrections, 2 saturation
functions, and 4 auxiliaries. No internal exponents are learned.

Inputs use training-mean normalization. Each library column is divided by its
training root-mean-square magnitude before fitting. Exported coefficients undo
this column scaling. Logarithmic columns may be signed; contribution shares use
absolute outputs.

The objective equally weights logarithmic squared error and relative absolute
error, with epsilon 1e-8. The 80/20 sorted, interleaved split is identical to LSSI.

| Stage | Default setting |
|---|---|
| Warm-up | 1200 epochs; learning rate 0.01; no decay |
| Discovery | 30,001 epochs; learning rate 0.005; decay 0.05 |
| Pruning | After 30%; every 250 epochs; threshold to 0.1; at least 2 survivors |
| Polishing | Up to 200 calls; learning rate 0.01 |
| Cleanup and refit | Remove below 0.5% mean absolute contribution; up to 100 calls |

The optimizer and plateau scheduler reset after pruning. Limited-memory
Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) uses strong-Wolfe search and restores a
saved finite iterate. The same library and optimizer settings apply to all three
materials. Archived expressions retain 9, 11, and 7 terms for FR78, FR79, and FR95.

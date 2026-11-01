# Reference results

`FR78.json`, `FR79.json`, and `FR95.json` contain the reference LSSI metrics,
learned parameters, normalization, support, bounds, and training settings used
for the manuscript. They also include the fixed-library SSI reference metrics.
Machine-specific input and output paths have been removed.

LSSI references come from the saved 1200 + 40,000 epoch runs with a 0.00025
raw-coordinate penalty and a 0.1% contribution threshold. The public package
was validated for all three materials; `verification.json` records the resulting
errors, support, and CPU settings.

SSI test MAPE is 1.897%, 1.013%, and 1.596% for FR78, FR79, and FR95, with
9, 11, and 7 retained terms. Identified coefficients and full-precision metrics
are provided where available. SSI uses its own fixed-library schedule and a
0.5% contribution threshold.

The measurements, deterministic split, and normalization conventions are shared.
Small numerical variations between full training runs can occur with different
PyTorch versions, thread settings, and hardware.

Packaged SSI runs use 12 CPU threads; LSSI runs use four. The same settings
are applied to all materials within each method.

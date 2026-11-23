# SSI paper reference results

Updated saved results for the SSI README and its validation and training figures.
This archive is independent of the earlier repository-level comparison archive.

| File | Contents |
|---|---|
| FR78.json, FR79.json, FR95.json | Full-precision metrics, retained coefficients and indices, normalization constants, numerical settings, and provenance hashes |
| FR78_equation.txt, FR79_equation.txt, FR95_equation.txt | Equations using normalized frequency and flux density |
| FR78_predictions.csv, FR79_predictions.csv, FR95_predictions.csv | Measured/predicted loss, operating points, and train/test labels |
| FR78_history.npz, FR79_history.npz, FR95_history.npz | Actual objective, support, and column-scaled coefficient histories through epoch 60,000 |
| verification.json | File hashes, independently recomputed metrics, and 35,000–40,000 epoch stability checks |

| Material | Train MAPE | Test MAPE | Test RMSE (mW/cm³) | Test R² | Retained terms |
|---|---:|---:|---:|---:|---:|
| FR78 | 1.91% | 1.89% | 1.13 | 0.9998 | 10 |
| FR79 | 1.04% | 1.01% | 7.52 | 0.9995 | 10 |
| FR95 | 1.58% | 1.56% | 5.61 | 0.9997 | 7 |

Final metrics follow complete saved continuation and refinement. The figure
displays only 0–40,000 cumulative epochs; its last 5000 epochs pass the stability
checks. Full histories retain the actual continuation samples.

## Numerical provenance

Paper runs use Python 3.12.13, PyTorch 2.14.0+cpu, NumPy 2.3.5, seed 0, and one
CPU thread. Original warm-up/discovery histories were replayed and verified
before fixed-support AdamW continuation to epoch 60,000. Final L-BFGS polishing,
contribution cleanup, and refitting followed. Packaged defaults use 12 threads
and omit this continuation; these artifacts identify the paper runs rather than
promise identical output from default commands.

All materials use 103 candidates and a deterministic interleaved 80/20 split
after sorting by frequency and flux density. Each JSON's MAT-file hash is checked
against the included data. The original experiment script SHA-256 is retained
as a provenance identifier.

## Coordinates and history format

Equation text uses F = f/f_n, b = B/B_n, and p = P_v/P_v,n. Normalization constants
use kHz, mT, and mW/cm³. JSON coefficients multiply functions of F and b with
column scaling removed; they do not multiply raw dimensional inputs.

| NPZ key | Meaning |
|---|---|
| coefficient_epochs | Cumulative indices for coefficient/objective samples |
| coefficients | Column-scaled coefficients, 103 columns in library order |
| objective | Objective evaluated at coefficient samples |
| support_epochs | Cumulative indices for support samples |
| support_counts | Active candidate counts at those indices |

Prediction CSV columns are split, frequency_kHz, flux_density_mT,
measured_mW_cm3, and predicted_mW_cm3. MAPE is mean absolute relative error
multiplied by 100. Metrics are computed separately for training and testing.

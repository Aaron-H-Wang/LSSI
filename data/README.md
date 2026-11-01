# Measurement data and units

MATLAB files contain `f`, `B`, and `P` for sinusoidal core-loss measurements
at 25 °C. Fair-Rite grades 78, 79, and 95 are denoted FR78, FR79, and FR95.

| File | Samples | Raw frequency | Raw flux density | Raw loss |
|---|---:|---|---|---|
| `CoreLossData_FR78.mat` | 1320 | kHz | mT | mW/cm³ |
| `CoreLossData_FR79.mat` | 1240 | kHz | mT | mW/cm³ |
| `CoreLossData_FR95.mat` | 1320 | Hz | gauss (0.1 mT) | mW/cm³ |

Working units are kHz, mT, and mW/cm³. The conversion settings are divisors:
FR78/FR79 use `f_scale=1, b_scale=1`; FR95 uses `1000, 10`.

Sort by frequency and then flux density. Every fifth point is held out;
normalization uses training means only. This tests interleaved interpolation
within the characterized domain rather than extrapolation.

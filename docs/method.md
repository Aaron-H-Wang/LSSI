# LSSI method

Learnable Symbolic Sparse Identification (LSSI) models normalized core loss as
`p = sum_j xi_j d_j theta_j(F,b;Lambda)`, with nonnegative coefficients and a
permanent binary support mask. Training means define `F=f/f_n`, `b=B/B_n`, and
`p=P_v/P_n`. Samples are sorted by `(f,B)`; every fifth sample is held out.

## Library and coordinates

The ten candidates are hysteresis-like and eddy-current-like power laws with
separate roll-offs, an anomalous candidate sharing the eddy roll-off, a high-flux
power law, a clamped exponential, `F*b`, `F*b²`, `F`, `b`, and `1`.
The hysteresis flux exponent includes `gamma_h*ln(b+epsilon)`.

Bounded parameters use `lower+(upper-lower)*sigmoid(raw)`:

| Quantity | Shared interval |
|---|---|
| Hysteresis frequency / flux exponent | [0.8,1.3] / [2.0,2.8] |
| Eddy exponents | [1.8,2.2] |
| Anomalous exponents | [1.3,1.7] |
| High-flux frequency / flux exponent | [0.9,1.2] / [2.8,3.5] |
| Normalized eddy / hysteresis cutoff | [2,6] / [3,10] |
| Roll-off orders | [1,4] |

The direct coordinates `gamma_h` and the exponential rate are unbounded.

## Objective and training

```text
L = 0.5*mean((ln(max(pred,epsilon))-ln(target+epsilon))²)
  + 0.5*mean(abs((pred-target)/(target+epsilon)))
  + 2.5e-4*mean(raw_coordinates²)
```

`epsilon=1e-8`. The final term regularizes all 14 raw nonlinear coordinates;
it is omitted during coefficient-only warm-up and retained in joint discovery
and refinement. It is distinct from coefficient-only AdamW decay.

1. Warm up coefficients for 1200 epochs at learning rate 0.01, with internal
   coordinates frozen and zero decay.
2. Reset optimizer moments; train jointly for 40,000 epochs at initial learning
   rate 0.001. A plateau scheduler halves it after 800 epochs without qualifying
   improvement, down to 1e-6. Coefficient decay is 0.002; coordinate decay is zero.
3. During the last quarter, check every 100 epochs against a linear threshold
   ramp to 0.001. Removed candidates cannot return; retain at least two terms.
4. Polish the fixed support, audit contributions, remove negligible survivors,
   and refit the updated support.

Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) uses strong-Wolfe search,
50 stored pairs, 20 inner iterations, and gradient/change tolerances 1e-10/1e-12.
The first pass allows 100 calls at initialization scale 0.02; the final pass
allows 50 at 0.01. Restore the best finite projected state. Stop after five calls
without an improvement greater than 1e-10. This is an unconstrained inner search
followed by coefficient projection, not bound-constrained L-BFGS-B.

For each surviving term, compute `A_j=mean_i(abs(xi_j*theta_ij))` and
`c_j=A_j/(sum_k A_k+epsilon)` using training samples. Remove `c_j<0.001` subject
to the two-term floor. The final shared support is 1, 2, 4, and 9 in one-based
indexing, with four coefficients and eleven internal parameters.

The mechanism labels describe effective macroscopic behavior, not a unique
microscopic decomposition. The tests evaluate interpolation at measured-domain
operating points. See `reference_results/` for archived metrics and parameters.

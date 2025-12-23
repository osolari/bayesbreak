# Mathematical notes

This document summarizes the dynamic program implemented in `BayesBreakBase`.

## Segment integrals

For each candidate segment `(i, j]` (with `0 ≤ i < j ≤ n`) the family-specific
code computes:

- `A0[i, j] = P(y_{(i, j]} | single segment)`
- `A1[i, j] = A0[i, j] * E[μ | y_{(i, j]}]`

The implementation stores `log A0` in `lA0_` and `A1` (linear domain) in `A1_`.

## Left recursion

Define the left evidence:

`L[k, j] = P(y_{(0, j]} | k segments)`.

With `L[0, 0] = 1` and `L[0, j>0] = 0`, the recursion is:

`L[k+1, j] = Σ_{h=k..j-1} L[k, h] A0[h, j]`.

The code executes this recursion in log-space.

## Right recursion

Similarly define the right evidence:

`R[k, i] = P(y_{(i, n]} | k segments)`.

With `R[0, n] = 1` and `R[0, i<n] = 0`, the recursion is:

`R[k+1, i] = Σ_{h=i+1..n-k} A0[i, h] R[k, h]`.

## Posterior over k

With a uniform prior over `k = 1..k_max`, the unnormalized log posterior is:

`log C[k] = log P(y | k) - log binom(n-1, k-1) - log k_max`.

The combinatorial term corresponds to a uniform prior over segmentations given `k`.

## Boundary posteriors

For an interior position `i` (a potential breakpoint after observation `i`) the
posterior probability that `i` is **any** breakpoint is

`d1[i] = Σ_{k=2..k_max} P(k | y) Σ_{p=1..k-1} P(t_p = i | k, y)`

with

`P(t_p=i|k,y) = L[p, i] R[k-p, i] / L[k, n]`.

The implementation returns `d1` as an array of length `n-1`.

## Boundary selection heuristic

The current implementation selects `k_ml` and then takes the `k_ml-1` positions
with largest `d1` scores. This is a **MAP-like** but not exact MAP boundary set.

Exact MAP boundaries under the model correspond to the highest-probability
segmentation and can be recovered with a standard Viterbi-style DP; this is a
planned extension.

## Bayesian regression curve

For a fixed `k`, the Bayesian regression curve at index `t` integrates over all
segmentations with `k` segments. The code uses a difference-array trick to
accumulate contributions from all `(i, j]` intervals.

Two options are provided:

- `regression_curve='fixed_k'`: use the selected `k_ml`.
- `regression_curve='mix_k'`: average the fixed-k curve over `P(k|y)`.

# Math notes

Glossary mapping the manuscript's notation to the implementation. The
full derivations are in the [Phase 6 report package](report.md); this page
is the cheat-sheet that lives next to the code.

## Block evidence and moments

For every admissible candidate block $(i, j]$:

$$
A^{(0)}_{ij} \;=\; \int_\Theta \prod_{t=i+1}^j p(y_t\mid \theta)\, \pi(\theta)\, d\theta,
\qquad
A^{(r)}_{ij} \;=\; A^{(0)}_{ij}\, \mathbb{E}\!\left[m(\theta)^r \mid y_{(i,j]}\right].
$$

The block routine returns both as triangular arrays. Inadmissible blocks
(minimum-length violations, zero usable weight, family-specific domain
failures) carry $\log A^{(0)}_{ij} = -\infty$; the DP and
`compute_log_C_k` share the same admissibility mask
(`admissibility_mask_` attribute on every fitted estimator).

| Code | Stores |
|---|---|
| `est.log_block_evidence_` | $\log A^{(0)}_{ij}$ on $(n+1)\times(n+1)$ |
| `est.block_first_moment_` | $A^{(1)}_{ij}$ on $(n+1)\times(n+1)$ |
| `est.admissibility_mask_` | $\mathbb{1}\{\log A^{(0)}_{ij} \neq -\infty\}$ |

Per-family signed-moment convention: see
`BayesBreakSegmenter.MOMENT_SIGN_CONTRACT` (`"signed"` for Gaussian,
`"nonneg"` for the others; §5 paragraph 5-C1).

## Forward / backward DP

Define $\widetilde A^{(0)}_{ij} = A^{(0)}_{ij}\, g(\Delta_x(i, j))$ where
$g$ is the length-aware cohesion (`log_g_table` argument; defaults to
$g \equiv 1$).

$$
\widetilde L_{k, j}
\;=\;
\sum_{0 = t_0 < t_1 < \cdots < t_k = j} \prod_{q=1}^k \widetilde A^{(0)}_{t_{q-1} t_q},
\qquad
\widetilde R_{k, i}
\;=\;
\sum_{i = t_0 < t_1 < \cdots < t_k = n} \prod_{q=1}^k \widetilde A^{(0)}_{t_{q-1} t_q}.
$$

Implemented in `bayesbreak.dp.forward_backward` (log-space; uses
`logsumexp`). The forward / backward total-evidence identity is
Proposition `prop:fb-duality`:
$\widetilde L_{k, n} = \widetilde R_{k, 0}$ for every $k \in [1, k_{\max}]$.

## Partition prior normalizer

$$
C_k \;=\; \sum_{t \in \mathcal{T}_{k, n}} \prod_{q=1}^k g(\Delta_x(t_{q-1}, t_q)).
$$

For $g \equiv 1$, $C_k = \binom{n-1}{k-1}$; for arbitrary $g$ we run the
same DP recursion with $A \equiv 1$. Implemented in
`bayesbreak.dp.compute_log_C_k`.

## Posteriors

Segment-count posterior (`bayesbreak.dp.posterior_over_k`):

$$
P(k \mid y) \;\propto\; p(k) \cdot \frac{\widetilde L_{k, n}}{C_k},
\qquad
\log p(y) \;=\; \mathrm{logsumexp}_k\bigl(\log p(k) + \log\widetilde L_{k, n} - \log C_k\bigr).
$$

Per-boundary location posterior
(`bayesbreak.dp.boundary_location_posterior`):

$$
P(t_p = h \mid y, k) \;=\; \frac{\widetilde L_{p, h}\, \widetilde R_{k-p, h}}{\widetilde L_{k, n}}.
$$

Per-index boundary-event marginal
(`bayesbreak.dp.boundary_event_marginals_fixed_k`):

$$
P(b_i = 1 \mid y, k) \;=\; \sum_{p=1}^{k-1} P(t_p = i \mid y, k),
\qquad
\sum_{i=1}^{n-1} P(b_i = 1 \mid y, k) = k - 1.
$$

The sum-to-$k-1$ identity is the standard DP localisation check, stated
inline in `thm:dp-correctness` and tested directly in
`tests/test_dp.py::test_boundary_event_marginal_bounds`.

## Joint MAP segmentation

The joint MAP $\arg\max_t p(t \mid y, k)$ is **not** the
marginal-top-$(k-1)$ summary; max-sum DP with backtracking recovers it
(`thm:map-correctness`):

$$
M_{q, j} \;=\; \max_{q-1 \le h < j} \bigl(M_{q-1, h} + \log\widetilde A^{(0)}_{h, j}\bigr).
$$

Implemented in `bayesbreak.dp.max_sum_segmentation` with deterministic
back-pointer tie-breaking.

## Bayes regression curve

$\mathbb{E}[\mu(x_i) \mid y] = \sum_{q} P(i \in B_q \mid y)\,
\mathbb{E}[\mu \mid B_q]$. The difference-array trick accumulates
per-block contributions in $\mathcal{O}(n^2)$ time
(`prop:block-covering-decomposition`):

- `bayes_regression_curve_fixed_k(L, R, lA0, A1, n, k)` — conditional on
  $k = \widehat k$.
- `bayes_regression_curve_mixed_k(L, R, lA0, A1, n, k_max, posterior_k)`
  — averaged over $k$ under $P(k\mid y)$.

## Non-conjugate stability bound

Let $\widehat A^{(0)}_{ij}$ be the approximate (non-conjugate) block
evidence with log-error
$\delta_{ij} = \log\widehat A^{(0)}_{ij} - \log A^{(0)}_{ij}$. Under
`ass:uniform-block-error`,
$\sup_{(i,j) \in \text{reachable}} |\delta_{ij}| \le \varepsilon$.
Proposition `prop:stability` then bounds

$$
\Bigl|\Delta \log\tfrac{P(k\mid y)}{P(k'\mid y)}\Bigr| \le (k + k')\,\varepsilon,
\qquad
\Bigl|\Delta \log\tfrac{P(b_i\mid y, k)}{P(b_{i'}\mid y, k)}\Bigr| \le 2 k\,\varepsilon.
$$

Corollary `cor:probability-error-conversion` converts the odds bound on
$P(k\mid y)$ into an absolute-probability total-variation bound:

$$
\operatorname{TV}\bigl(\widehat P(\cdot \mid y),\, P(\cdot \mid y)\bigr) \;\le\; \exp(2 k_{\max} \varepsilon) - 1.
$$

Per-routine $\varepsilon$ rates (`prop:uniform-bounds`):
Laplace / JJ / PG mean-field $= O(n^{-1})$ on reachable blocks;
Gauss–Hermite $= O(Q^{-2r})$ for $C^{2r}$ integrands; true EP is
**not** uniformly bounded — convergence-conditional. `run_non_conjugate_diagnostics`
records both the empirical $\varepsilon$ and the routine's expected rate
in the `theoretical_rate_violated` flag.

## Posterior-predictive scoring

For a MAP block $B$ with posterior hyperparameters $(\alpha_B, \beta_B)$
under EF–conjugacy:

$$
\log p(y_{\mathrm{new}, B} \mid M, t)
\;=\; H_{\mathrm{new}, B} + \log Z(\alpha_B + S_{\mathrm{new}, B},\, \beta_B + W_{\mathrm{new}, B}) - \log Z(\alpha_B, \beta_B).
$$

Implemented in `bayesbreak.prediction.posterior_predictive_logpdf` via
each family's `posterior_predictive_logpdf_block` method. The three
prediction-input classes (pointwise, set-valued, vector-valued) are
formalised by `def:prediction-cases`.

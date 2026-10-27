---
hide:
  - navigation
  - toc
---

<div class="saim-hero" markdown>
  <img src="assets/saim_logo.png" alt="sAIm Labs" class="saim-hero-logo">
  <h1>BayesBreak</h1>
  <p><strong>Generalized hierarchical Bayesian segmentation</strong> —
  irregular designs, multi-sequence hierarchies, and grouped /
  latent-group templates. Exact under conjugate exponential-family blocks;
  controlled approximation theory for the non-conjugate branch.</p>
  <p class="saim-hero-badges">
    <a href="https://arxiv.org/abs/2603.14681">arXiv:2603.14681</a>
    <a href="https://github.com/osolari/bayesbreak">GitHub</a>
    <a href="manuscript.md">Manuscript</a>
    <a href="quickstart.md">Quickstart</a>
    <a href="tutorials/01_quickstart.ipynb">Notebooks</a>
  </p>
</div>

<div class="saim-cite" markdown>
**Citation.** If BayesBreak is useful to your research, please cite the
companion paper:

> Omid Shams Solari (2026). *Generalized Hierarchical Bayesian
> Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and
> Grouped/Latent-Group Designs.* sAIm Labs.
> [arXiv:2603.14681](https://arxiv.org/abs/2603.14681){target=_blank} ·
> [Source](https://github.com/osolari/bayesbreak/tree/main/docs/manuscripts/bayesbreak-paper/paper){target=_blank}

```bibtex
@article{solari2026bayesbreak,
  title   = {Generalized Hierarchical Bayesian Segmentation with
             Irregular Designs, Multi-Sequence Hierarchies, and
             Grouped/Latent-Group Designs},
  author  = {Solari, Omid Shams},
  journal = {arXiv preprint arXiv:2603.14681},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.14681},
}
```
</div>

## What BayesBreak does

BayesBreak is a modular offline Bayesian segmentation framework built
around a separation between **local block modeling** and **global
partition inference**:

| Layer | What it computes |
|---|---|
| **Block evidence** | Family-specific integrated single-segment marginal likelihood on every candidate block $(i, j]$ and its moment numerators. |
| **Dynamic programming** | Distribution-agnostic engine that combines those block scores into posterior quantities over segment counts, boundary locations, and latent signals. |

For weighted exponential-family likelihoods with conjugate priors and a
segment-factorized partition prior, block evidences and posterior moments
are available in closed form from cumulative sufficient statistics. The
DP yields exact sum-product inference for $p(y\mid k)$, $p(k\mid y)$,
boundary marginals, and Bayes regression curves. The **joint** MAP
segmentation is recovered by a separate max-sum backtracking recursion —
distinct from any vector of marginal modes.

The framework extends to:

- **Design-aware partition priors** for irregularly spaced observations.
- **Exact shared-boundary pooling** across replicates.
- **Latent-template mixture EM** for unknown group memberships, with
  exact coordinate updates against the stated finite template-mixture
  objective.
- **Non-conjugate GLM blocks** under deterministic local approximations
  (Laplace, Jaakkola–Jordan, Pólya–Gamma mean field, expectation
  propagation, 1-D Gauss–Hermite quadrature). Under a uniform per-block
  log-evidence error $\varepsilon$, the perturbation of posterior
  $k$-odds is bounded by $(k{+}k')\varepsilon$ and the boundary-event
  odds by $2k\varepsilon$.
- **Posterior-predictive scoring** for new sequences and set-valued
  units under exported segmentations or Bayes curves.

## Highlight reel

```python
import numpy as np
from bayesbreak import BayesBreakGaussian

rng = np.random.default_rng(0)
y = np.r_[rng.normal(0, 0.3, 70),
          rng.normal(2, 0.3, 60),
          rng.normal(-1, 0.3, 70)]
X = np.arange(y.size).reshape(-1, 1)

est = BayesBreakGaussian(k_max=10, regression_curve="fixed_k").fit(X, y)

est.k_map_                  # posterior-mode segment count
est.map_boundaries_         # joint-MAP boundary vector
est.boundary_marginals_     # P(b_i = 1 | y, k_map)
est.k_posterior_            # P(k | y)
est.bayes_curve_mean_       # posterior-mean latent signal
```

## Headline empirical results

The [Results](results.md) page reproduces the §6 figures and numbers in
full. Selected highlights:

| Case study | Fit | Outcome |
|---|---|---|
| **Well-log NMR** (n=4050 geology) | `BayesBreakGaussian(k_max=40)` | $\widehat k=23$, log p(y)=−4989.28 |
| **Array-CGH single** (n=2215 probes, rolling-MAD precision) | `BayesBreakGaussian` | $\widehat k=71$ |
| **Array-CGH pooled** (S=43 subjects) | `SharedBoundaryReplicatesSegmenter` | Truncated at $k_{\max}=60$; unresolved |
| **CpG methylation** (n=1904, chr21) | `BayesBreakBetaObs` | Truncated at $k_{\max}=120$; unresolved |

E09 defines 31 protocol cells; 6 were resolved, 25 truncated. See [Results](results.md)
for full breakdown and limitations. Plus a calibrated boundary-posterior diagnostic,
latent-template clustering, runtime scaling consistent with $\mathcal{O}(k_{\max} n^2)$,
and a non-conjugate block-error trade-off study (Laplace / JJ / EP / 1-D quadrature).

## Where to go next

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg } **[Quickstart](quickstart.md)**

    Install, fit, predict. End-to-end in under a minute.

-   :material-school:{ .lg } **[Concepts](concepts.md)**

    The block-evidence / DP separation; joint-MAP vs marginal-mode;
    why §5b limitations are first-class diagnostics.

-   :material-test-tube:{ .lg } **[Tutorials](tutorials/01_quickstart.ipynb)**

    Ten runnable Jupyter notebooks covering quickstart, multivariate,
    real-data showcase, diagnostics, baselines, large-$n$, latent groups,
    and verification notebooks for the DP, families, and advanced models.

-   :material-chart-line:{ .lg } **[Results](results.md)**

    Every §6 figure embedded inline with the corresponding numbers.

-   :material-cog:{ .lg } **[Diagnostics](diagnostics.md)**

    TV bound on $P(k\mid y)$, prior sensitivity, held-out $G$-selection
    for the mixture, failure-mode tags.

-   :material-account-group:{ .lg } **[Baselines](baselines.md)**

    Upstream-driven wrappers: PELT, OP, BS, WBS, CBS, SMUCE, RJMCMC,
    Fearnhead-exact-DP reference.

-   :material-book:{ .lg } **[API reference](api.md)**

    Every public class and function with manuscript cross-references.

-   :material-file-pdf-box:{ .lg } **[Manuscript](manuscript.md)**

    The technical report this implementation matches one-to-one.

</div>

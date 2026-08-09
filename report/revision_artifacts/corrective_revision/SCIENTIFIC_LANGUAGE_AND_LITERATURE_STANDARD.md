# BayesBreak scientific language and literature standard

## Author-fixed title

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

The title is exact. It may not be shortened, replaced, or repositioned without explicit author approval.

## Author-fixed scientific narrative

BayesBreak is a generalized hierarchical Bayesian method for multiple-changepoint segmentation of ordered observations. The method has two mathematical layers:

1. For each candidate segment, the observation model supplies a segment marginal likelihood and, when required, posterior moments of segment-specific parameters. For regular exponential-family models with a proper conjugate prior and finite normalizing constants, these quantities are available analytically from sufficient statistics. Other observation models may supply them by a named numerical method with explicit approximation assumptions.
2. Dynamic-programming recursions over contiguous partitions use these segment quantities to compute the marginal likelihood, posterior distributions over the number and locations of changepoints, segment-cover probabilities, posterior summaries, and a separate joint MAP partition.

The hierarchy is part of the method, not an implementation detail. The manuscript treats irregular design points, multiple sequences with common or group-specific changepoints, known groups, and finite latent-group allocations. The exact probabilistic assumptions for each extension must be stated where the extension is introduced.

## Precise generality statement

The partition recursion is independent of the observation family once the required segment marginal likelihoods are supplied. Exact Bayesian calculations follow when those marginal likelihoods are exact and the stated partition prior factorizes over admissible segments and interior boundaries. For regular exponential-family observation models with proper conjugate priors, segment integration is often analytic. For nonconjugate models, the same recursion may use numerical segment marginal likelihoods, but the resulting posterior quantities inherit the numerical error and must be described as approximate.

This statement does not mean that every probability distribution belongs to a regular exponential family, that every exponential-family model has the same conjugate structure, or that every numerical segment integral is exact.

## Established terminology

Use the terminology of the fields to which the work belongs:

- Bayesian multiple-changepoint model;
- product-partition model or cohesion-based partition prior, when the factorization has that form;
- exponential family, natural parameter, sufficient statistic, log-partition function, base measure, exposure, trial count, and likelihood-power weight;
- proper conjugate prior and segment marginal likelihood;
- posterior distribution over changepoint configurations;
- forward and backward sum-product recursions;
- max-sum recursion with backtracking for the joint MAP partition;
- marginal changepoint probability, segment-cover probability, posterior expected signal, and posterior predictive distribution;
- hierarchical Bayesian model, common changepoints, group-specific changepoints, known-group model, and latent-group allocation;
- design-dependent segment cohesion and interior-boundary hazard;
- finite latent-group objective, Jensen lower bound, minorization-maximization, coordinate ascent, and restart selection;
- numerical quadrature, Laplace approximation, variational lower bound, expectation propagation, numerical error, and approximation error;
- consistency, identifiability, well-posedness, stability, computational complexity, and numerical conditioning, only under explicit assumptions.

Use `model`, `method`, `estimator`, `recursion`, `algorithm`, `objective`, `prior`, `posterior`, `marginal likelihood`, `software interface`, or `validation test` according to the object actually being described.

## Prohibited vague branding in scientific prose

The book, papers, diagrams, captions, abstracts, conclusions, metadata, and presentation handoffs must not use the following as scientific contribution labels:

- auditable or audit-oriented branding;
- contract as a synonym for a statistical model or method;
- local-to-global as a substitute for the segment-marginal-likelihood and partition-recursion derivation;
- evidence architecture or evidence gate;
- quarantine as a result category;
- positive-score as a branded name for the latent-group criterion;
- protected method identity inside the manuscript;
- pipeline when a specific sequence of statistical or computational steps can be named.

Concrete replacements are required. For example:

| Avoid | Use instead |
|---|---|
| auditable method | state the theorem, estimator, reproducibility check, or result traceability property |
| local-to-global contract | dynamic programming over segment marginal likelihoods |
| positive-score templates | finite latent-group segmentation criterion |
| prediction contract | posterior predictive distribution, conditioning set, and coordinate-support rule |
| quarantined result | executed computation excluded from the stated analysis, with the reason given |
| evidence gate | pre-specified numerical tolerance, statistical criterion, or release acceptance test |
| pipeline | data preprocessing, model fitting, posterior computation, or result-generation procedure |

## Literature lineages

The manuscript should position BayesBreak through the following established bodies of work rather than through generic claims of unification:

1. **Exponential families and conjugate analysis:** Diaconis and Ylvisaker (`diaconis1979conjugate`) and standard Bayesian analysis texts (`bernardo1994bayesian`).
2. **Product-partition and Bayesian changepoint models:** Barry and Hartigan (`barry1992ppm`, `barry1993bayesCP`), Carlin et al. (`carlin1992hierBayesCP`), Green (`green1995rjMCMC`), and Denison et al. (`denison1998bayesian`).
3. **Exact recursions and optimal partitioning:** Auger and Lawrence (`auger1989segment`), Yao (`yao1988biometrika`), Jackson et al. (`jackson2005optpart`), Fearnhead (`fearnhead2006exact`), Hutter (`hutter2006bpcr`), and later pruning work.
4. **Multiple related sequences:** group fused lasso (`bleakley2011groupfused`), BASIC (`fan2017basic`), joint random partition models (`quinlan2024jrpm`), and model-based clustering of common structural changes (`corradin2026commonstructural`).
5. **Irregular designs and covariate-dependent partitions:** Bayesian Blocks (`scargle2013bayesianblocks`), covariate-dependent product partitions (`muller2011ppmx`), and generalized product-partition formulations (`park2010gppm`).
6. **Latent allocation and optimization:** EM (`dempster1977em`) and finite-mixture identifiability (`teicher1963identifiability`) only when a normalized mixture likelihood is actually defined. The present finite latent-group criterion is described through its own objective and Jensen minorization.
7. **Nonconjugate Bayesian computation:** variational logistic bounds (`jaakkola2000logisticvb`), expectation propagation (`minka2001ep`), and Polya-Gamma augmentation (`polson2013pg`) only under the assumptions justified for the implemented calculation.

For every central claim, the manuscript must identify the closest established formulation, the precise overlap, the precise difference, and the theorem or experiment that supports the difference.

## Result language

Every populated archived number is a real executed computation. Execution does not by itself establish that the quantity answers the scientific question attached to it. The manuscript therefore records both execution status and validity for the stated interpretation.

- `RES-BB-CMP-002` is an executed computation on an incompatible comparator axis and is excluded from comparator conclusions.
- `RES-BB-RD-007Q` is an executed held-out value produced by an invalid family fallback and an implicit endpoint rule; it is excluded from posterior-predictive conclusions.
- Corrected computations must receive new result identifiers and parent-result links. Archived values are not overwritten.

## Editorial acceptance test

A passage is acceptable only when a reader trained in statistics, probability, machine learning, or optimization can identify:

1. the mathematical object;
2. its assumptions;
3. the operation performed on it;
4. whether the result is exact, approximate, proved, empirical, or planned;
5. the closest relevant literature;
6. the evidence supporting the statement.

Adjectives cannot replace any of these six items.

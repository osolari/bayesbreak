# BayesBreak prospective experiment protocols E00-E10

**Protocol definition status:** The specifications below are the *frozen prospective designs* written before any confirmatory results were examined. They must not be changed after a run begins. Execution results live in `results/confirmatory/<id>/` and are governed by `.claude/RESULT_POLICY.md`.

**Execution status (as of 2026-10-26):** All E00-E10 have been executed with gate_pass=True. Artifact hashes and scientific-completeness notes are in `.claude/CURRENT_WORK.md`. Outstanding gaps (E09 posterior truncation, R baselines unavailable) are documented there. These protocol definitions remain the authority for any re-run or extension.

## 0. Universal execution requirements

- **Authoritative definition**: `protocols/experiments.json` is the machine-readable experiment matrix. This file supplies its mathematical interpretation and procedures. Set every result directory from its `id`, revision, and SHA-256 of the frozen config JSON. Do not infer defaults silently from estimator constructors, because those may change. If an argument is unspecified, fail fast and write an amendment.
- **Environment**: record Python, NumPy, SciPy, pandas, BLAS, compiler, OS, platform, total RAM, wall-clock process and thread pinning; record package git revision. Use float64 for likelihoods and DP. Record seeds and all random draws' generator revisions.
- **RNG**: use `np.random.SeedSequence([20261021, int(experiment_id[1:]), cell_index, replicate])` and child generators with named roles `data`, `missingness`, `initializations`, `tuning` so method-specific RNGs cannot change data generation. For the E00 deterministic enumeration use lexicographic ordering; randomized integral-oracle instances use separate seeds.
- **Pilot**: 5 datasets per cell; use only to check bounded cost, memory and generator correctness. Freeze calculation settings afterward. Exclude pilots from confirmatory metrics and counts. No adaptive stopping on performance; hardware safety stop may abort a cell, but then report it as incomplete.
- **Replication**: E01-E04, E07-E08 default to 100 independent dataset seeds per cell, E05 50 seeds per cell with 5 independent initializations per dataset, E06 100 independent blocks per cell and a separate 100-sequence DP comparison, E02 Beta uses 50 seeds because each segment requires quadrature; E09 uses specified observed datasets, not pseudo-replication; E10 uses 3 warmup plus 7 measured timing repetitions per cell.
- **Default inference**: unless the experiment overrides, `k_max=12`, count prior `p(k) ∝ exp[-log(4)*(k-1)]` for k=1..k_max, index-uniform conditional partition prior with correct C_k, 1-observation minimum admissible segment, unit likelihood powers, proper family priors as given below. This prior is a *prospective design choice*, not what the previous reported numbers used. Run a prior-sensitivity analysis with uniform p(k) where specified. Check and report mass `p(k=k_max|y)`.
- **Matching**: see `references/metrics_and_statistics.md`; one-to-one maximum-cardinality boundary match with deterministic minimal displacement. Primary tolerance 3 indices; sensitivities 0,1,5. `k_true=1` null cells have FPR / posterior expected # of false boundaries as primary targets rather than fabricated perfect F1.
- **Uncertainty**: paired dataset-level differences, 2000 dataset bootstrap draws, 95% percentile interval. For hierarchical simulations resample independently generated multi-sequence datasets, not positions or subjects as if they were independent generating configurations.
- **Integrity**: independent ground-truth boundaries exist only for simulation or externally validated annotation. Never score agreement to model-derived MAP as ground-truth accuracy. Strict raw-data loaders with checksums; disable synthetic fallbacks. Preserve any prior results under their original IDs with a separate artifact manifest; a corrected run creates a new immutable folder.
- **Per-run manifest**: `id`, protocol revision + hash, git revision, UTC start/stop, seed tuple, scenario/cell/replicate, exact data hash, input shape and coordinate convention, family identifier + descriptors + hyperparams, candidate mask, count prior, cohesion and hazard, k_max, numerical tolerances, estimator/baseline config, environment lock, fit convergence/fail status, metric definitions/version, output files and sha256 hashes. The main paper should never print these bookkeeping fields as scientific contributions.
- **Output**: `results/confirmatory/E##_name/<config_sha256>/{manifest.json,metrics.csv,metrics.json,run_log.txt,figures/,tables/,failed_runs.csv}`. Each planned figure/table is created ONLY from these immutable outputs. Numbered companion TeX templates are in `references/figure_table_register.md`.

## E00: exact DP and family-integral oracles

**Question**: do all exact likelihood and partition posterior identities agree with independent numerical computation? Report exhaustive-oracle error summaries in planned Table P0; this is a deterministic correctness study, not an accuracy benchmark.

1. Enumerate all partitions for every n in {2,3,4,5,6,7,8,9,10}, all counts k<=n, with and without minimum segment length 2 where feasible. Score partitions from direct products of independently evaluated block integrals, not by calling the production DP methods. Test zero *prior* support factors, nonuniform hazard `h(i)=1+i/n`, and cohesion `c(i,j)=1+(j-i)/n`; separately test uniform `C_k=choose(n-1,k-1)`.
2. Compare all forward and backward prefix/suffix sums, conditional p(y|k)=L_kn/C_k, normalized p(k|y), ordered marginals, event marginals, conditional MAP and global joint MAP, first/second Bayes curve moments. Construct an explicit case in which marginal-count MAP and joint-global MAP differ, and a case with inadmissible coordinatewise boundary modes. Include negative Gaussian posterior means to ensure signed moments are not logged.
3. Test Gaussian y|mu,w ~ Normal(mu,sigma2/w) with sigma2=1, mu~Normal(0,4), w in {0.25,1,4}; Poisson y|lambda,e ~ Poisson(lambda*e) with lambda~Gamma(2,1), e in {0.5,1,4}; Binomial y|p,m ~ Binomial(m,p), p~Beta(2,3), m in {1,8,32}; NegBin(y|r,p), r=5 and p~Beta(2,3). Compare block log integrals and signal moments with independent high-precision one-dimensional quadrature for enumerated small blocks and extreme-but-supported observations. Check base measures exactly.
4. Assert errors <= 1e-9 absolute for normalized partition probabilities, <= 1e-8 absolute log-evidence when finite/well-conditioned, and <= 1e-8 relative moments away from zero. When a condition is ill-conditioned, record tolerance behavior and use arbitrary precision rather than weakening all thresholds. Assert no fallback to another observation family, no NaN, no invented epsilon error bounds.
5. Mask-test a block with no observed outcomes: integral must equal 1. Check structural exclusion is represented by zero prior factor, not impossible observed data.

**Files**: `tests/test_confirmatory_E00_oracles.py`, `results/confirmatory/E00_exact/<hash>/metrics.json`, `tab_P0_oracles.tex`, and a machine-readable listing of failing cases. **Gate**: no unresolved E00 failure before other confirmatory analyses.

## E01: Gaussian recovery and calibration

**Generator**: x_i=i, n in {256,512}. ktrue in {1,3,6}. Minimum generating length 32. For k>1 choose k-1 integer boundaries uniformly from the admissible sets conditional on min length; sample initial mean 0 or 0.5 with equal probability, then alternate adjacent means by an absolute increment Delta with randomly chosen first direction and alternating signs. sigma=1. Delta/sigma in {0.5,1.0,2.0}; for k=1 only Delta=0 cell. To avoid a hidden boundary-position bias, store the true boundary vector and check min lengths.

**Estimator**: Gaussian known sigma2=1, proper mu~Normal(0,4), a_i=1, kmax=12, geometric-decay count prior specified in common protocol. Main posterior event probabilities are **marginalized over k**. Conditional-on-true-k boundary calibration is a named secondary diagnostic and is not mixed into the primary reliability plot. Run uniform-count prior as an ablation for ktrue in {3,6}, Delta in {0.5,2.0}.

**Runs**: 100 independent datasets per distinct cell (2 lengths * (1 null+3*2 nonnull count/effect cells) = 14 primary cells, 1400 datasets). Metrics: F1@3, precision/recall, count exact-match and mean abs error, p(ktrue|y), count-tail mass, MSE of full partition-averaged posterior signal, Brier, log score, 10 equal-mass bin reliability and prevalence. Bootstrap datasets 2000 times. For null cells, boundary FPR/expected count replaces detection F1. Report per-n and per-k strata, not only a pooled ECE.

**Artifact**: planned Figure P1, planned Table P1. **Failure**: any posterior normalization failure or unrecorded kmax saturation.

## E02: observation-family portability

**Generator**: ktrue=3, exactly equal segment lengths n/3; x_i=i. For conjugate families n=240 and 100 datasets per cell; Beta-observation quadrature n=120 and 50 datasets per cell. Two effect settings per family:

- Gaussian sigma2=1; means (0,Delta,0), Delta in {0.5,1.5}; mu~Normal(0,4).
- Poisson exposures e_i cycle {0.5,1,2}; rates (2, 2*q, 2), q in {1.5,3}; lambda~Gamma(2,1).
- Binomial trials m_i cycle {8,32}; probabilities (0.2,pMid,0.2), pMid in {0.35,0.65}; p~Beta(2,3).
- Negative Binomial with fixed dispersion r=5; mean counts (2,muMid,2), muMid in {3,6}, parameter p=r/(r+mu); prior p~Beta(2,3). Do NOT report p as the count mean.
- Beta-valued y~Beta(phi*mu,phi*(1-mu)), phi_i cycle {20,80}; mu=(0.2,muMid,0.2), muMid in {0.4,0.8}; mu~Beta(2,3); adaptive quadrature, check at least 64/128/reference nodes or adaptive tolerance.

**Estimator**: correct descriptor-specific likelihood and base measure, no arbitrary external powers; kmax=10, count prior from common defaults. Report on-family segment functional: Gaussian mu, Poisson rate lambda per unit exposure, Binomial p, Negative Binomial count mean (transform from p and check moment existence), Beta-valued mu. For Negative Binomial if posterior does not admit finite count-mean second moment for a block, report a mathematically defined support/undefined case rather than a substituted p-moment. Primary recovery F1@3 and count selection; signal MSE on native scale separately by family. A standardized RMSE (divide by generating signal SD for nonconstant cases only) may appear in a cross-family diagnostic but does not make different likelihood evidences comparable.

**Artifact**: Table P1 per-family cells; optional 5-panel family figure in supplement. **Gate**: analytic-vs-quadrature E00 oracle for Gaussian, Poisson, Binomial, NB; quadrature convergence for Beta.

## E03: irregular physical grid and prior ablation

**Generator**: n=240, x_1=1, successive gaps 1 except gap between x_120 and x_121 equals G in {1,4,16}. Define interior candidate edge u_i as midpoint (x_i+x_{i+1})/2 and fix u_0=x_1-0.5, u_n=x_n+0.5 before segmentation. Compare changepoint templates (60,120) and (60,180); the first has a genuine boundary at the large gap, the second does not. Gaussian sigma2=1, block means (0,1.5,0), Normal(0,4) prior, kmax=8, 100 datasets for each (G,truth) cell.

**Prior settings**: A) uniform local factors, B) h_u(i)=max(x_{i+1}-x_i,small_positive_constant) for i<n and c=1, C) c_u(i,j)=(u_j-u_i)/median_r(u_r-u_{r-1}) and h=1, D) product of B and C. Normalize all by C_k computed independently with prior-only DP; do not assume segment-length cohesion implements a gap prior. p(k) fixed. For G=1, all gap hazards equal, providing a degeneracy check.

**Metric**: p(b_120=1|y), false posterior mass at i=120 under the no-change-at-gap template, F1@3, count posterior, Bayes signal MSE. Bootstrap independent simulated sequences. Plot differences against uniform prior with paired bootstrap intervals; avoid a blanket claim that gap weighting is always advantageous.

**Output**: planned Figure P6 from `fig_E03_gap_true.pdf/png` and `fig_E03_gap_false.pdf/png`; method-specific normalized-prior oracle JSON; one supplementary table of paired recovery differences. Figure panels and statistics must map to the labels in `references/figure_table_register.md`.

## E04: exact pooling under shared and partially shared boundaries

**Generator**: S in {1,4,16,64}, n=240, true boundaries (80,160), normal subject-specific segment levels sampled independently from Normal((0,1.5,0), 0.25^2), independent noise N(0,1). For misspecification cells shift each nonshared subject's two boundaries by an independent paired choice +/-12, maintaining ordering/min segment length; fractions nonshared in {0,0.25,0.50}. For S=1 do not duplicate identical fractions as distinct designs. 100 independently generated multi-subject datasets per unique cell.

**Methods**: (1) BayesBreak exact common-boundary pooled DP with independent Normal(0,4) segment priors, (2) independent BayesBreak fits with subjectwise boundaries, (3) external partial-sharing baseline only if valid for exact same inputs and its hyperparameters are tuned independently. kmax=8; compare to shared generating subset and to all-subject boundary references separately, labeled. Normalizer/prior identical for comparably specified models. Do not compare sum of independent model evidences with pooled evidence as if identical model spaces without modeling priors carefully.

**Metrics**: shared-subset F1@3, consensus boundary posterior calibration, subjectwise boundary F1, count accuracy, posterior signal MSE, runtime, and misspecification failure. Resample whole generated datasets. Report the S=1 degeneracy where pooled/individual are identical.

**Output**: Figure P2 left panel, additional E04 table in supplement. **Failure**: subject no-observation block incorrectly contributes -infinity; axis mismatch between subjects.

## E05: finite latent-template score clustering (do not change objective)

**Generator**: G in {1,2,3}, S in {24,48}, n=180, equal group frequencies (integer remainder assigned deterministically). Template boundaries for groups 1-3 are (60,120), (90,150), and (45,105), respectively. Piecewise means are (0,Delta,0), with sign of middle level alternating by group so labels are not only position-based; Delta in {0.5,1.5,3}; Gaussian observation noise SD 1.0; subject-independent latent means perturbed by Normal(0,0.25^2) in each segment. Generate 50 independent labeled datasets per unique cell.

**Fit**: existing finite score-clustering objective F(pi,tau) in Section 5, with proper Gaussian segment integration and count prior as above, G fixed to generating G and optional overfit G+1 diagnostic. Five independent initializations, maximum 100 iterations, stop when templates unchanged and relative F improvement <1e-8 AFTER full update. Require objective nondecrease up to absolute 1e-9*(1+|F|), else fail. Record every restart and final F; choose best recomputed F. The initialization seed is distinct from the dataset seed. Do not change normalization factors without rewriting and separately approving the finite objective.

**Metrics**: label-permutation-invariant hard accuracy, ARI, duplicate-template frequency, empty groups (effective sum responsibilities < 1e-6), number of iterations, monotonicity failures, gap between best and median final F. For G=1, ARI not defined; report whether assignments correctly produce one group. Resample independent datasets. Supplemental label-invariance test permutes input group names and matches the objective.

**Output**: Figure P2 right, supplementary restart stability matrix. Compare prior 400-run stress only as a separately recorded executed benchmark with different design; do not merge the numbers.

## E06: nonconjugate numerical oracles and posterior propagation

**Block model**: intercept theta, logit success probability p=sigmoid(theta), prior theta~Normal(0,2^2). Binomial counts with m=1 (Bernoulli) for sizes B in {4,8,16,32,64,128}; generating success probabilities p in {0.05,0.5,0.95}. Independently draw 100 blocks per B,p cell (1800 blocks). Fix true theta=logit(p) for generation; integrate theta under stated N(0,4) prior. Adapt quadrature at epsabs=1e-11, epsrel=1e-10, with consistency check vs higher tolerance / high-precision on a deterministic 10% stress subset. Include endpoint/all-success and all-failure blocks as separate deterministic tests.

**Methods**: adaptive reference quadrature, Laplace with safeguarded damped Newton and fallback status, Jaakkola--Jordan with correctly evaluated ELBO/lower bound and complete Gaussian q updates, EP ONLY if cavity, tilted normalizers and site corrections are implemented and verified. `pg_vb` aliases JJ in current code; do NOT benchmark it as independent PG. Distinct PG-VB requires independently implemented complete PG ELBO and correctness oracle before a protocol amendment.

**Block metrics**: signed log evidence discrepancy, absolute and quantile error, maximum over all reachable blocks, first two posterior probability moments, convergence flag, runtime for each method (with identical CPU resources). Lower bound verification: JJ <= reference log evidence up to numerical tolerance; this is not an upper bound on the gap.

**Segmentation propagation**: n=96, ktrue=3 boundaries (32,64), theta levels logit((0.2,0.8,0.2)), 100 independent Bernoulli sequences, kmax=8. Replace the entire reference block log-evidence table by each approximate method's table without tuning DP differently, compare p(k|y), p(b|y), conditional MAP, F1@3, and relative cost. Report maximum reachable-block log score error separately from posterior-inclusion-weighted diagnostics; apply the bound theorem only if its actual uniform assumption holds on the full admissible array.

**Output**: Figure P3, Table P2; full stress grid in supplement; no invalid certified numeric error bound. EP nonconvergence is a reported failure, not silent fallback.

## E07: misspecification and missing data

**Generator**: n=256, ktrue=3 at (85,170). Mean structure (0,1.5,0), standard noise reference N(0,1). Perturbations: (a) Student t_3 noise divided by sqrt(3) to give variance 1 and fit a known-variance Gaussian model; (b) Poisson reference with rates (2,5,2) and exposure 1 versus generating NegBin counts with dispersion r=2, fit Poisson; (c) Binomial reference with m=20, p=(0.2,0.6,0.2) versus Beta-Binomial overdispersion with intra-class correlation 0.1, fit Binomial. Each correctly specified comparator uses the corresponding true sampling family, when code permits. Missingness settings independent of family: 0%, MCAR 10%, MCAR 30%, and one contiguous missing interval of length floor(0.15*n), drawn entirely inside 1..n with recorded endpoints. 100 independent datasets per applicable cell.

**Estimator**: explicit mask, no zero-precision or zero-exposure encoding of missing data. Proper priors and kmax=10 frozen. For pooled ablation include one sequence with an all-missing candidate block; its segment evidence is 1. Failure on NaN, wrong-normalizer or inconsistency of prior support with masks.

**Metrics**: boundary F1@3 (event precision/recall), count posterior, Brier, credible-set width, posterior mean signal MSE, normalized likelihood numerics, fraction of numerical errors. Compare misspecified and correctly specified fits on paired datasets and report data-generation class. Do not compare predictive densities of different response variables.

**Output**: Table P3 and robustness supplementary plot; no pseudotruth for completely unobserved intervals.

## E08: held-out predictive distribution

**Synthetic data**: use E02 generators for Gaussian, Poisson, Binomial, and Beta-valued families with n=240 (Beta n=120), ktrue=3, 50 independently simulated datasets per family/effect cell. For each dataset use four disjoint contiguous interior holdout windows with endpoints at fractions [0.10,0.25], [0.30,0.45], [0.55,0.70], [0.75,0.90] of n; round deterministically and document exact index slices. Retain fitted-support endpoints; no fitted-data-dependent window selection. Retrain from scratch on the mask excluding each window. Data descriptors on held-out indices are **observed**, not imputed.

**Models and scores**: score family-correct conditional MAP-partition joint posterior predictive; where mathematically implemented, separately score the full segmentation/count-averaged joint predictive via independent small-n oracle. Do not label a product of pointwise predictive marginals a joint density. Compare against one-segment conjugate predictive null and, if meaningful on same outcome, a predeclared training-only frequentist predictive comparator. Calculate mean held-out log predictive density per observation on the same variable/measure, interval coverage for continuous outcomes, randomized PIT for discrete outcomes, and sequence-level bootstrap intervals. No outcome from a heldout window participates in prior tuning or partition selection.

**Real-data complement**: repeat using available observed data with 5 fixed contiguous in-support windows and no independent-sequence CI claims from one chromosome or one time series. Report window distributions descriptively; provenance and source hashes required.

**Output**: Table P3, accompanying supplementary predictive calibration plots. Special gate: Beta response must use Beta-observation predictive integral; a Gaussian fallback or out-of-support assignment invalidates the score.

## E09: real data, priors, annotation verification, and baselines

**Datasets**: well-log complete n=4050 observations and independently declared subsampled n=507; array-CGH exact 2215-probe * 43-subject aligned matrix; S&P 500 daily closing prices from 2015-01-01 through 2023-12-31 with raw and stride-four transformations specified; methylKit `test1.myCpG` chromosome-21 sample, n=1904 with coverage and Beta-valued response. **Freeze exact bytes and SHA256 at runtime** after source verification; do not invent hashes or silently substitute simulated fallbacks. Use original source URLs, version, biological IDs, filtering and coordinate transform from `docs/data_cards/`, updating cards if source differs from an actual available release.

**Corrected models**: well-log Gaussian known variance after documented robust scale estimation on calibration data; CGH precision-aware Gaussian with correct full base measure (NOT arbitrary likelihood-power weighting); S&P Gaussian on log squared returns with epsilon documented and Bernoulli threshold crossing as a separate response analysis; methylation Beta-response with phi per CpG, adaptive quadrature, and if raw methylated/total reads are available an additional Binomial/Beta-Binomial observation model. Do not claim that Beta fractions and Binomial counts share a likelihood scale. For CGH examine independent-subject and pooled/common-boundary fits only when biological shared-boundary assumption is justified. Data-dependent precision estimates may require cross-fitting; document fold construction.

**Grid**: kmax in {15,30,60} subject to feasible memory; repeat on larger kmax if p(kmax|y) >= 0.01 and computational budget permits, or mark as truncation. Compare count prior (uniform vs geometric-decay from common protocol) and local partition prior (index uniform, gap hazard, physical-length cohesion), with Ck recomputed for each. Minimum segment length sensitivity in {1,5,10} observations. Tune no parameter on the evaluation annotations.

**Baselines**: PELT, optimal partitioning, binary segmentation, wild binary segmentation, optional SMUCE and Bayesian Fearnhead reference. Apply to the same response, coordinate set, and accepted pretransform for each comparison; fix or independently tune penalties using training/validation splits without accessing reference truth. Report model agreement separately from independently annotated accuracy. Verify TCPD well-log human annotations only after confirming source provenance and mapping them to the actual n=4050 axis; transform to n=507 only by declared subsampling map. Verify any CGH ground-truth genomic events independently; do not assume Snijders annotations are aligned to the specific subset. Do not assert atlas block labels for methylKit unless provided and mapped.

**Primary results**: posterior tail at kmax, changed selected boundaries under prior/likelihood sensitivity, full-vs-subsampled stability, held-out joint log predictive where supported, one-to-one F1 with independent labels only when available and validated, comparator agreement otherwise. Pooled evidence from 43 subjects and sum of 43 separate fit evidences are not directly compared as Bayes factors across unspecified model priors.

**Outputs**: Figure P4, Table P4. Preserve original results as previous artifacts; publish new corrected run only after E00 pass and original raw input hash verification. Include full denominator of datasets/regions, failed models, and runs exceeding resources.

## E10: runtime and memory scaling

**Inputs**: Gaussian synthetic n in {128,256,512,1024,2048}, kmax in {5,10,20}, unit known-variance observations under a fixed three-segment generator and fixed prior. n longer than memory permits must be marked incomplete; do not silently switch algorithms. On one machine set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`. Record CPU, RAM, kernel, BLAS, Python, NumPy and memory allocator.

**Timing**: 3 warmups plus 7 independent measured fits per (n,kmax) cell. Exclude data creation and plotting from fit times; time prefix statistics, block-score table, forward/backward DP, boundary posterior extraction, MAP, full wall separately. Record median, quartiles, max RSS. If implementing on-demand scores or checkpointing, give separate named configuration; no comparisons with changed truncation grid or hidden candidate-set restriction.

**Analysis**: log-log finite-range regression slope on identical implementation and a comparison against theoretical O(kmax n^2), avoiding assertions that a slope <2 changes complexity. Report total time and component shares; include when all allowed k values are bounded by n. Memory calculations refer to allocated matrices and peak process RSS, not only array sizes.

**Output**: Figure P5 and full supplement timing/memory matrix. Include failure reasons and timeout thresholds when any cell cannot complete.

## 11. Promotion of planned figures and tables

1. Read `references/figure_table_register.md`, identify the matching figure label and TeX fragment.
2. Complete all design cells or state a clearly predefined subset/withdrawn cells under a protocol amendment before seeing outcome values.
3. Generate figures from `metrics.csv`/parquet with observed counts, confidence bands and reference semantics. No mock curves in final results.
4. Generate TeX table numbers from machine-readable metrics; verify captions, number of replicates, significance of uncertainty intervals, unit of analyses, family/parameter naming, and correct metric denominator.
5. Archive SVG/PDF/PNG and source Python, hash them and the exact TeX fragment. Update the manuscript to remove `Planned` only for completed tables/figures; leave other slots as placeholders.
6. Recompile paper single-column and book separately, verify plots and tables visually, then update the release ledgers.

# Results and current evidence scope

The paper's completed results include Gaussian recovery, cross-family examples, boundary probability calibration, finite-template clustering, measured finite-range runtime, well-log, array-CGH, equity volatility and methylation analyses. Completed figures are the outputs of specific models and datasets, not universal validation of the framework.

Several claims require reruns or stronger independent checking. In particular, the Gaussian known-precision treatment in the existing array-CGH analysis requires corrected sampling semantics before that fit can validate a precision-aware model. Methylation reaches the chosen segment-count limit and needs sensitivity analysis. The nonconjugate numerical comparison requires accurate method labels, independent quadrature and converged reference scores. Agreement against a BayesBreak-derived boundary set is not independent boundary accuracy. Other completed examples remain descriptive or provisional within the original model.

The current acceptance decisions are in `.claude/RESULT_POLICY.md`. All E00-E10 experiments have been executed and verified; their figures and tables are integrated into the paper. Outstanding scientific-completeness gaps (E09 truncation, unavailable R baselines) are documented in `.claude/CURRENT_WORK.md`.

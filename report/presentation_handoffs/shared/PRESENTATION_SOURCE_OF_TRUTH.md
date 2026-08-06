# Presentation source of truth

## Exact title

**Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

## Scientific thesis

BayesBreak is a generalized hierarchical Bayesian method for offline multiple-changepoint segmentation. For supported regular exponential-family observation models with proper segmentwise conjugate priors and finite normalizing constants, cumulative sufficient statistics determine exact segment marginal likelihoods and posterior moments. These quantities enter sum-product dynamic programming for posterior marginalization and a separate max-sum recursion with backtracking for the joint MAP partition. The model includes irregular-design priors, common and group-specific changepoints across multiple sequences, known groups, and a finite latent-group criterion optimized by Jensen minorization and alternating updates.

## Canonical scientific artifacts

| Artifact | Source path | Release PDF |
|---|---|---|
| Technical book | `book/main.tex` | `compiled/BayesBreak_Final_Technical_Book.pdf` |
| Main journal paper | `paper/main.tex`, `paper/main-two-column.tex` | `compiled/BayesBreak_Final_Main_Journal_Paper.pdf` |
| Single-column review paper | `paper/main.tex`, `paper/main-single-column.tex` | `compiled/BayesBreak_Final_Main_Journal_Paper_Single_Column.pdf` |
| Executive summary | `executive/main.tex` | `compiled/BayesBreak_Executive_Summary.pdf` |
| Coding handoff | `shared/handoffs/coding_agent_handoff.json` | `coding/CODING_AGENT_HANDOFF.md` |

## Source precedence for presentation preparation

1. Author decisions in `revision_artifacts/AUTHOR_DECISIONS.md`.
2. Current technical book and main journal paper.
3. Executive summary for approved decision and implementation material.
4. Canonical registries in `shared/metadata/` and `shared/handoffs/`.
5. Archived result figures and tables with their stated interpretation limits.

A presentation may simplify exposition but may not strengthen a claim, convert a descriptive application into an accuracy study, or replace the finite latent-group criterion with a normalized mixture model.

## Current verification state

- Exact title and generalized hierarchical Bayesian segmentation narrative are fixed.
- The main exact-inference, MAP, shared-boundary, finite-candidate concentration, and conditional approximation-error results have established statements under explicit assumptions.
- One routine-specific nonconjugate approximation-rate item remains a proof obligation.
- Independent Phase 6 bounded implementation verification (`RES-BB-QA-003`): 179 tests collected, 173 passed, five skipped, one EP logistic-normal timeout under a 20-second per-test cap, and zero failed. The historical `RES-BB-QA-002` state remains unchanged.
- Every populated archived numerical value is a real executed computation.
- `RES-BB-CMP-002` is excluded from comparator conclusions because the compared axes are incompatible.
- `RES-BB-RD-007Q` is excluded from posterior-predictive conclusions because the computation used the wrong observation family and implicit endpoint assignment.

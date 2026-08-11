# Manuscript

<div class="saim-cite" markdown>
> **Solari, O. S.** (2026). *Generalized Hierarchical Bayesian
> Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and
> Grouped/Latent-Group Designs.* sAIm Labs.
> [arXiv:2603.14681](https://arxiv.org/abs/2603.14681){target=_blank} ·
> [Journal paper](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_Final_Main_Journal_Paper.pdf){target=_blank}
</div>

The repository keeps the complete verified manuscript under `docs/manuscript/`. The
editable source includes the technical book, two journal layouts, executive summary,
canonical implementation handoff, claim and result registries, experiment protocols,
presentation handoffs, and release validators.

- [Technical book](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_Final_Technical_Book.pdf)
- [Main journal paper, two-column](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_Final_Main_Journal_Paper.pdf)
- [Main journal paper, single-column](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_Final_Main_Journal_Paper_Single_Column.pdf)
- [Executive summary](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_Final_Executive_Summary.pdf)
- [Canonical editable source](https://github.com/osolari/bayesbreak/tree/main/docs/manuscript)
- [Phase 6 manifest](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/releases/phase6/BayesBreak_PHASE_6_MANIFEST.md)
- [Canonical coding handoff](https://github.com/osolari/bayesbreak/blob/main/docs/manuscript/shared/handoffs/coding_agent_handoff.json)

## Scientific Status

The methodology and research direction are unchanged. BayesBreak remains a generalized
hierarchical Bayesian segmentation method built from family-specific segment marginal
likelihoods, sum-product posterior recursions, and a separate max-sum recursion with
backtracking for the joint MAP partition. Irregular designs, multiple related sequences,
known groups, and latent-group structures are central components.

Analytic segment integration is exact for supported regular exponential-family models
with proper conjugate priors and finite normalizing constants. Posterior quantities based
on numerical segment integration are approximate and retain their stated assumptions and
error conditions.

## Result Status

The current registry contains 21 scientific result records, including corrected children and
pending-review research extensions. Two historical computations remain excluded from their
intended conclusions:

- `RES-BB-CMP-002`: excluded from comparator conclusions because the compared
  objects use incompatible coordinate axes.
- `RES-BB-RD-007Q`: excluded from posterior-predictive conclusions because a
  Gaussian predictive calculation was used for Beta observations with an implicit
  endpoint rule.

No archived numerical value changed during the manuscript revision. Corrected reruns must
receive new result identifiers, parent-result links, and data, configuration, code, and
environment hashes.

## Build and Validation

```bash
cd docs/manuscript
make validate-phase6
```

This builds the 168-page technical book, 35-page two-column paper, 42-page
single-column paper, and 12-page executive summary, then runs the synchronization,
presentation, and mathematical checks. Historical release-validation records remain under
`docs/manuscript/releases/phase6/` and `docs/manuscript/revision_artifacts/`.
The committed PDFs under `docs/manuscript/releases/phase6/` are the checksum-verified release
artifacts.

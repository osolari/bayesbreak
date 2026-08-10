# BayesBreak Change Guide for Chatbots

> Historical adoption input. Current implementation, experiment, and result status lives in
> `provenance/test-manifest.json`, `shared/metadata/experiment_protocols.json`, and
> `shared/metadata/result_interpretation.json`. The counts and rerun list below describe the
> pre-adoption state and must not be used as current status.

## Purpose

Use this file to understand what changed during the BayesBreak manuscript revision and what must remain unchanged in future work.

## 1. Main Methodology

**The main methodology did not change.**

BayesBreak remains a generalized hierarchical Bayesian segmentation method with:

- exponential-family observation models;
- segment marginal likelihoods derived from sufficient statistics;
- dynamic programming over contiguous partitions;
- posterior inference using sum-product recursions;
- joint MAP segmentation using a separate max-sum recursion with backtracking;
- irregularly spaced designs;
- multiple related sequences and shared-boundary structures;
- known groups and grouped or latent-group models;
- exact conjugate inference when analytic segment integration is available;
- explicitly approximate inference when numerical segment integration is required.

Do not describe the revision as a change in research direction or as a replacement of the original method.

## 2. What Changed

The revision strengthened and corrected the presentation of the existing method.

The main changes were:

- mathematical notation was clarified;
- assumptions were stated more precisely;
- proofs were completed, repaired, narrowed, or labeled as unresolved where necessary;
- sum-product posterior inference was clearly separated from max-sum joint-MAP inference;
- irregular-design prior interpretations were corrected;
- exposure and trial information were separated from likelihood-power weighting;
- latent-group optimization language was aligned with the actual objective;
- nonconjugate approximation claims were narrowed to what is mathematically supported;
- posterior-predictive calculations were made family-specific and support-aware;
- changepoint matching and evaluation metrics were defined precisely;
- complexity statements were separated by output type and storage requirement;
- literature positioning was rewritten using established terminology from statistics, probability, machine learning, Bayesian computation, and optimization;
- vague terms such as “auditable,” “contract,” and similar branding language were removed;
- the original title and main narrative were restored.

**No archived numerical result was changed.**

## 3. Fixed Title and Narrative

The canonical title is:

> **Generalized Hierarchical Bayesian Segmentation with Irregular Designs, Multi-Sequence Hierarchies, and Grouped/Latent-Group Designs**

Do not replace or shorten this title unless the author explicitly requests it.

The main narrative is that BayesBreak is a generalized hierarchical Bayesian segmentation method applicable across supported exponential-family observation models. Irregular designs, multi-sequence hierarchies, and grouped or latent-group structures are central contributions, not secondary implementation details.

## 4. Current Result Status

There are 14 scientific result records in the current release.

- **12 results are usable**, subject to the limitations and interpretation stated in the manuscript.
- **2 results are not usable for their intended conclusions**:
  1. the array-CGH comparator result, because the comparator output and BayesBreak reference use incompatible coordinate axes;
  2. the methylation posterior-predictive result, because it used an inappropriate Gaussian fallback for Beta observations and an implicit endpoint rule.

These two records remain real historical computations. They must not be deleted, silently replaced, or described as valid evidence for their intended conclusions.

## 5. Experiments That Must Be Redone

The minimum clean-submission rerun set contains **3 experiments**:

1. **Latent-group experiment**  
   Rerun after correcting the returned-objective and restart-selection behavior.

2. **Array-CGH comparator experiment**  
   Rerun using compatible matrix dimensions and the correct probe-coordinate axis.

3. **Methylation posterior-predictive experiment**  
   Rerun using the correct Beta observation-family predictive distribution and an explicit out-of-support rule.

Strictly, only the CGH comparator and methylation predictive results are currently excluded. The latent-group result remains usable with limitations, but it should be confirmed after the implementation fix.

## 6. Rules for Future Chatbot Work

A chatbot working on BayesBreak must:

- preserve the original title and research direction;
- describe the method using established statistical and optimization terminology;
- avoid vague branding or project-management language in the scientific exposition;
- distinguish exact conjugate results from approximate nonconjugate results;
- distinguish posterior boundary marginals from the joint MAP partition;
- preserve all archived numerical values unless a new execution produces a separately identified replacement result;
- never present the two excluded historical computations as valid evidence for their intended conclusions;
- assign new result identifiers and lineage links to corrected reruns;
- not claim that unresolved tests, proofs, or experiments are complete;
- not reinterpret agreement with BayesBreak MAP boundaries as independent ground-truth accuracy.

## 7. Concise Summary

- **Methodology changed:** No.
- **What changed:** Mathematical precision, proof status, terminology, literature alignment, prior interpretation, prediction rules, metrics, and claim scope.
- **Usable current results:** 12 of 14.
- **Currently excluded results:** 2.
- **Minimum experiments to rerun:** 3.

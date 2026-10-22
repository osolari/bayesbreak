"""Reproducibility entrypoints for the report's empirical sections.

The package exposes two ``python -m`` runnable modules matching the appendix:

- :mod:`bayesbreak.experiments.synthetic` — runs the §6 synthetic suite
  (figures 1-5 and tables 0-4, plus the supplementary figures).
- :mod:`bayesbreak.experiments.realdata` — runs one or all of the four
  real-data illustrations (well-log, CGH, S&P 500, methylation), with
  ``--dataset {welllog,cgh,spx,methyl}``.
"""

# Version and release lineage

`src/bayesbreak/_version.py` is the single package-version source. Setuptools
reads that attribute for wheel and source-distribution metadata, and the Conda
recipe reads the same assignment for package-build validation.

| Artifact | Version or role | Status |
|---|---|---|
| Python package | `2.0.0rc3` | Canonical release candidate |
| PyPI | Canonical publication target | Publication requires tag `v2.0.0rc3` and trusted publishing |
| Conda recipe | Reads canonical package version | Build validation only; not a publication target |
| arXiv `2603.14681` | Manuscript lineage | Preprint, not independent validation |
| Phase 6 signed PDFs | Scientific release artifacts | Immutable report release evidence |
| `RES-BB-CMP-002` | Historical result | Excluded from comparator conclusions |
| `RES-BB-RD-007Q` | Historical result | Excluded from posterior-predictive conclusions |

The release candidate remains unreleased until an explicit `v2.0.0rc3` tag is
created and the guarded PyPI workflow succeeds. No journal venue, permanent
data release, or corrected scientific result is implied by the package version.

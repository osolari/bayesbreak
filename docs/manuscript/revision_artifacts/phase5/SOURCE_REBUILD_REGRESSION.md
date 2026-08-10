# Phase 5 clean-source rebuild regression

## Scope

A clean copy of the Phase 5 unified Overleaf project was built in an empty workspace after removal of generated build products, caches, and font files. This check tests whether the distributed source reproduces the four compiled document targets and the Phase 5 structural checks. It is a build-regression check, not the independent scientific verification reserved for Phase 6.

## Commands

The clean source was rebuilt with the documented project targets:

```text
make paper-single
make executive
make handoff-check skeleton-check presentation-check
python3 scripts/validate_phase5.py
```

The technical-book and two-column-paper targets had already completed successfully in the same clean workspace during the initial all-target build. The initial all-target invocation reached the external execution-time limit while compiling the single-column paper; the remaining target-specific commands then completed without a LaTeX error.

## Results

| Target | Pages | Released PDF SHA-256 | Clean-build PDF SHA-256 | Rendered-page differences at 72 dpi |
|---|---:|---|---|---:|
| Technical book | 165 | `e218f99a61ee3d5bcadc40d9ecc0cbb3931be4e9fd8fcdf64f42f9c599a65b71` | `7d81d711425a4a195640179fabe147497b64959c72255a753cb527852dc62bd0` | 0 / 165 |
| Main paper, two-column | 35 | `b79ccc6aee46741662362b34d9a36c15d598bd24748bf1dc524d35f92ab3626d` | `2fc71ac498d08d913119cef7c4d865543217e9c4a12fbe56fa53225a6aaed4b8` | 0 / 35 |
| Main paper, single-column | 42 | `240c96a81fdbd96b9fc53cfebef18f07b7d626e1706fbe97a89c75d7de653f2e` | `b1e3c45a795afe475353e45963d8ab2c495087950890ce899869ffa0d2cc4a8d` | 0 / 42 |
| Executive summary | 12 | `c89c1d2ea924d111f18446cbde80f8396ca92f06c72a949578efafd9268a9a38` | `51a89564e56ef38cf0251294fb753510b88f6fcaddc5d7e7ab86a521a8a9496c` | 0 / 12 |

The PDF byte hashes differ because TeX writes build-dependent metadata. Each released and clean-built PDF was independently rasterized to grayscale at 72 dpi. The resulting page images were byte-identical on all 254 pages.

## Structural checks

The clean project passed:

- canonical handoff synchronization;
- repository-skeleton interface and explicit-incomplete-state checks;
- presentation-handoff completeness and roadmap-location checks;
- exact-title and terminology checks;
- proof-placement and bibliography-annotation checks;
- archived-result hash checks;
- no-slide-source checks.

## Conclusion

The distributed Phase 5 source reproduces the released page content for the technical book, both journal-paper layouts, and the executive summary. No source-rebuild discrepancy remains open.

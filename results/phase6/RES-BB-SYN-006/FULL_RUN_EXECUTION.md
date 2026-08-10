# RES-BB-SYN-006 full-run execution record

The explicitly approved corrected EPR-BB-015 suite executed from commit
`96464039e12b43207735835b004b0a59a9966b57`.

- 400 datasets ran across eight predeclared cells, with 50 datasets per cell.
- All 400 top-level cell records executed; no record was filtered or replaced.
- Elapsed wall time was 1,114.26 seconds and peak RSS was 169.66 MB.
- All 50 EP fits reached the predeclared 20-second fit-only timeout. Those outcomes remain
  timed out, with no imputed approximation diagnostics.
- `results.json` SHA-256 is
  `f0731f55c1682fd42e89b3ea0f67ec593c887cfbfeae3176fc8cc623e81f7024`.
- The aggregate input-identity SHA-256 is
  `106e7ea09e4b7c04d326895e6f17990627c721633ad7eed8471bfb936fcff59a`.

The result remains `pending-independent-review`. This execution record does not promote any
scientific conclusion, register a valid-for-interpretation result, or alter manuscript claims.
The exploratory notebook under `tutorials/11_result_provenance_explorer.ipynb` validates identity,
seeds, hashes, retained statuses, and plotting behavior without changing that review boundary.

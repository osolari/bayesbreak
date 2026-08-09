# RES-BB-SYN-006 EPR-BB-015 resource pilot

The authorized one-dataset-per-cell pilot executed from code commit
`88eb9daddc81249379635f031327fc2e39fb22d6`.

- Eight predeclared cells ran in 77.02 seconds with 157 MB peak RSS.
- The projected 50-repetition full suite is 3,851 seconds (64.2 minutes), above the
  predeclared 30-minute renewed-approval threshold.
- The null Gaussian cell selected four segments, producing three false boundaries.
- Heavy-tailed Gaussian data recovered both true boundaries with F1@3 = 1.00.
- Zero-inflated Poisson data saturated at `k=8` and produced F1@3 = 0.444.
- Dense Gaussian changes selected `k=15` with F1@3 = 0.880.
- The four-observation short segment selected `k=4` with F1@3 = 0.800.
- The prior-conflict cell returned an explicit failure (`No valid segment counts produced
  finite evidence`) because its minimum-length prior excluded every feasible truth-compatible
  partition. The failure remains in the record.
- Shared-boundary heterogeneity produced shared F1@3 = 0.444 versus mean independent F1@3 =
  0.628 and selected the subject-specific boundary near index 60 as shared.
- The logistic-normal reference selected `k=3`. EP consumed 75.73 seconds, had maximum reachable
  block error 6.575 log units, and empirical posterior TV 0.505. The lower-order quadrature and
  Laplace comparisons were fast but still had large maximum block errors.

The full suite has not been executed. It requires renewed approval of the projected resource cost
and review of whether to retain the current EP budget or predeclare a bounded timeout.

## Bounded EP re-pilot

The approved re-pilot executed from commit
`236ec8c509224688cc6e29d855a2df182a869ca7` with a predeclared 20-second EP
subprocess timeout.

- All eight cells completed in 21.28 seconds with 154 MB peak RSS.
- The projected 50-repetition full suite is 1,064 seconds (17.7 minutes), below the
  predeclared 30-minute renewed-approval threshold.
- The seven non-EP scientific outcomes and the quadrature/Laplace logistic diagnostics are
  byte-for-byte or field-for-field unchanged from the first pilot.
- EP timed out at 20.01 seconds and is retained as a timed-out scientific outcome; no EP error
  or posterior TV value is imputed for that run.
- The prior-conflict cell remains an explicit failed outcome.

The full suite remains unexecuted and requires separate explicit approval.

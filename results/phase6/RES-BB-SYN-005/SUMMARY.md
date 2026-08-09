# RES-BB-SYN-005 latent-group corrected rerun

Parent: `RES-BB-SYN-002`. Protocol: `EPR-BB-005`.
Code commit: `734ea3b241f0c0ae0ecbc30ad2ae144a2a2f3750`.

The archived-design cell used 50 seeded datasets with 24 sequences of length 80 at sigma=1.0. Mean hard accuracy was 0.974 (95% interval 0.954 to 0.995); mean ARI was 0.918.

All 400 objective traces were monotone, every returned final objective equaled the last trace value, and all 1,200 restarts were valid. Stress cells show expected failure behavior: low separation and duplicate templates do not support recovery claims, and overspecified groups increase collapse/redundancy.

This result supports the stated finite latent-group criterion in the declared synthetic design. It is not evidence for normalized finite-mixture identifiability or universal recovery.

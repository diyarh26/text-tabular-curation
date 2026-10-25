# Final LinkedIn Kaggle results

## LinkedIn salary: completed Kaggle evaluation

COMPLETE; 120/120 successful evaluations; metric r2; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 16, accumulation 16, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.492263 | 0.654557 | 0.691551 | 0.742389 | +0.036994 | +0.050838 |
| LightGBM | 0.480691 | 0.623402 | 0.662462 | 0.734879 | +0.039060 | +0.072417 |
| TabM | 0.467717 | 0.689403 | 0.719763 | 0.747477 | +0.030360 | +0.027714 |
| TabPFN-v2 | 0.496956 | 0.670713 | 0.697635 | 0.736120 | +0.026922 | +0.038485 |
| TabPFN-v2p5 | 0.504455 | 0.697526 | 0.722821 | 0.745030 | +0.025295 | +0.022208 |

Joint Signal 5/5; TAR Gain 5/5.
Passing-only means, independently by criterion: Joint +0.031726; TAR +0.042333.
All-five means: Joint +0.031726; TAR +0.042333.
Paper rule on these matched folds: 5/5 satisfy both rounded-condition-mean margins >=0.001.

All 120 expected evaluations passed the saved-result audit. All 115 previously completed run files and the experiment manifest are byte-identical to the earlier archive. Five fold-5 TAR evaluations have been added. This is physical batch 16 with 16-step accumulation, not the separate physical-256 VM experiment.

# FDA Food: completed six-fold evaluation

All 120 evaluations succeeded and passed the audit. Dataset: 7,918 rows; three severity classes.
Physical batch 64, four-step accumulation, effective batch 256; float32; official 50-epoch cap with early stopping.

Six-fold mean macro one-vs-rest AUROC; deltas computed from unrounded condition means.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint delta | TAR delta |
|---|---:|---:|---:|---:|---:|---:|
| TabM | 0.596236 | 0.826705 | 0.793869 | 0.824680 | -0.032836 | +0.030811 |
| CatBoost | 0.641255 | 0.818751 | 0.828252 | 0.840159 | +0.009501 | +0.011907 |
| LightGBM | 0.617900 | 0.814574 | 0.814254 | 0.837659 | -0.000320 | +0.023405 |
| TabPFN-v2 | 0.637524 | 0.836882 | 0.840462 | 0.846312 | +0.003581 | +0.005850 |
| TabPFN-v2p5 | 0.629681 | 0.844439 | 0.850983 | 0.858454 | +0.006543 | +0.007471 |

Course rule: Joint Signal 3/5, TAR Gain 5/5: PASSES.
Paper acceptance rule applied to six course folds: 3/5 learners meet both rounded-score margins >=0.001: PASSES.
The paper-style calculation uses rounded condition means before calculating deltas; it does not reproduce the paper's five-seed experiment.

CatBoost, TabPFN-v2 and TabPFN-v2.5 each have positive Joint Signal in all six folds. Every learner has positive TAR gain in all six folds.
Joint gains for the three qualifying learners remain below the user's preferred +0.01 AUROC: +0.009501, +0.003581, +0.006543.

Both AUROC and negative log loss are retained. Original reports predate the AUROC reporting correction and are preserved unchanged.
This analysis reran no training. Full Trainer checkpoints remain in the original downloaded ZIP.

Source archive SHA256: `d84776eada9e048ac134dad708eb6eb9117cff71f55a3b10fb8438e7c559abb9`.

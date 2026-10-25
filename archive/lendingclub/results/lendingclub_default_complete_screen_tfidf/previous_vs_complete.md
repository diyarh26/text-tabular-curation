# LendingClub complete-case CPU screen

All 18 new evaluations succeeded: CatBoost and LightGBM x three TF-IDF conditions x folds 0, 1, 2. CPU only; no E5 or TAR.

Both versions use 15,000 rows, the same chosen columns and the same pinned MulTaBench settings. The complete-case version was filtered on the full source before random sampling.

| Dataset | Learner | Structured | Text TF-IDF | Joint TF-IDF | Joint delta | Fold wins |
|---|---|---:|---:|---:|---:|---:|
| Previous sample | CatBoost | 0.6917 | 0.5687 | 0.6940 | +0.002359 | 2/3 |
| Previous sample | LightGBM | 0.6675 | 0.5538 | 0.6633 | -0.004241 | 1/3 |
| Complete-case sample | CatBoost | 0.6977 | 0.5513 | 0.6942 | -0.003476 | 1/3 |
| Complete-case sample | LightGBM | 0.6717 | 0.5350 | 0.6685 | -0.003201 | 1/3 |

The new sample has zero missing inputs. Both learners nevertheless have negative mean joint gains. Neither reaches the +0.01 joint AUROC goal.

The two versions contain different sampled loans, so this comparison does not isolate the causal effect of removing missing values.

Deltas use unrounded mean AUROC and compare joint against the stronger unimodal mean. Original data and earlier results remain unchanged.

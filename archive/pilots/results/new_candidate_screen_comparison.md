# New candidate CPU screening results

All 54 evaluations succeeded: 3 datasets x 2 learners x 3 conditions x 3 folds.

Actual pinned MulTaBench wrappers, preprocessing, hyperparameters and outer splits. Text uses TF-IDF; no E5 or TAR.

The joint delta is the unrounded mean joint score minus the better unrounded unimodal mean score. Fold wins require beating both unimodal scores within a fold.

## AUROC

Binary AUROC for LendingClub; macro one-vs-rest AUROC for the three-class FDA tasks. These measure different prediction problems, so raw gains are not a universal cross-dataset ranking.

| Dataset | Learner | Structured | Text TF-IDF | Joint TF-IDF | Joint delta | Fold wins |
|---|---|---:|---:|---:|---:|---:|
| LendingClub | CatBoost | 0.6917 | 0.5687 | 0.6940 | +0.002359 | 2/3 |
| LendingClub | LightGBM | 0.6675 | 0.5538 | 0.6633 | -0.004241 | 1/3 |
| FDA Device | CatBoost | 0.7463 | 0.8085 | 0.8363 | +0.027816 | 3/3 |
| FDA Device | LightGBM | 0.7262 | 0.8013 | 0.8224 | +0.021095 | 3/3 |
| FDA Food | CatBoost | 0.6424 | 0.8193 | 0.8296 | +0.010354 | 3/3 |
| FDA Food | LightGBM | 0.6166 | 0.8161 | 0.8187 | +0.002605 | 2/3 |

## FDA official multiclass metric: negative log loss

Higher is better. A positive joint delta is a reduction in log loss relative to the better unimodal model. Its scale is different from AUROC.

| Dataset | Learner | Structured | Text TF-IDF | Joint TF-IDF | Joint delta | Fold wins |
|---|---|---:|---:|---:|---:|---:|
| FDA Device | CatBoost | -0.3490 | -0.3222 | -0.3024 | +0.019867 | 3/3 |
| FDA Device | LightGBM | -0.3924 | -0.3672 | -0.3865 | -0.019251 | 0/3 |
| FDA Food | CatBoost | -0.8674 | -0.6760 | -0.6603 | +0.015705 | 3/3 |
| FDA Food | LightGBM | -0.9225 | -0.7059 | -0.7096 | -0.003680 | 0/3 |

## Prepared data and interpretation

- LendingClub: 15,000 resolved accepted loans with descriptions; 3 text and 10 structured features. The original missing-value rule and sample are retained.
- FDA Device: 15,000 events; FDA Food: 7,918 events. One randomly selected product per event, original three severity classes, 3 text and 8 structured features.
- Explicit FDA recall-severity text has been masked, and the audited pasted administrative template removed. Legitimate device regulatory classes remain.
- Seed 42; no class balancing, minimum word count or text truncation.
- This screen measures possible joint signal. It does not establish TAR gain or final acceptance with the five-learner GPU pipeline.

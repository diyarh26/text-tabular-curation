# All saved experiment results — 2026-10-24

This report reads existing JSON results only. No model was trained or rerun.
Classification curation uses AUROC (macro one-vs-rest for multiclass); regression uses R². Repository negative log loss/RMSE are diagnostic and are not substituted for those curation metrics.
Joint Δ = mean(joint frozen) − max(mean(structured), mean(text)). TAR Δ = mean(joint TAR) − mean(joint frozen). Decisions use unrounded values.
Course counts are independent: Joint >0 in >=3/5 learners and TAR >0 in >=3/5. The paper rule is shown separately using rounded condition means, >=0.001 and the same three learners.
All comparisons average the SAME fully completed folds across all four conditions. A partial fold's baselines are never compared with fewer TAR folds.
Passing-only means follow the user's requested definition: average each delta only over learners positive on that criterion. They exclude negative results; counts and all-five means are also shown so they are not mistaken for whole-pool averages.

## Project ranking

This is a project recommendation based on both criteria, gain sizes and completion. AUROC and R² are different metrics; this is not a numerical leaderboard mixing their scales.

| Rank | Dataset | Status | Joint/TAR learners | Passing-only Joint mean | Passing-only TAR mean |
|---|---|---|---:|---:|---:|
| 1 (provisional) | LinkedIn salary — provisional | INCOMPLETE | 5/5; 5/5 | +0.031991 | +0.041941 |
| 2; best completed | FDA Device — Azure VM, physical batch 256 | PASSES | 4/5; 4/5 | +0.012439 | +0.017156 |
| 3 | FDA Food — physical batch 256 | PASSES | 3/5; 5/5 | +0.006542 | +0.017050 |
| 4 | Steam recommendations — physical batch 256 | PASSES | 5/5; 5/5 | +0.004455 | +0.005190 |
| 5 | LendingClub long descriptions, purpose removed | FAILS | 4/5; 1/5 | +0.015776 | +0.001080 |
| 6; historical partial | LA Airbnb log price — historical partial evaluation | INCOMPLETE | 3/5; 2/5 | +0.005871 | +0.004127 |

Food batch 64 x 4 is the same prepared dataset under a separate execution protocol, not an additional candidate; it also passes. Food batch 256 is the reference-setting run used in the ranking.
LinkedIn is the leading provisional candidate: all five learners exceed +0.01 on both five-fold R² mean deltas. Finish the five missing fold-5 TAR evaluations before issuing a final verdict.
Device is the strongest completed option under the requested passing-only summary; its positive Joint/TAR means exceed +0.01, although fewer than three individual learners have Joint gains >=+0.01. Passing-only averages do not imply that every learner meets that target.

## LinkedIn salary — provisional

INCOMPLETE: matched completed folds only; 115/120 successful evaluations; metric r2; table averages folds [0, 1, 2, 3, 4].
Physical batch 16, accumulation 16, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.492761 | 0.655176 | 0.692700 | 0.742969 | +0.037524 | +0.050269 |
| LightGBM | 0.481569 | 0.622942 | 0.663772 | 0.734913 | +0.040830 | +0.071141 |
| TabM | 0.469058 | 0.690110 | 0.720467 | 0.747035 | +0.030356 | +0.026568 |
| TabPFN-v2 | 0.496282 | 0.670511 | 0.696866 | 0.735992 | +0.026356 | +0.039126 |
| TabPFN-v2p5 | 0.504104 | 0.697938 | 0.722829 | 0.745428 | +0.024891 | +0.022599 |

Joint Signal 5/5; TAR Gain 5/5.
Passing-only means, independently by criterion: Joint +0.031991; TAR +0.041941.
All-five means: Joint +0.031991; TAR +0.041941.
Paper rule on these matched folds: 5/5 satisfy both rounded-condition-mean margins >=0.001.
No final acceptance verdict. Partial next-fold baselines were excluded from all four condition means.
Fold 5 has its 15 baselines, but all five joint_tar evaluations are missing. The saved fold-5 Trainer checkpoint is at epoch 30. This archive is a snapshot, not proof of the live session's current state.

## FDA Food — physical batch 256

COMPLETE; 120/120 successful evaluations; metric auroc; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 256, accumulation 1, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.641255 | 0.818751 | 0.828252 | 0.839561 | +0.009501 | +0.011309 |
| LightGBM | 0.617900 | 0.814574 | 0.814254 | 0.839194 | -0.000320 | +0.024940 |
| TabM | 0.596236 | 0.826705 | 0.793869 | 0.828256 | -0.032836 | +0.034387 |
| TabPFN-v2 | 0.637524 | 0.836882 | 0.840462 | 0.847428 | +0.003581 | +0.006966 |
| TabPFN-v2p5 | 0.629681 | 0.844439 | 0.850983 | 0.858630 | +0.006543 | +0.007647 |

Joint Signal 3/5; TAR Gain 5/5.
Passing-only means, independently by criterion: Joint +0.006542; TAR +0.017050.
All-five means: Joint -0.002706; TAR +0.017050.
Paper rule on these matched folds: 3/5 satisfy both rounded-condition-mean margins >=0.001.

## FDA Food — batch 64 x 4

COMPLETE; 120/120 successful evaluations; metric auroc; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 64, accumulation 4, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.641255 | 0.818751 | 0.828252 | 0.840159 | +0.009501 | +0.011907 |
| LightGBM | 0.617900 | 0.814574 | 0.814254 | 0.837659 | -0.000320 | +0.023405 |
| TabM | 0.596236 | 0.826705 | 0.793869 | 0.824680 | -0.032836 | +0.030811 |
| TabPFN-v2 | 0.637524 | 0.836882 | 0.840462 | 0.846312 | +0.003581 | +0.005850 |
| TabPFN-v2p5 | 0.629681 | 0.844439 | 0.850983 | 0.858454 | +0.006543 | +0.007471 |

Joint Signal 3/5; TAR Gain 5/5.
Passing-only means, independently by criterion: Joint +0.006542; TAR +0.015889.
All-five means: Joint -0.002706; TAR +0.015889.
Paper rule on these matched folds: 3/5 satisfy both rounded-condition-mean margins >=0.001.

## LendingClub long descriptions, purpose removed

COMPLETE; 120/120 successful evaluations; metric auroc; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 64, accumulation 4, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.692850 | 0.602012 | 0.705545 | 0.706626 | +0.012696 | +0.001080 |
| LightGBM | 0.668879 | 0.576649 | 0.684176 | 0.680838 | +0.015297 | -0.003338 |
| TabM | 0.693417 | 0.561383 | 0.684301 | 0.673401 | -0.009115 | -0.010901 |
| TabPFN-v2 | 0.695522 | 0.609975 | 0.712864 | 0.712192 | +0.017342 | -0.000672 |
| TabPFN-v2p5 | 0.696468 | 0.620285 | 0.714237 | 0.713705 | +0.017769 | -0.000532 |

Joint Signal 4/5; TAR Gain 1/5.
Passing-only means, independently by criterion: Joint +0.015776; TAR +0.001080.
All-five means: Joint +0.010798; TAR -0.002872.
Paper rule on these matched folds: 1/5 satisfy both rounded-condition-mean margins >=0.001.

## FDA Device — Azure VM, physical batch 256

COMPLETE; 120/120 successful evaluations; metric auroc; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 256, accumulation 1, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.744903 | 0.818817 | 0.838227 | 0.838142 | +0.019411 | -0.000086 |
| LightGBM | 0.726180 | 0.807479 | 0.821299 | 0.833134 | +0.013820 | +0.011835 |
| TabM | 0.686019 | 0.829348 | 0.747631 | 0.785585 | -0.081717 | +0.037954 |
| TabPFN-v2 | 0.739422 | 0.813963 | 0.822052 | 0.838755 | +0.008088 | +0.016703 |
| TabPFN-v2p5 | 0.745958 | 0.838396 | 0.846834 | 0.848967 | +0.008438 | +0.002133 |

Joint Signal 4/5; TAR Gain 4/5.
Passing-only means, independently by criterion: Joint +0.012439; TAR +0.017156.
All-five means: Joint -0.006392; TAR +0.013708.
Paper rule on these matched folds: 3/5 satisfy both rounded-condition-mean margins >=0.001.
The original GitHub final audit declares failure using negative log loss; the AUROC curation calculation above passes. Original files are preserved unchanged.

## Steam recommendations — physical batch 256

COMPLETE; 120/120 successful evaluations; metric auroc; table averages folds [0, 1, 2, 3, 4, 5].
Physical batch 256, accumulation 1, effective batch 256.

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.812060 | 0.953720 | 0.962941 | 0.967664 | +0.009221 | +0.004723 |
| LightGBM | 0.800669 | 0.951757 | 0.958121 | 0.965342 | +0.006364 | +0.007220 |
| TabM | 0.805882 | 0.958497 | 0.958608 | 0.965264 | +0.000111 | +0.006656 |
| TabPFN-v2 | 0.814306 | 0.957452 | 0.960643 | 0.965343 | +0.003191 | +0.004700 |
| TabPFN-v2p5 | 0.804657 | 0.960660 | 0.964046 | 0.966695 | +0.003386 | +0.002649 |

Joint Signal 5/5; TAR Gain 5/5.
Passing-only means, independently by criterion: Joint +0.004455; TAR +0.005190.
All-five means: Joint +0.004455; TAR +0.005190.
Paper rule on these matched folds: 5/5 satisfy both rounded-condition-mean margins >=0.001.

## LA Airbnb log price — historical partial evaluation

INCOMPLETE: matched completed folds only; 55/120 successful evaluations; metric r2; table averages folds [0, 1].

| Learner | Structured | Text only | Joint frozen | Joint TAR | Joint Δ | TAR Δ |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 0.839040 | 0.566431 | 0.837795 | 0.840570 | -0.001245 | +0.002776 |
| LightGBM | 0.827712 | 0.531908 | 0.826168 | 0.831647 | -0.001544 | +0.005478 |
| TabM | 0.833906 | 0.610739 | 0.843298 | 0.841849 | +0.009392 | -0.001450 |
| TabPFN-v2 | 0.838883 | 0.589555 | 0.841849 | 0.839198 | +0.002966 | -0.002652 |
| TabPFN-v2p5 | 0.849740 | 0.617358 | 0.854995 | 0.852251 | +0.005255 | -0.002744 |

Joint Signal 3/5; TAR Gain 2/5.
Passing-only means, independently by criterion: Joint +0.005871; TAR +0.004127.
All-five means: Joint +0.002965; TAR +0.000282.
Paper rule on these matched folds: 0/5 satisfy both rounded-condition-mean margins >=0.001.
No final acceptance verdict. Partial next-fold baselines were excluded from all four condition means.
Historical stopped experiment; not a completed six-fold candidate evaluation.


## Earlier CPU TF-IDF screens

Each screen has two learners, three conditions and three folds (18 runs), with no TAR evaluation. These are screening indicators, not final acceptance evidence.

### fda_device_severity_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.746273 | 0.808526 | 0.836341 | +0.027816 |
| LightGBM | 0.726211 | 0.801272 | 0.822367 | +0.021095 |

### fda_food_severity_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.642358 | 0.819270 | 0.829624 | +0.010354 |
| LightGBM | 0.616579 | 0.816114 | 0.818719 | +0.002605 |

### glassdoor_outlook_B_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.650307 | 0.726930 | 0.747400 | +0.020469 |
| LightGBM | 0.628800 | 0.721828 | 0.736583 | +0.014755 |

### glassdoor_outlook_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.846522 | 0.726930 | 0.846150 | -0.000372 |
| LightGBM | 0.829093 | 0.721828 | 0.834638 | +0.005545 |

### lendingclub_default_complete_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.697719 | 0.551278 | 0.694244 | -0.003476 |
| LightGBM | 0.671697 | 0.534963 | 0.668496 | -0.003201 |

### lendingclub_default_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.691672 | 0.568661 | 0.694031 | +0.002359 |
| LightGBM | 0.667501 | 0.553773 | 0.663260 | -0.004241 |

### lendingclub_longdesc_no_purpose_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.694668 | 0.596407 | 0.704795 | +0.010127 |
| LightGBM | 0.672577 | 0.588620 | 0.682120 | +0.009543 |

### linkedin_salary_screen_tfidf

Metric: r2

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.493807 | 0.641941 | 0.694118 | +0.052177 |
| LightGBM | 0.482774 | 0.609735 | 0.660394 | +0.050659 |

### steam_recommended_screen_tfidf

Metric: auroc

| Learner | Structured | Text only | Joint frozen | Joint Δ |
|---|---:|---:|---:|---:|
| CatBoost | 0.812167 | 0.834491 | 0.893311 | +0.058820 |
| LightGBM | 0.802507 | 0.829982 | 0.893213 | +0.063231 |

## Earlier Colab TAR pilots

One fold, CatBoost and LightGBM, ten-epoch pilot cap, TAR only. The historical microbatch-loss implementation predates the corrected full-run loss normalization. Do not use these pilots for a final verdict or compare their TAR scores with TF-IDF to claim isolated TAR gain.

| Dataset | Learner | Metric | Pilot joint TAR |
|---|---|---|---:|
| linkedin_salary | CatBoost | r2 | 0.715992 |
| linkedin_salary | LightGBM | r2 | 0.696862 |
| steam_recommended | CatBoost | auroc | 0.968514 |
| steam_recommended | LightGBM | auroc | 0.965765 |

Glassdoor has CPU screens only. No completed MulTaBench GPU evaluation was found for Indiegogo or FDA Drug in these sources; they are not assigned a final-result rank.
The failed Kaggle Device-256 attempt is not an extra completed experiment. Its OOM does not alter the independent completed Azure VM Device results.

## Audit and sources

Every imported successful run was checked for dataset, model, condition, fold, encoder, pinned MulTaBench version, train/test sizes, finite metric and duplicate identity. Available manifest data/config hashes, source-code hashes and saved split hashes were verified. Downloaded model/optimizer weight files were not extracted or executed.
Device was read from GitHub origin/main; its 120 run-file hashes also match the uploaded final audit. Other originals remain in Downloads/local results; imported small artifacts are preserved under sources/.
Legacy Airbnb has no experiment manifest; its provenance evidence is weaker than the modern full runs. Its two-fold table is diagnostic only.


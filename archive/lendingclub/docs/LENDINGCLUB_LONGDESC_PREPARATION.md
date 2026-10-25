# LendingClub: longer descriptions, no purpose

All 18 new CPU evaluations succeeded: CatBoost and LightGBM x structured/text/joint TF-IDF x folds 0, 1, 2.

The full verified complete-case pool contained 115,099 loans. Requiring strictly more than 50 description content words left 26,061 loans, before the seed-42 random sample of 15,000. Date prefixes are excluded from counting and retained in model text.

The purpose column is omitted from both the model CSV and config. Text columns: desc, title, emp_title. Structured columns: term, emp_length, home_ownership, grade, loan_amnt, annual_inc, dti, fico_score, int_rate.

The sample has no missing values; 12,822 Fully Paid (85.48%) and 2,178 Charged Off (14.52%). Description content length: minimum 51, median 68 words.

| Dataset | Learner | Structured | Text TF-IDF | Joint TF-IDF | Joint delta | Fold wins |
|---|---|---:|---:|---:|---:|---:|
| Original sample | CatBoost | 0.6917 | 0.5687 | 0.6940 | +0.002359 | 2/3 |
| Original sample | LightGBM | 0.6675 | 0.5538 | 0.6633 | -0.004241 | 1/3 |
| Complete inputs | CatBoost | 0.6977 | 0.5513 | 0.6942 | -0.003476 | 1/3 |
| Complete inputs | LightGBM | 0.6717 | 0.5350 | 0.6685 | -0.003201 | 1/3 |
| >50 words; no purpose | CatBoost | 0.6947 | 0.5964 | 0.7048 | +0.010127 | 3/3 |
| >50 words; no purpose | LightGBM | 0.6726 | 0.5886 | 0.6821 | +0.009543 | 3/3 |

Both learners show positive joint gains in all three folds. CatBoost exceeds +0.01 AUROC by a small margin; LightGBM remains slightly below +0.01.

The versions differ in sampled loans and eligibility. The latest version also removes purpose. This comparison evaluates the combined changes and does not isolate their individual effects.

These are TF-IDF screening results. Frozen E5 and TAR have not been tested on this version. Earlier datasets and results are unchanged.

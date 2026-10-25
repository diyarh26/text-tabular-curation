# Candidate dataset screening comparison

CPU TF-IDF screen: CatBoost and LightGBM; structured only, text only, and joint; three outer folds, each with 10,000 training and 5,000 test rows. These are screening results, not frozen-E5 or TAR results.

| Dataset | Learner | Metric (higher is better) | Structured | Text only | Joint | Joint minus best unimodal | Fold wins |
|---|---|---|---:|---:|---:|---:|---:|
| LinkedIn salary | CatBoost | R2 | 0.4938 | 0.6419 | 0.6941 | +0.0522 | 3/3 |
| LinkedIn salary | LightGBM | R2 | 0.4828 | 0.6097 | 0.6604 | +0.0507 | 3/3 |
| Glassdoor outlook B | CatBoost | negative log loss | -0.9274 | -0.8459 | -0.8241 | +0.0218 | 3/3 |
| Glassdoor outlook B | LightGBM | negative log loss | -0.9731 | -0.8566 | -0.8492 | +0.0074 | 3/3 |
| Steam recommendations | CatBoost | AUROC | 0.8122 | 0.8345 | 0.8933 | +0.0588 | 3/3 |
| Steam recommendations | LightGBM | AUROC | 0.8025 | 0.8300 | 0.8932 | +0.0632 | 3/3 |

Joint gains are measured against the stronger unimodal baseline. The metric scales differ across tasks, so raw gains should not be ranked directly across datasets. All three candidates show positive mean joint signal for both screened learners. None has established TAR gain in this screen; full acceptance requires both criteria for at least three of five learners.

## Steam sample and provenance

- Exactly 15,000 reviews: 13,405 recommended and 1,595 not recommended.
- Reviews come from 123 games. Largest game: 793 rows (5.3%); largest two games together: 9.6%.
- No minimum review word count, no game quotas, and no target-stratified sampling. Only English-labelled, nonempty reviews with a target are eligible; review IDs are deduplicated.
- Acquisition uses 400 uniformly sampled 100-row blocks across the 21,747,371-row source (40,000 candidate rows); 17,326 eligible English reviews form the pool. A simple random sample of 15,000 pool rows uses seed 42. This is two-stage block sampling, rather than a direct simple random sample of all source reviews.
- The existing rare-category rule combines games with fewer than 20 sampled rows into Other, leaving 121 model categories. This changes categories, not row selection.
- Text: review. Structured: game, review year, playtime at review in hours, games owned, reviews written, Steam purchase, free copy, and early access.
- Games owned and reviews written are retained by project decision; their values are not verified snapshots at review time.
- Duplicate review IDs are removed; repeated review text is retained. The sample has 13,325 unique review texts. Common short phrases can occur in multiple records.
- Per-block downloads, source metadata, the candidate Parquet file, and selected review IDs are cached under data/raw/steam/. Summary and CSV checksum: ../data/processed/steam_recommended.summary.json.

## Saved reports

- [LinkedIn salary](linkedin_salary_screen_tfidf/evaluation_report.csv)
- [Glassdoor outlook B](glassdoor_outlook_B_screen_tfidf/evaluation_report.csv)
- [Steam recommendations](steam_recommended_screen_tfidf/evaluation_report.csv)

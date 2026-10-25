# Candidate preparation and completed CPU screen

The user approved the CPU screen. All 54 evaluations have completed successfully. See [the comparison](../results/new_candidate_screen_comparison.md).

Seed 42. Keep between 1,000 and 15,000 rows per dataset. No class balancing, minimum word count, or text truncation.

The original downloaded archives are unchanged. Identifiers and source row positions are stored separately under data/raw/candidate_screens/, outside the model CSVs.

## Prepared samples

| Dataset | Rows | Classes | Text columns | Structured columns |
|---|---:|---|---:|---:|
| lendingclub_default | 15,000 | 0: 12,689, 1: 2,311 | 3 | 10 |
| fda_device_severity | 15,000 | Class II: 13,426, Class I: 988, Class III: 586 | 3 | 8 |
| fda_food_severity | 7,918 | Class II: 4,065, Class I: 3,182, Class III: 671 | 3 | 8 |

## lendingclub_default

- Target: 0 = Fully Paid; 1 = Charged Off. All other loan statuses excluded.
- Text: desc, title, emp_title.
- Categorical: term, emp_length, home_ownership, purpose, grade.
- Numerical: loan_amnt, annual_inc, dti, fico_score, int_rate.
- Selection: Accepted-loan table only, resolved loans with a nonempty cleaned desc, simple random sample up to 15000.
- Counts: `{"eligible_rows_after_text_cleanup": 123292, "eligible_rows_removed_by_row_cap": 108292, "footer_rows_missing_loan_status": 33, "resolved_paid_or_charged_off_rows": 1345310, "resolved_rows_with_nonempty_raw_description": 123292, "selected_rows": 15000, "source_csv_rows": 2260701}`.
- Constant features: none.
- Duplicate complete feature rows: 0.
- Duplicate combined text rows: 0.
- Remaining class-phrase field values: 0; device regulatory class is different from recall severity.
- FDA leakage cleanup: 0 rows edited; row selection and labels unchanged.
- CSV SHA256: `c782b1664cd4a91d50f77b463ec10bf4947e27b2b9b33e97b731d87810a0211b`.

| Text | Nonempty | Distinct | Median words | Maximum words |
|---|---:|---:|---:|---:|
| desc | 15,000 | 14,967 | 29 | 766 |
| title | 14,999 | 5,635 | 2 | 16 |
| emp_title | 13,981 | 11,681 | 2 | 11 |

Missing values by column:

title: 1, emp_title: 1019, emp_length: 610

## fda_device_severity

- Target: Original Class I / Class II / Class III, without merging or ordering as a numeric target.
- Text: product_description, reason_for_recall, distribution_pattern.
- Categorical: recalling_firm, city, state, country, voluntary_mandated, initial_firm_notification.
- Numerical: recall_year, recall_month.
- Selection: Exclude unknown severities; uniform random product within each event; then random events up to 15000.
- Counts: `{"eligible_events": 15107, "eligible_product_rows": 40299, "events_removed_by_row_cap": 107, "events_with_multiple_products": 4543, "events_with_multiple_severities": 5, "excluded_unknown_severity": 1, "missing_or_invalid_initiation_date_in_event_pool": 0, "product_rows_removed_by_one_per_event": 25192, "selected_rows": 15000, "source_rows": 40300}`.
- Constant features: none.
- Duplicate complete feature rows: 0.
- Duplicate combined text rows: 1.
- Remaining class-phrase field values: 62; device regulatory class is different from recall severity.
- FDA leakage cleanup: 12 rows edited; row selection and labels unchanged.
- CSV SHA256: `b005b5ecf1bd1026c649f5989861cc6b8d3338ac1d09e67fefdc59c3793a6ec7`.

| Text | Nonempty | Distinct | Median words | Maximum words |
|---|---:|---:|---:|---:|
| product_description | 15,000 | 14,921 | 27 | 734 |
| reason_for_recall | 14,990 | 14,862 | 27 | 331 |
| distribution_pattern | 14,999 | 12,509 | 17 | 471 |

Missing values by column:

reason_for_recall: 10, distribution_pattern: 1, city: 2, state: 1689, country: 3, voluntary_mandated: 8, initial_firm_notification: 360

## fda_food_severity

- Target: Original Class I / Class II / Class III, without merging or ordering as a numeric target.
- Text: product_description, reason_for_recall, distribution_pattern.
- Categorical: recalling_firm, city, state, country, voluntary_mandated, initial_firm_notification.
- Numerical: recall_year, recall_month.
- Selection: Exclude unknown severities; uniform random product within each event; then random events up to 15000.
- Counts: `{"eligible_events": 7918, "eligible_product_rows": 29503, "events_removed_by_row_cap": 0, "events_with_multiple_products": 2784, "events_with_multiple_severities": 177, "excluded_unknown_severity": 2, "missing_or_invalid_initiation_date_in_event_pool": 1, "product_rows_removed_by_one_per_event": 21585, "selected_rows": 7918, "source_rows": 29505}`.
- Constant features: none.
- Duplicate complete feature rows: 0.
- Duplicate combined text rows: 1.
- Remaining class-phrase field values: 0; device regulatory class is different from recall severity.
- FDA leakage cleanup: 1 rows edited; row selection and labels unchanged.
- CSV SHA256: `0a8ff4ce3ff48fd2beb16edb9ff197b4f38fc026a87cc82308be220baaee539d`.

| Text | Nonempty | Distinct | Median words | Maximum words |
|---|---:|---:|---:|---:|
| product_description | 7,918 | 7,900 | 22 | 672 |
| reason_for_recall | 7,918 | 6,854 | 12 | 165 |
| distribution_pattern | 7,918 | 6,274 | 7 | 183 |

Missing values by column:

state: 150, voluntary_mandated: 6, initial_firm_notification: 6, recall_year: 1, recall_month: 1

## Approved FDA text cleanup

Explicit recall-severity references have been masked with the same [recall severity] placeholder, while retaining legitimate device regulatory class, laser class, and clinical class descriptions.
The audited pasted FDA administrative review template has been removed. Row selection, labels, class proportions, and structured fields are unchanged.
Before/after edits and trace identifiers are recorded under data/raw/candidate_screens/. No models have been fitted.

## Approved screen (completed)

- CPU only: CatBoost and LightGBM, the existing pinned MulTaBench wrappers and hyperparameters.
- Three conditions: structured, unstructured with TF-IDF, joint with TF-IDF.
- Three official outer folds (0, 1, 2); 18 evaluations per dataset, 54 total.
- LendingClub: AUROC. FDA: official negative log loss plus macro one-vs-rest AUROC.
- Joint delta is joint minus the better unimodal condition, computed separately for each metric.
- This cannot establish TAR gain or final dataset acceptance; no E5 or GPU run is included.
- Do not compare a log-loss delta numerically against an AUROC delta.

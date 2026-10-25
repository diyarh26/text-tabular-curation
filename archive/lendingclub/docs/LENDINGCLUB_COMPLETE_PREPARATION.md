# Complete-case LendingClub preparation

A separate dataset. Filtering was applied to the full accepted-loan source before sampling. The user subsequently approved the CPU screen; all 18 evaluations completed successfully. See [the old/new comparison](../results/lendingclub_default_complete_screen_tfidf/previous_vs_complete.md).

## Filters, in order

1. Read the accepted-loan table once; its compressed duplicate and the rejected-applicant table are not added.
2. Keep only Fully Paid and Charged Off loans. Exclude CSV footer rows and all other statuses.
3. Remove HTML markup, decode entities and normalize text whitespace. Require a nonempty description.
4. Derive origination FICO as the mean of fico_range_low/high and parse numeric inputs before checking completeness.
5. Require every one of the 13 chosen inputs and the target to be nonmissing; numeric values must be finite. Retain the valid home_ownership code NONE.
6. Randomly select 15,000 eligible loans without replacement, seed 42. No class balancing or minimum word count.

Completeness applies to the chosen 3 text and 10 structured inputs, not the 151 original columns. No values are imputed.

Text: desc, title, emp_title.

Categorical: term, emp_length, home_ownership, purpose, grade.

Numerical: loan_amnt, annual_inc, dti, fico_score, int_rate.

## Original accepted loans: all original statuses

The denominator excludes the 33 nondata CSV footer rows. Rejected applications have no repayment outcome and are a separate table.

| Status | Count | Percentage of original accepted loans |
|---|---:|---:|
| Fully Paid | 1,076,751 | 47.6298% |
| Current | 878,317 | 38.8521% |
| Charged Off | 268,559 | 11.8796% |
| Late (31-120 days) | 21,467 | 0.9496% |
| In Grace Period | 8,436 | 0.3732% |
| Late (16-30 days) | 4,349 | 0.1924% |
| Does not meet the credit policy. Status:Fully Paid | 1,988 | 0.0879% |
| Does not meet the credit policy. Status:Charged Off | 761 | 0.0337% |
| Default | 40 | 0.0018% |
| **Total** | **2,260,668** | **100%** |

## Binary outcome comparison

The original resolved-loan row below uses only the same two target classes, before description/completeness filters. Original all-status percentages above use a different denominator.

| Stage | Rows | Fully Paid count | Fully Paid % | Charged Off count | Charged Off % |
|---|---:|---:|---:|---:|---:|
| Original resolved loans: before feature filters | 1,345,310 | 1,076,751 | 80.0374% | 268,559 | 19.9626% |
| Resolved loans with a description | 123,292 | 104,389 | 84.6681% | 18,903 | 15.3319% |
| Full clean pool: all chosen inputs complete | 115,099 | 97,789 | 84.9608% | 17,310 | 15.0392% |
| NEW complete-case random 15K sample | 15,000 | 12,737 | 84.9133% | 2,263 | 15.0867% |
| Previous 15K sample: missing values allowed | 15,000 | 12,689 | 84.5933% | 2,311 | 15.4067% |

## Checks and outputs

- The full cleaned pool and the new 15K sample have zero missing values in every chosen input and target.
- Selected loan IDs are unique; identifiers are stored separately and excluded from model inputs.
- The new sample contains loans drawn from the full clean pool, not just the previous sample's complete rows.
- The previous LendingClub dataset and its model results are byte-for-byte unchanged.
- New sample: `data\processed\lendingclub_default_complete.csv`.
- Full clean pool: `data\raw\candidate_screens\lendingclub_default_complete_pool.csv` (local, ignored by Git).
- Config: `configs/lendingclub_default_complete.yaml`.
- Sample SHA256: `c0e8baff542f6dd401590ac22b7549293113f8f35eab774d86678a41077cc766`.

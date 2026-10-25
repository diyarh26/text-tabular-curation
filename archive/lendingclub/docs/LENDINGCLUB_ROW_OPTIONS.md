# LendingClub row-selection options

Update: the user subsequently requested a separate complete-case version. It was created by filtering
the full source before sampling; see [the complete-case preparation](LENDINGCLUB_COMPLETE_PREPARATION.md).
The earlier sample and its completed model results remain available unchanged.

Row selection remains unchanged. The approved CPU screen completed using option A and the original prepared sample.

## Missing values

The current 15,000-row sample has descriptions on every row. Missing values occur in:

| Selected field | Missing rows |
|---|---:|
| title | 1 |
| emp_title | 1,019 |
| emp_length | 610 |

These overlap: 1,027 rows have at least one missing chosen input; 13,973 are complete.
All other chosen inputs and the target are filled.

| Option | Exact meaning | Rows | Trade-off |
|---|---|---:|---|
| A. Keep the current sample | Require desc, permit missing ancillary fields; use the official learner preprocessing. | 15,000 | Retains the current borrower sample, including incomplete records. |
| B. Remove incomplete rows from the current sample | Require all 13 chosen inputs to be filled in the already selected loans. | 13,973 | Complete records, with a smaller sample. |
| C. Sample again from complete source records | First require all 13 inputs in the full eligible source pool, then randomly select 15,000. | 15,000 | Complete records and the full sample size; changes the sampled borrowers and excludes incomplete records. |

Read-only counts of the full source: 123,292 resolved loans have a nonempty cleaned description;
115,099 have every chosen input filled. The complete pool contains 97,789 Fully Paid and 17,310 Charged Off loans.
No target balancing is proposed for any option. Completeness concerns the 13 selected inputs, not all 151 original columns.

Recommendation for the initial screen: A, since the official pipeline supports missing values and
complete-case filtering changes the population. If complete inputs are a firm requirement, C
is feasible with ample eligible source records. The user's latest selection was to retain the current samples and rules.

## Description quality is a separate choice

The existing rule is nonempty description, with no minimum length or manual quality score.
After removing date prefixes only for this inspection (the prepared text is unchanged):

- Median description length: 23 words (29 including the date prefixes).
- 1,077 descriptions have at most 3 words.
- 3,892 descriptions have at most 10 words, including the preceding group.
- None become empty merely from removing the date prefixes.

Examples of short content include pool, furniture, consolidation, and thanks.
Nonempty does not guarantee a detailed explanation. No quality or length filter has been applied.

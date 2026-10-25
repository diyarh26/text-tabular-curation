# LinkedIn salary dataset

Documentation checked against `prep/linkedin.py`, `configs/linkedin_salary.yaml`, and the prepared CSV on 2026-10-25. Final batch-256 evaluation is still in progress.

## Task and source

Predict `log_salary`, the natural logarithm of the source's annualized USD `normalized_salary`, from a job posting's text and structured metadata. The task predicts advertised pay, not actual employee compensation.

Source: [LinkedIn Job Postings (2023-2024), published by arshkon on Kaggle](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings). The preparation script describes this snapshot as US postings from March-April 2024. Kaggle's public dataset listing identifies the source license as **CC BY-SA 4.0** (checked 2026-10-25). The source dataset's terms and attribution apply to this cleaned derivative; the repository's MIT code license does not relicense the data.

Prepared file: `data/processed/linkedin_salary.csv`, 15,000 rows, 14 input columns and one target column. File size in the checked Windows working copy: 45,339,719 bytes. Hashes use LF-normalized bytes to account for Git's Windows line endings:

- CSV: `8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa`.
- YAML config: `859863d245ed81b761d27682bf4b1657e5687b8a60f87d22dbe26e87c8cbc2ce`.

## Columns

| Column | Role / type | Meaning | Missing rows |
|---|---|---|---:|
| `title` | Text | Job title, with matching money/contact patterns masked | 0 |
| `description` | Text | Job description, masked and limited to first 500 words | 0 |
| `experience_level` | Categorical | Source's formatted experience level | 3,365 |
| `work_type` | Categorical | Source's formatted employment type | 0 |
| `state` | Categorical | Two-letter state code extracted from the end of location; rare values merged | 2,706 |
| `application_type` | Categorical | Application channel/type | 0 |
| `company_industry` | Categorical | First listed company industry; rare values merged | 231 |
| `job_industry` | Categorical | First mapped job industry; rare values merged | 157 |
| `remote_allowed` | Numeric / flag | Source remote flag; missing source flag filled with zero | 0 |
| `company_size` | Numeric | Source's numeric company-size code, not an exact headcount | 1,000 |
| `company_employees` | Numeric | Latest available employee-count record for the company in the source snapshot | 210 |
| `company_followers` | Numeric | Follower count from the same latest available company record | 210 |
| `n_skills` | Numeric | Number of linked job-skill records; absent records mapped to zero | 0 |
| `n_benefits` | Numeric | Number of linked benefit records; absent records mapped to zero | 0 |
| `log_salary` | Target / numeric | Natural log of annualized USD salary | 0 |

There are six categorical and six numerical structured inputs. Missing structured values are retained and handled by each official learner's native preprocessing. This is not a complete-case dataset.

## Preparation and excluded information

Run `python prep/linkedin.py` with the raw source files under `data/raw/linkedin/`:

```text
postings.csv
companies/companies.csv
companies/employee_counts.csv
companies/company_industries.csv
mappings/industries.csv
jobs/job_industries.csv
jobs/job_skills.csv
jobs/benefits.csv
```

The preparation procedure:

1. Keep source postings with a nonmissing normalized salary and currency USD.
2. Keep annualized salary between USD 15,000 and USD 1,000,000.
3. Join the selected company and job metadata; derive state and skill/benefit counts.
4. Mask recognized money expressions with `[amount]` in both text fields. Mask matching phone/email expressions, then collapse whitespace.
5. Truncate each cleaned description to its first 500 whitespace-separated words.
6. Drop rows with empty title or description and duplicate cleaned/truncated descriptions.
7. Sample at most 15,000 eligible rows randomly with seed 42, preserving the resulting order for evaluation.
8. In the selected sample, merge categories occurring fewer than 20 times into `Other` for state, company industry and job industry.

Only the declared 14 inputs and log target are exported. Source salary amounts, salary ranges, currency/pay-period fields, IDs, company names, URLs, raw locations, source timestamps, views and application counts are not exported as input features. The raw annualized salary is used to construct the target and is not an additional feature.

Observed text properties in the prepared CSV:

| Property | Title | Description |
|---|---:|---:|
| Median words | 3 | 483 |
| Maximum words | 29 | 500 |
| Unique strings | 11,591 | 15,000 |
| Rows containing `[amount]` | 389 | 9,504 |

E5 truncates at **512 tokens**, not a fixed word count. There is no English-language detector or HTML-removal step in this script. Do not describe those as implemented cleaning steps.

Target salary median is USD 83,750; the 5th-95th percentile range is USD 35,360-200,000. R-squared is evaluated on log salary, not salary in raw dollars.

## Evaluation and limitations

Use the exact sample order and six official splits, with 10,000 outer training rows and 5,000 outer test rows. All supervised encoder tuning is confined to each outer training partition. Native learner validation and text projection remain inside the official model wrappers. Preserve the distinct physical-batch-256 and 16x16 experiment identities.

Limitations to discuss in the paper:

- Pattern-based money masking reduces direct target leakage but does not prove every pay hint was removed. Titles, seniority and location deliberately remain predictive.
- The exact-description deduplication does not establish company-disjoint, time-disjoint, or near-duplicate-free evaluation.
- Company employee/follower records use the latest available snapshot; the code does not prove those records predate each posting. The task is snapshot prediction, not a validated historical deployment simulation.
- Missing state does not necessarily mean a bad row; the location extraction also misses postings without a final two-letter state code.
- The USD and salary-availability filters, deduplication and sample limit define a selected subset of job postings; results need not generalize to all jobs or realized compensation.
- Final six-fold batch-256 results and native MulTaBench integration are pending; see the compliance audit.

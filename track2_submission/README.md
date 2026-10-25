# LinkedIn Salary — Track 2

Predict log annual salary from 15,000 job postings with two text and twelve
structured inputs. Source: [arshkon / LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings), CC BY-SA 4.0.

The submission has two Python files:

- `src/preprocess.py`: original cleaning, joins, salary/contact masking,
  deduplication, 500-word description limit and seed-42 sampling.
- `src/evaluate.py`: native MulTaBench dataset registration, five learners ×
  four conditions × six splits (120 evaluations), printed fold/mean results,
  deltas and `evaluation_results.csv`.

## Reproduction

Use Linux, a Python 3.11 environment, Git and a CUDA GPU with sufficient memory for
physical batch 256. From this directory:

```bash
python src/evaluate.py --install
# The cleaned data/dataset.csv is supplied. Recreate it only if desired:
python src/preprocess.py --raw-dir /path/to/linkedin-job-postings
python src/evaluate.py
```

The raw source directory must contain `postings.csv`, `companies/companies.csv`,
`companies/employee_counts.csv`, `companies/company_industries.csv`,
`mappings/industries.csv`, `jobs/job_industries.csv`, `jobs/job_skills.csv` and
`jobs/benefits.csv`.

The evaluation script obtains official MulTaBench source at master-history
commit `ff0c9c8b40a2085a06a7771190e53a261341802f`. Its visible dataset-only
registration patch is included in the script. It runs unchanged official
embedding and `benchmark.py` entry points; learner, preprocessing, split and
LoRA implementations are unchanged. Third-party code and generated caches/raw
results go to `~/.cache/linkedin_track2/`, outside this submission directory.

TAR: E5-small-v2, physical train/evaluation batch **256**, accumulation 1,
float32, learning rate 0.0001, weight decay 0.001, rank 16 on the last three
layers, maximum 50 epochs and early-stopping patience 3. One encoder is tuned
per fold and reused by all five learners. No automatic batch reduction occurs.

The printed course verdict uses unrounded six-fold mean R-squared. Joint Signal
is joint frozen minus the better unimodal mean; TAR Gain is joint TAR minus
joint frozen. Each criterion requires a positive delta for at least 3/5 learners,
counted independently. Native RMSE is retained in the results CSV.

## Results provenance

The original experiment used official components through the earlier
`tt_curation` harness, with persistent checkpoints and zero data-loader workers.
The standalone native reproduction here uses upstream defaults, including four
workers; existing results are not relabelled as a new native run. Original source
and raw records remain preserved in the project repository at revision
`bfe0ba3b846cb8bd719d43a586dc4d25b36088cd` and its subsequent result commits.

## Current delivery status

`evaluation_results.csv` is **preliminary**: 95/120 original physical-batch-256
records are available, and its means use the four fully completed folds only.
The remaining evaluations and the separately written `final_report.pdf` are
pending. No placeholder PDF or fabricated final scores are included.

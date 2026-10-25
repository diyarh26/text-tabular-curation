# Track 2 delivery preparation

The selected submission dataset is LinkedIn salary. The paper is being written separately by the student.

Required contents are the curated CSV, its creation/evaluation source, a performance CSV for five learners and four conditions, and the English NeurIPS paper. See `docs/SUBMISSION_REQUIREMENTS.md` for the assignment checklist.

The repository currently contains preliminary original batch-256 results. To import the remaining records, copy only the matching experiment's original manifest and `folds/*/runs/*.json` files into `results/linkedin_batch256/linkedin_salary/`. Preserve all existing JSON bytes. Never fill missing rows from the 16x16 experiment. Then run:

```bash
python scripts/report_linkedin.py
```

When all 120 evaluations are audited and the final paper is ready:

```bash
python scripts/build_submission.py --paper paper/report.pdf --output artifacts/submission/linkedin_submission.zip
```

The builder uses an explicit file allowlist. It includes the selected CSV, the actual preparation/evaluation source copied once into runnable paths, required imports and dependency pins, the native MulTaBench integration, and result calculation. The completed delivery adds the performance CSVs, original manifest, 120 raw JSONs and paper.

It excludes all notebooks (including the resume notebook), transfer/resume tools, tests, toy data, old candidates, duplicate source snapshots, general repository-management tools, credentials, raw downloads, model checkpoints and the course handout. All original files remain preserved in the repository.

To review just the selected code and curated CSV before the paper and remaining results are ready:

```bash
python scripts/build_submission.py --code-only --output artifacts/submission/linkedin_code_and_data.zip
```

This code/data bundle is not the final submission. Its standalone [README](CODE_README.md) explains the three main steps, environment setup, original batch-256 execution and native MulTaBench execution. The original runtime hashes are verified when packaging. Historical helper filenames are retained because they are actual imports, not extra experiments to run.

Before upload, verify the paper uses the NeurIPS template, has at most four main-text pages, and uses appendices only for result tables/metrics. The builder does not validate those layout requirements. Verify Moodle's exact closing hour, file naming and upload fields there. Building the ZIP does not submit it.

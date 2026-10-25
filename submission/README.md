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

The builder rejects a missing/non-PDF paper and incomplete/mixed evaluation results. It includes the selected CSV, evaluation CSVs and raw JSONs, original manifest, native dataset integration, unmodified original runtime snapshot, dependency pins, and reproduction instructions. It excludes archives, credentials, raw downloads, model checkpoints and the course handout. The package keeps the repository's relative paths.

Before upload, verify the paper uses the NeurIPS template, has at most four main-text pages, and uses appendices only for result tables/metrics. The builder does not validate those layout requirements. Verify Moodle's exact closing hour, file naming and upload fields there. Building the ZIP does not submit it.

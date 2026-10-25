# Track 2 delivery

The submission is in [`track2_submission/`](../track2_submission/README.md):

```text
track2_submission/
  data/dataset.csv
  src/preprocess.py
  src/evaluate.py
  evaluation_results.csv
  final_report.pdf           # pending, written separately
  README.md
```

There are exactly two Python files. Cleaning is in `preprocess.py`; the native
MulTaBench registration, official evaluation commands and printed/saved results
are in `evaluate.py`. Neither imports other project code. All older source and
original results remain preserved in the repository, outside this delivery.

Build a review ZIP with the current preliminary results and no placeholder PDF:

```bash
python scripts/build_submission.py --code-only
```

Once all 120 original batch-256 records are available and the final paper is ready:

```bash
python scripts/build_submission.py --paper /path/to/final_report.pdf
```

The final builder audits the original results, rejects incomplete/mixed runs,
and packages only the six files in the layout above. No notebooks, tests, extra
source trees, raw JSONs, checkpoints or package manifests are added.

Before Moodle upload, verify the paper's NeurIPS format and the assignment's
four-page main-text and appendix rules. ZIP creation does not submit the project.

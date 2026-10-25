# LinkedIn-focused repository cleanup plan

Prepared 2026-10-25; implementation update on the same date. Native dataset integration and the broad code/data reorganization are now implemented. The LinkedIn paper remains assigned to a separate chat.

Completed: original VM code snapshot and recovery tags, three registered native feature views, exact loader/split parity, real native LightGBM structured CLI parity, 1,662 files archived with verified hashes, original 95 result records imported unchanged, a LinkedIn README, and a guarded final-package builder. The remaining 25 evaluations and paper are still pending.

## Target

Make the repository understandable as a LinkedIn salary text-tabular curation project. The normal workflow should expose one physical-batch-256 evaluation protocol and the official-source dataset integration. Keep earlier attempts together in `archive/`, with their origin and limitations visible. Preserve original results and source revisions rather than rewriting experiment history.

Work was prepared on `linkedin-submission-preparation`. The other agent's FDA results from `origin/main` were merged before archiving. The LinkedIn-focused default-branch update will be published only after the final local validation; no history rewrite is needed.

## Proposed layout

```text
README.md                      # LinkedIn objective, honest status/results, setup, commands
LICENSE                        # Code license; dataset attribution documented separately
pyproject.toml
requirements/
configs/linkedin_salary.yaml
data/processed/linkedin_salary.csv
prep/linkedin.py
src/tt_curation/               # Existing official-component runner, with scope explained
integration/multabench/        # Native dataset registration/data route, still to implement
scripts/                       # Canonical batch-256 run/report/validation and resume tools
notebooks/                     # Canonical batch-256 start and VM continuation only
tests/                         # Active pipeline/integration/resume checks
docs/                          # Dataset card, protocol, reproducibility, course checklist
results/linkedin_batch256/     # Final audited JSONs/CSVs/manifest; no model weights
paper/                         # LinkedIn paper, written by user in a separate chat
submission/                    # Release package instructions / generated package manifest
archive/                       # All earlier candidates, pilots and alternative-batch attempts
  README.md
  airbnb/
  steam/
  glassdoor/
  fda/
  lendingclub/
  linkedin_microbatch/
  pilots/
```

Exact script names will be chosen after the native integration decision; this tree is not a claim those integration files exist already. A frozen runtime snapshot may be retained under the LinkedIn reproduction tools if needed to execute the original VM-v1 experiment without changing its manifest.

## Keep active

- The exact selected LinkedIn CSV, config and creation code.
- Core official-model/preprocessing integration, dataset typing and condition definitions.
- The original batch-256 runtime needed to reproduce/resume the saved experiment, with an explicit source pin.
- Physical-batch-256 start/continuation notebooks, live fold reporting, exact dependency pins and provenance checks.
- Final six-fold metrics and raw run JSONs after completion. Keep partial files labelled preliminary until then.
- Smoke/toy support only if it is still used by the canonical setup/tests.

## Move into the archive

- Other candidate configs, preparation scripts, curated data and result summaries.
- TF-IDF screens, Colab pilots, early micro-batch notebooks and alternate batch-size experiments.
- Candidate-specific launchers, notebook builders, tests and handoff instructions.
- Existing Airbnb report and its bibliography/style, under `archive/airbnb/paper/`.

Archive source code is historical evidence. Moving files may break original relative imports, so archive instructions should point to the original Git revision for rerunning those experiments. Do not pretend every relocated historical script is a supported current entry point.

## Import dependencies that make this more than file moving

The active LinkedIn batch-256 launcher imports `run_full_linkedin.py`; that module imports the persistence helper from `colab_tar_compare.py`. Those files are still active dependencies despite their older names. They cannot simply be archived. Either retain and document them or carefully extract their active code and validate a new protocol separately.

Moving or editing these files changes code/manifest identity. The running VM-v1 continuation must continue checking out its pinned `bfe0ba3` runtime, with the original CSV/config and existing checkpoints. Do not adapt an old manifest to make a reorganized runtime appear unchanged.

## Files to preserve locally, outside Git publication

The VM backup and resume supplement currently under `docs/`, rolling ZIPs, raw downloads, adapters, optimizer states and trainer checkpoints are operational backups, not portfolio files. Preserve their bytes. Their final local storage location should be indexed once the active run finishes; moving them while a builder/resume path depends on them is unnecessary.

Do not add large model weights, checkpoint ZIPs, credentials, `.env`, raw source downloads or the course handout to GitHub. No history rewrite, repository visibility change, upstream pull request, or Moodle upload is part of ordinary file cleanup.

## Validation before the final merge

1. Inventory tracked and untracked files; preserve user-authored untracked work and concurrent remote commits.
2. Finish and test native MulTaBench dataset integration against the chosen pinned source, with an honest execution-difference record.
3. Organize the supported workflow and archive earlier candidates; repair active imports, commands, tests and documentation links.
4. Test canonical dataset validation, native loading/split parity, batch/LoRA guards and continuation preservation. Run GPU checks on available hardware; do not claim CPU tests prove GPU fit.
5. Import and audit all 120 batch-256 evaluations; generate the final CSVs and verdict without borrowing 16x16 records.
6. Make the README's results table and paper tables derive from those same final records.
7. Include the separate-chat paper only after it is checked against the assignment and evidence.
8. Build the submission package and verify every required file, data checksum, source revision and final-status label.
9. Review the final Git diff, merge while retaining remote work, and push the LinkedIn-focused default-branch version.

The native-integration gap, final 25 evaluations, paper and Moodle-specific upload details are open items. They are not hidden by cosmetic cleanup.

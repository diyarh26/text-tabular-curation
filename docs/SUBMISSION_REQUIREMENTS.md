# LinkedIn submission requirements

Checked: 2026-10-25. Scope: Track 2, standard project, one new dataset.

This checklist summarizes the local eight-page course handout, `NLP Course (097215) Final Project_ Text-Tabular Learning.pdf`, SHA-256 `406ca54fc55e77d54058b0b14fa32ffddf3bb24d5a41ddc69ac08880399a8811`. The copy in Downloads named `nlp_final_project_instructions.pdf` has the same extracted text and working source/template links. The handout stays local because it is course material.

## Required by the PDF

| Requirement | Location in handout | Planned deliverable / status |
|---|---|---|
| One new text-tabular dataset, not already included in MulTaBench | Sections 3.1 and 3.3.1, pp. 4-5 | LinkedIn salary; absent from the checked official dataset registry |
| Curated data in CSV format | Section 3.4, p. 6 | `data/processed/linkedin_salary.csv` already exists; 15,000 rows, 14 inputs plus target |
| Code used to create the dataset | Section 3.4, p. 6 | `prep/linkedin.py`; raw inputs and transformations documented in `LINKEDIN_DATASET_CARD.md` |
| Code used to evaluate the dataset, using official MulTaBench source | Sections 3.2 and 3.4, pp. 4 and 6 | Official model/preprocessing imports verified; native dataset integration remains unfinished; see `MULTABENCH_COMPLIANCE_AUDIT.md` |
| All five learners: TabM, CatBoost, LightGBM, TabPFN 2.0, TabPFN 2.5 | Section 3.2, p. 4 | Present in current evaluation |
| All four conditions: structured, text only, joint frozen, joint TAR | Section 3.2, p. 4 | Present in current evaluation |
| Strict improvement of joint frozen over both unimodal conditions, and joint TAR over joint frozen | Section 3.2.1, p. 4 | Preliminary four-fold means meet both criteria for 5/5 learners; final six-fold result pending |
| Performance CSV covering every learner and condition | Section 3.4, p. 6 | Final `evaluation_report.csv`: 20 learner/condition rows, six-fold means and standard deviations; retain per-fold raw JSONs as supporting evidence |
| Short paper in English, using the NeurIPS LaTeX template | Section 4, p. 6 | User will write the LinkedIn paper in a separate chat; `paper/README.md` records the handoff |
| At most four main-text pages, excluding references and appendices | Section 4, p. 6 | Verify compiled PDF before packaging |
| Appendices contain only result tables and metrics | Section 4, p. 6 | Put methodology and limitations in the main text, not an appendix |
| Individual submission through Moodle by Monday, October 26, 2026 | p. 1 and submission box, p. 7 | No submission made by this repository workflow |

The PDF says delta magnitude is the primary grading measure for dataset quality, and larger robust gaps are valued (p. 5). It does not prescribe a numerical delta threshold beyond strict improvement. It says the standard one-dataset scope can earn a top grade. The optional outstanding scope requires a systematic generation/mining method and at least five passing datasets; simply collecting our earlier experiments is not that extension.

Required paper contents (Section 4.1, p. 6):

1. Title and abstract.
2. Introduction: problem, text-tabular context, and contribution.
3. Related work: cite MulTaBench, the learners used, and relevant representation/fine-tuning work.
4. Method / Dataset: data origins, curation procedure, and evaluation pipeline.
5. Results: experimental setup, baselines, metrics, tables and charts.
6. Conclusion and summary: outcomes, limitations, and future work.

AI tools are permitted and encouraged; the student remains responsible for all submitted content (pp. 1-2). The handout specifies a 20-point daily late penalty unless covered by applicable Technion regulations (p. 7). It does not give an exact submission hour.

## Staff forum clarifications supplied by the user

These are separate from the PDF and were not independently accessed on Moodle. The user supplied the complete four-question reply from Eilam Shapira on 2026-10-25:

- There is no material difference between `master` and `paper_version` for this purpose; using `master` is preferred. The project's pinned official revision is an ancestor of the checked current `master`, rather than the `paper_version` tag.
- Count the two criteria independently: at least 3/5 for Joint Signal and at least 3/5 for TAR Gain. The qualifying learners may differ.
- Use raw, unrounded scores for pass/fail. A near-zero margin may satisfy the inequality but is a weak submission.
- A new dataset must run inside MulTaBench. The quoted staff answer rejected submitting only an external CSV loader/wrapper around the evaluation functions.

The final clarification makes native dataset integration an explicit completion item. Importing the upstream model code alone does not establish that requirement.

## Evaluation conventions already selected for this project

These come from the official pinned source and prior project decisions; the PDF does not itself specify six folds, a 15,000-row sample, or a batch size:

- Official source commit: `ff0c9c8b40a2085a06a7771190e53a261341802f`.
- Six splits: repeated three-fold CV with two repeats and seed 42; 10,000 train and 5,000 test rows per split.
- Five learners x four conditions x six folds = 120 evaluations.
- Primary report metric: R-squared on `log_salary`. Keep official RMSE diagnostics too.
- TAR: physical batch 256, accumulation 1, E5-small-v2, float32, up to 50 epochs with patience 3; source weight decay 0.001.
- Current experiment is `linkedin_vm_physical256_v1` at project commit `bfe0ba3b846cb8bd719d43a586dc4d25b36088cd`. It is distinct from the newer native-v2 launcher and the earlier 16x16 run.

## Supporting files recommended for reproducibility

Keep a dataset card, data/config checksums, dependency pins, source revision, six split hashes, exact commands, raw results, and an explanation of execution additions. These make the code and CSV reproducible; they are not additional named PDF deliverables.

The Hugging Face model-upload and separate inference/training requirements in Section 2.3 belong to Track 1. The Track 2 list does not explicitly require uploading the six TAR adapters. Keep the adapters/checkpoints for resume and audit, without inventing an extra Track 2 requirement.

Planned package structure (filenames are our convention, not mandated by the PDF):

```text
submission/
  README.md
  linkedin_salary.csv
  evaluation_report.csv
  curation_verdict.csv
  per_fold_results.csv
  report.pdf
  source/                 # preparation, native integration, evaluation, pins
  provenance/             # manifest, checksums, source and split identities
```

Do not mark this package final while the batch-256 run is incomplete, native integration is untested, or the paper is absent. Exact Moodle upload format, naming convention and closing hour should be checked in the submission page; the PDF does not specify them.

## Current readiness

| Item | Status |
|---|---|
| Chosen curated CSV and creation script | Present |
| Column and cleaning documentation | Prepared in `LINKEDIN_DATASET_CARD.md` |
| Dependency and upstream source pins | Present |
| Six official split identities | Recomputed and matched to saved manifest |
| Physical batch-256 evaluation | 95/120 successful records in latest locally available VM backup; folds 0-3 complete |
| Final six-fold CSVs and verdict | Pending remaining 25 evaluations and final audit |
| Native MulTaBench dataset integration | Required; not implemented/tested yet |
| LinkedIn paper | User will write in a separate chat |
| Final package and Moodle submission | Not ready / not submitted |

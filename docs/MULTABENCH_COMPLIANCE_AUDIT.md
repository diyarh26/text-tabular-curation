# LinkedIn batch-256 source and integration audit

Audit date: 2026-10-25. This is a source/data/manifest audit plus existing local regression tests, not a new GPU evaluation.

## Conclusion

The existing run uses MulTaBench's actual learner classes, feature preprocessing, E5/LoRA implementation, default core training hyperparameters, and six train/test splits. It was produced by the project's runner, which constructs `MultimodalDataset(..., dataset_id=None)` itself.

Native dataset integration has now been added separately: LinkedIn is registered inside an isolated pinned MulTaBench checkout, with typed full/structured/text views and local cache lookup. The original native benchmark/evaluation, model, preprocessing, split and LoRA files are unchanged. Native loader parity and a real structured LightGBM CLI evaluation passed. See [NATIVE_MULTABENCH.md](NATIVE_MULTABENCH.md) and [the loader evidence](audit/native_linkedin_loader_parity.json).

Do not describe the historical run as if it had used the newly added native CLI route. Its provenance and execution differences below remain valid. No 120-run rerun was started as part of this integration.

## Revisions checked

| Component | Revision |
|---|---|
| Official source installed for the experiment | `ff0c9c8b40a2085a06a7771190e53a261341802f` |
| Original VM batch-256 project source | `bfe0ba3b846cb8bd719d43a586dc4d25b36088cd` |
| Original VM protocol | `linkedin_vm_physical256_v1` |
| Latest official master inspected on audit date | `13dab34b2cb5dee7f532d1603cb8cb93bf63be63` |

The latest upstream master was checked, not installed over the running experiment. The original VM manifest's 13 project-source hashes all match the corresponding files at `bfe0ba3`. The current working tree has a later version of `scripts/run_full_linkedin_vm.py`; do not substitute it into the pinned continuation.

GitHub's commit comparison confirms that `ff0c9c8` is an ancestor of the checked current `master`. This follows the staff's stated preference for master-based code, supplied in full by the user on the audit date. The response did not instruct students to upgrade an experiment to every subsequent master commit. No switch to `paper_version` is needed for this project.

Machine-readable upstream comparison: [audit/multabench_source_comparison_2026-10-25.json](audit/multabench_source_comparison_2026-10-25.json).

## What matches the pinned official code

| Component | Evidence |
|---|---|
| Five learner implementations | `runner._model_class` directly imports upstream TabM, CatBoost, LightGBM, TabPFNv2 and TabPFNv2p5; model implementations are not copied into a local replacement |
| Feature typing and preprocessing | CSV inputs are explicitly typed by YAML, then upstream model wrappers perform their native preprocessing, target transforms and text projection |
| Outer splits | `multabench.benchmark.load.split_dataset` / `benchmark.splits.get_split`; all six saved index hashes independently recomputed and matched |
| Input sample | Exact 15,000-row CSV and config checked against the VM manifest's LF-normalized hashes |
| Text encoder | Upstream `intfloat/e5-small-v2`, passage and column-name formatting, native pooling and 512-token truncation |
| TAR implementation | Native `finetune_e5` and `finetune_with_lora`; regression labels are natively discretized into 20 quantile bins for the auxiliary classification objective |
| Train/eval batch | 256 / 256; gradient accumulation 1; native mean loss, no accumulation-loss override at this batch |
| LoRA | Rank 16, alpha 32, dropout 0.1, last three transformer layers; upstream target-module list |
| Optimizer-related settings | Learning rate 0.0001; weight decay 0.001; float32; native Trainer defaults for other settings |
| Training budget | 50 epochs maximum, validation and saving each epoch, patience 3, best validation-loss model restored |
| Fold-specific TAR | One encoder trained only from the outer training rows per fold, shared by all five learners, analogous to upstream fold-specific cached TAR embeddings |
| Metrics | Native target transformation and RMSE metric retained; R-squared added for the course report, using the same true/predicted values |

The checked installed upstream files match the pinned source bytes after CRLF-to-LF normalization. The source comparison JSON lists the checked files; it is not a claim that every installed dependency file was audited.

## Execution additions and differences

| Area | Current VM-v1 behavior | Official pinned behavior / implication |
|---|---|---|
| Dataset integration | Original run: our CSV/YAML loader; `dataset_id=None`; our experiment loop | New dataset patch supplies native registered IDs and typed views; historical scores retain their original execution identity |
| Data-loader workers | Persistent helper forces zero workers | Official LoRA default is four. This is an execution difference that must be disclosed, not hidden by calling the entire run identical |
| Checkpoint lifetime | Persistent trainer/optimizer/scheduler/RNG state and completed adapters | Official LoRA uses a temporary checkpoint directory |
| Restart semantics | Explicit restoration of early-stopping callback state | Not explicitly enabled by official TrainingArguments. This preserves patience on continuation; it is still an execution addition |
| Encoder reuse | Shared encoder interception, in-memory embedding memoization, idle frozen model moved to CPU | Upstream normally uses its dataset embedding-cache route; our additions reduce repeated work and VRAM use |
| Reporting and monitoring | R-squared criterion report, per-fold tables, manifests, failure audits and ZIP exports | Additional orchestration/reporting, not upstream entry-point output |
| Resume scheduling | Clean processes for missing phases; one visible GPU per worker | An operational continuation arrangement; original pinned model and training code remains in use |

Do not change zero workers to four in the active VM-v1 continuation merely to align this table. A protocol change must be a separate experiment. The newer `linkedin_native_physical256_v2` notebook restores four workers, but it is not the source of the saved VM-v1 scores and encountered a Kaggle TAR memory failure.

## Current upstream changes

Compared with the pinned official revision, the checked latest master has identical E5 fine-tuning, LoRA, train-argument, split, data-loader, text preprocessing, metric, CatBoost, LightGBM and TabM files. The abstract model differs only in comments (identical Python AST).

TabPFN has a real update: many-class classification support above ten classes and a TabPFN-v2.5 display-name correction. The added many-class branch is not used for this regression task. This audit does not run a numerical comparison of the two TabPFN revisions. Keep the original revision pinned to preserve provenance.

## Native integration implementation and validation

1. Dataset-only patch and native recipe supplied under `integration/multabench/`; original Kaggle attribution is in the registry and metadata.
2. Exact selected rows exported into typed native data/metadata, with required CSV retained unchanged.
3. Local cache routing and full/text cache sharing use upstream's actual cache reader. Missing caches fail; learner, preprocessing, split, embedding-computation and LoRA code is unchanged.
4. Native CLI recognizes all views. All four input/encoder contracts and six split identities passed; real LightGBM structured fold 0 matches the existing runner's score within `1e-12`.
5. GPU frozen-text/TAR numerical parity and a full native GPU run have not been performed locally. This is disclosed rather than inferred from CPU tests. Integration alone does not replace or erase the original 95 saved records.

The delivery is an offline/local-data native integration. No dataset has been published to Hugging Face and no pull request has been submitted to upstream. The original VM-v1 code is also preserved, unmodified, in `reproduction/original_vm256/`.

## Submission boundaries

Local validation of the existing VM and continuation tools passed: `pytest tests/test_linkedin_vm.py tests/test_vm256_resume.py -m "not bench" -q` produced 31 passed, 4 deselected. This checks the existing guard/report/resume code; it is not a GPU-memory test or a test of the missing native integration.

- Course acceptance uses raw R-squared fold means and independent 3/5 counts, following the staff answers supplied by the user. This is a course reporting convention, not a claim that upstream's default result metric is R-squared.
- Physical batch 256 matches the pinned source default. It does not, by itself, establish complete pipeline equivalence.
- At this audit, the latest local backup has 95 successful evaluations: folds 0-3 complete, 15 baseline evaluations on fold 4, none on fold 5. Final results must await all 120.
- The existing Kaggle 16x16 experiment is separate supporting evidence; it must not fill missing rows in the batch-256 experiment.

## Primary source references

- [Pinned official benchmark entry point](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/benchmark.py)
- [Native evaluation functions](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/baselines/benchmarks/evaluate.py)
- [Official dataset registry](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/datasets/all_datasets.py)
- [Dataset format and loading documentation](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/docs/datasets.md)
- [LoRA defaults](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/train_args.py)
- [Official LoRA Trainer construction](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/finetune/lora.py)
- [Six-fold split implementation](https://github.com/alanarazi7/MulTaBench/blob/ff0c9c8b40a2085a06a7771190e53a261341802f/multabench/benchmark/splits.py)

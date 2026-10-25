# Original batch-256 code, preserved

The `code/` directory contains **unmodified Git blobs** from project commit
`bfe0ba3b846cb8bd719d43a586dc4d25b36088cd`, the source of the active
`linkedin_vm_physical256_v1` experiment. `source_manifest.json` records their raw
SHA-256 hashes. This includes every code file recorded by the experiment manifest,
plus the original dataset preparation/configuration, setup, README and license.

The exact CSV is kept once at `data/processed/linkedin_salary.csv` in the main
repository. It is not resampled or duplicated here. The raw dataset remains local.
This snapshot is evidence, rather than a second independently installed package.

To rerun/resume the original implementation with its original paths, use a separate
checkout of the preserved revision, for example:

```bash
git worktree add --detach ../linkedin-original bfe0ba3b846cb8bd719d43a586dc4d25b36088cd
```

Use the existing source-pinned resume notebook for the active VM-to-Kaggle
continuation. Never install the reorganized or dataset-patched runtime over that
environment and never rewrite its manifest to bypass a mismatch.

The native dataset integration is provided separately under `integration/multabench/`.
Its upstream evaluation, learner, preprocessing, split and LoRA files are unchanged.

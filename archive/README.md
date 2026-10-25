# Earlier experiments

This folder preserves earlier dataset candidates, pilots and alternative execution attempts. The supported main workflow is now LinkedIn salary prediction with physical batch 256.

Candidates and their code, configs, curated data, results, notebooks, tests and notes are grouped under `airbnb/`, `steam/`, `glassdoor/`, `fda/`, `lendingclub/`, `linkedin_legacy/` and `pilots/`. The old Airbnb paper is under `airbnb/paper/`.

`move_manifest.csv` records original paths, archived paths and content hashes for the 1,662 files moved in the broader cleanup. Every file's bytes were checked before and after the move. The LF-normalized hashes also support checking across Git's Windows/Linux line-ending conversion. Original Git history is retained.

Runtime logs and Python bytecode in those local archives remain ignored by Git; their presence in the move manifest records local preservation, not a requirement to publish generated files. All archived source, configs, curated data and evaluation records are retained in Git.

The exact original VM-v1 runtime is additionally preserved, unmodified, under [reproduction/original_vm256/](../reproduction/original_vm256/). Git tags `original-linkedin-vm256-bfe0ba3` and `before-linkedin-cleanup` preserve original source revisions.

Historical artifacts are not final LinkedIn submission results. Use their original Git revisions when reproducing them; relative paths may differ after archival.

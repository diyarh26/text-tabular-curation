"""Preserve an old experiment and reuse only validated, TAR-independent baseline JSONs.

Use after approving a different TAR micro-batch. Data, configuration, model code,
dependencies and every other scientific setting must match exactly. No TAR adapter,
training checkpoint, tuned result or partial aggregate is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from colab_tar_compare import write_json
from run_full_linkedin import MODELS, audit_runs, fold_dir


def migrate(root, expected, source_revision):
    root = Path(root).resolve()
    output = root / "linkedin_salary"
    manifest = output / "manifest.json"
    old = json.loads(manifest.read_text(encoding="utf-8"))
    if old == expected:
        print("The experiment already matches the requested settings; no migration needed.")
        return 0
    differences = {key for key in old.keys() | expected.keys() if old.get(key) != expected.get(key)}
    if not differences <= {"micro_batch", "source_revision"} or old.get("micro_batch") == expected.get("micro_batch"):
        raise ValueError(f"Only a TAR micro-batch/source revision change can reuse baselines; differences: {sorted(differences)}")
    if old.get("micro_batch") != 256 or expected.get("micro_batch") != 16:
        raise ValueError("This migration is restricted to the approved 256 -> 16 adjustment.")
    pending = []
    for fold in range(6):
        problems = audit_runs(output, [fold])
        for model in MODELS:
            for condition in ("structured", "unstructured", "joint_frozen"):
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                if not path.exists():
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") != "ok":
                    continue
                prefix = f"fold{fold}/{path.name}:"
                if any(issue.startswith(prefix) for issue in problems):
                    raise ValueError(f"Cannot import invalid baseline: {path}")
                pending.append((path.relative_to(root), hashlib.sha256(path.read_bytes()).hexdigest()))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = root.with_name(root.name + "_batch256_backup_" + stamp)
    staging = root.with_name(root.name + "_micro16_staging_" + stamp)
    # Both explicit paths must stay within the same working parent before moving the tree.
    if (backup.parent.resolve() != root.parent.resolve() or staging.parent.resolve() != root.parent.resolve()
            or backup.exists() or staging.exists()):
        raise ValueError("Unexpected or occupied backup path")
    # Build the new output completely before moving the old tree; a copy error leaves the old run intact.
    staging.mkdir()
    write_json(staging / "linkedin_salary/manifest.json", expected)
    (staging / "source_revision.txt").write_text(source_revision + "\n", encoding="utf-8")
    imported = []
    for relative, digest in pending:
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baseline changed during migration: {relative}")
        imported.append({"file": relative.as_posix(), "sha256": digest})
    write_json(staging / "baseline_reuse.json", {
        "reason": "Approved TAR micro-batch 16, accumulation 16, effective batch 256; baseline training unchanged",
        "backup_directory": str(backup), "source_manifest": old,
        "migration_utility_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "imported_baselines": imported, "tar_results_imported": 0,
    })
    # Preserve the entire previous experiment; never delete or overwrite its checkpoints/results.
    root.rename(backup)
    if old.get("source_revision"):
        (backup / "source_revision.txt").write_text(old["source_revision"] + "\n", encoding="utf-8")
    staging.rename(root)
    print(f"Preserved previous experiment in {backup}")
    print(f"Imported {len(imported)} validated baseline evaluations; TAR starts fresh with micro-batch 16.")
    return len(imported)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    args = parser.parse_args()
    from tt_curation import REPO_ROOT
    source_revision = (REPO_ROOT / "SOURCE_REVISION").read_text(encoding="utf-8").strip()
    with tempfile.TemporaryDirectory(prefix="linkedin_microbatch_manifest_") as temporary:
        subprocess.run([sys.executable, str(Path(__file__).with_name("run_full_linkedin.py")),
                        "--results_root", temporary, "--micro_batch", "16", "--prepare_only"], check=True)
        expected = json.loads((Path(temporary) / "linkedin_salary/manifest.json").read_text(encoding="utf-8"))
    migrate(args.results_root, expected, source_revision)


if __name__ == "__main__":
    main()

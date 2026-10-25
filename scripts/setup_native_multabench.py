"""Prepare an isolated pinned MulTaBench checkout with dataset-only integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "ff0c9c8b40a2085a06a7771190e53a261341802f"
INTEGRATION = ROOT / "integration/multabench"
PATCH_FILES = {"multabench/datasets/all_datasets.py", "multabench/datasets/info.py",
               "multabench/datasets/hub.py", "multabench/benchmark/load.py", "multabench/embeddings/hub.py"}
OVERLAY = "multabench/datasets/linkedin_salary.py"
OVERLAY_FILES = {OVERLAY, "multabench/datasets/prepare_linkedin.py"}


def git(checkout, *args):
    return subprocess.check_output(["git", "-c", "core.autocrlf=false", "-C", str(checkout), *args],
                                   stderr=subprocess.PIPE)


def setup(checkout):
    checkout = Path(checkout).resolve()
    patch = INTEGRATION / "linkedin_salary.patch"
    if checkout.exists() and not (checkout / ".git").is_dir():
        raise ValueError("Destination must be absent or an existing pinned Git checkout")
    if not checkout.exists():
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/alanarazi7/MulTaBench.git",
                        str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "-c", "core.autocrlf=false", "checkout", "--detach", COMMIT], check=True)
    if git(checkout, "rev-parse", "HEAD").decode().strip() != COMMIT:
        raise ValueError("Refusing a different upstream revision")
    expected_patch = patch.read_bytes()
    diff = git(checkout, "diff", "--no-ext-diff", "--binary", "HEAD", "--", *sorted(PATCH_FILES))
    if not diff:
        if git(checkout, "status", "--porcelain", "--untracked-files=all").strip():
            raise ValueError("Refusing unrelated existing changes in the upstream checkout")
        git(checkout, "apply", "--check", str(patch))
        git(checkout, "apply", str(patch))
        for relative in OVERLAY_FILES:
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(INTEGRATION / "overlay" / relative, destination)
    elif diff != expected_patch:
        raise ValueError("Existing upstream changes differ from the reviewed dataset patch")
    for relative in OVERLAY_FILES:
        if not (checkout / relative).exists() or (checkout / relative).read_bytes() != (INTEGRATION / "overlay" / relative).read_bytes():
            raise ValueError("Native LinkedIn recipe differs: " + relative)
    changed = set(git(checkout, "diff", "--name-only", "HEAD").decode().splitlines())
    untracked = set(git(checkout, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    if changed != PATCH_FILES or untracked != OVERLAY_FILES:
        raise ValueError(f"Unexpected upstream modifications: {changed}, {untracked}")
    # Every upstream tracked file outside the explicit data/lookup patch stays byte-identical.
    report = dict(upstream_commit=COMMIT, patch_sha256=hashlib.sha256(expected_patch).hexdigest(),
        overlay_sha256={p: hashlib.sha256((INTEGRATION / "overlay" / p).read_bytes()).hexdigest()
                        for p in sorted(OVERLAY_FILES)}, modified_files=sorted(changed),
        added_files=sorted(untracked), unchanged_evaluation_entrypoint=True,
        unchanged_models_preprocessing_splits_and_lora=True)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", default=str(ROOT / "artifacts/multabench-native"))
    args = parser.parse_args()
    setup(args.checkout)

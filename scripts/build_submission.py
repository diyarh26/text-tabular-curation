"""Package only LinkedIn creation/evaluation/reporting code and required support."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from report_linkedin import ROOT, report


ORIGINAL = ROOT / "reproduction/original_vm256"
# These are the actual execution files, copied once into runnable root paths.
# The old setup script depends on the excluded toy dataset; setup is documented instead.
ORIGINAL_FILES = (
    "LICENSE", "pyproject.toml", "requirements/constraints.txt",
    "prep/linkedin.py", "configs/linkedin_salary.yaml",
    "scripts/run_full_linkedin_vm.py", "scripts/run_full_linkedin.py",
    "scripts/colab_tar_compare.py", "scripts/run_linkedin_vm256.sh",
    "src/tt_curation/__init__.py", "src/tt_curation/__main__.py",
    "src/tt_curation/conditions.py", "src/tt_curation/criteria.py",
    "src/tt_curation/data.py", "src/tt_curation/runner.py", "src/tt_curation/spec.py",
)
SUPPORT_FILES = (
    ".gitignore", "data/processed/linkedin_salary.csv",
    "scripts/report_linkedin.py", "scripts/setup_native_multabench.py",
    "scripts/verify_native_linkedin.py", "scripts/report_native_linkedin.py",
    "scripts/run_native_linkedin.sh",
    "integration/multabench/linkedin_salary.patch",
    "integration/multabench/overlay/multabench/datasets/linkedin_salary.py",
    "integration/multabench/overlay/multabench/datasets/prepare_linkedin.py",
    "reproduction/original_vm256/experiment_manifest.json",
    "reproduction/original_vm256/original_run_sha256.json",
    "docs/LINKEDIN_DATASET_CARD.md",
)


def code_paths():
    original = json.loads((ORIGINAL / "source_manifest.json").read_text(encoding="utf-8"))
    paths = {relative: ORIGINAL / "code" / relative for relative in ORIGINAL_FILES}
    for relative, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["raw_git_blob_sha256"][relative]:
            raise ValueError("Original evaluation/preparation source changed: " + relative)
    paths.update({relative: ROOT / relative for relative in SUPPORT_FILES})
    paths["README.md"] = ROOT / "submission/CODE_README.md"
    return paths


def write_package(paths, destination, **metadata):
    destination = Path(destination).resolve()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing delivery files: " + str(missing))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = json.loads((ORIGINAL / "source_manifest.json").read_text(encoding="utf-8"))
    package = dict(code_revision=revision, original_experiment_revision=source["source_revision"],
        original_source_sha256={name: source["raw_git_blob_sha256"][name] for name in ORIGINAL_FILES},
        **metadata, files={name: hashlib.sha256(path.read_bytes()).hexdigest()
                           for name, path in sorted(paths.items())})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Choose a new output instead of overwriting an existing submission ZIP")
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(paths.items()):
            archive.write(path, name)
        archive.writestr("package_manifest.json", json.dumps(package, indent=2) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
    print(f"Prepared {destination}; {len(paths)} selected files. No Moodle upload performed.")
    return destination


def build_code(destination):
    """A reviewable code/data bundle, independent of the pending paper/results."""
    return write_package(code_paths(), destination, delivery="code_and_data_only",
                         final_submission=False, complete_evaluations=None)


def build(results, paper, destination):
    results,paper,destination=Path(results).resolve(),Path(paper).resolve(),Path(destination).resolve()
    if not paper.is_file() or paper.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Supply the final LinkedIn paper PDF; file missing or PDF header invalid")
    status=report(results)
    if not status["complete"]:
        raise ValueError("Submission is not complete: all 120 original batch-256 evaluations are required")
    paths = code_paths()
    paths["paper/report.pdf"] = paper
    result_prefix="results/linkedin_batch256/linkedin_salary/"
    for name in ["evaluation_report.csv","curation_verdict.csv","per_fold_results.csv","final_audit.json","manifest.json","summary.txt"]:
        paths[result_prefix+name]=results/name
    for path in results.glob("folds/*/runs/*.json"):
        paths[result_prefix+path.relative_to(results).as_posix()]=path
    return write_package(paths, destination, delivery="complete_track2", final_submission=True,
        complete_evaluations=120,
        paper_format_check="User must verify four main-text pages, NeurIPS template and appendix limits")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default=str(ROOT/"results/linkedin_batch256/linkedin_salary"))
    parser.add_argument("--paper")
    parser.add_argument("--code-only", action="store_true", help="Build selected code and CSV before final results/paper")
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.code_only:
        if args.paper:
            parser.error("--paper is only used for a complete submission")
        build_code(args.output or ROOT/"artifacts/submission/linkedin_code_and_data.zip")
    else:
        if not args.paper:
            parser.error("--paper is required unless --code-only is selected")
        build(args.results,args.paper,args.output or ROOT/"artifacts/submission/linkedin_submission.zip")

"""Build the requested Track 2 layout, with exactly two Python source files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile

from report_linkedin import ROOT, report

DELIVERY = ROOT / "track2_submission"
PREFIX = "track2_submission/"
FILES = ("src/preprocess.py", "src/evaluate.py", "data/dataset.csv",
         "evaluation_results.csv", "README.md")
CSV_SHA256_LF = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"


def selected_files():
    paths = {PREFIX + relative: DELIVERY / relative for relative in FILES}
    if hashlib.sha256((DELIVERY / "data/dataset.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest() != CSV_SHA256_LF:
        raise ValueError("Curated submission CSV differs from the selected dataset")
    return paths


def write_package(paths, destination):
    destination = Path(destination).resolve()
    contents = {name: value if isinstance(value, bytes) else value.read_bytes()
                for name, value in paths.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, content in sorted(contents.items()):
            archive.writestr(name, content)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or any(archive.read(name) != value for name, value in contents.items()):
            raise ValueError("ZIP integrity failure")
    print(f"Prepared {destination}; exactly two Python files. No Moodle upload performed.")
    return destination


def build_code(destination):
    """Review bundle: two scripts, dataset, current results CSV and README."""
    return write_package(selected_files(), destination)


def build(results, paper, destination):
    results, paper = Path(results).resolve(), Path(paper).resolve()
    if not paper.is_file() or paper.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Supply the final LinkedIn paper PDF; file missing or PDF header invalid")
    status = report(results)
    if not status["complete"]:
        raise ValueError("Submission is not complete: all 120 original batch-256 evaluations are required")
    paths = selected_files()
    paths[PREFIX + "evaluation_results.csv"] = results / "evaluation_report.csv"
    paths[PREFIX + "final_report.pdf"] = paper
    readme = (DELIVERY / "README.md").read_text(encoding="utf-8").split("## Current delivery status")[0]
    paths[PREFIX + "README.md"] = (readme + "## Delivery status\n\n"
        "All 120 original physical-batch-256 evaluations have passed the provenance audit. "
        "The results CSV contains six-fold means for every learner and condition; "
        "the final paper is included.\n").encode("utf-8")
    return write_package(paths, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results/linkedin_batch256/linkedin_salary"))
    parser.add_argument("--paper")
    parser.add_argument("--code-only", action="store_true", help="Review the two scripts, data and current results before final delivery")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.code_only:
        if args.paper:
            parser.error("--paper is only used for the complete submission")
        build_code(args.output or ROOT / "artifacts/submission/track2_submission_review.zip")
    else:
        if not args.paper:
            parser.error("--paper is required unless --code-only is selected")
        build(args.results, args.paper, args.output or ROOT / "artifacts/submission/track2_submission.zip")

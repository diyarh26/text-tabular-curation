"""Build a complete Track 2 delivery; reject incomplete results or a missing paper."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from report_linkedin import ROOT, report


def build(results, paper, destination):
    results,paper,destination=Path(results).resolve(),Path(paper).resolve(),Path(destination).resolve()
    if not paper.is_file() or paper.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Supply the final LinkedIn paper PDF; no placeholder paper is accepted")
    status=report(results)
    if not status["complete"]:
        raise ValueError("Submission is not complete: all 120 original batch-256 evaluations are required")
    source_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    paths={"data/processed/linkedin_salary.csv":ROOT/"data/processed/linkedin_salary.csv", "paper/report.pdf":paper,
        "submission/README.md":ROOT/"submission/README.md"}
    result_prefix="results/linkedin_batch256/linkedin_salary/"
    for name in ["evaluation_report.csv","curation_verdict.csv","per_fold_results.csv","final_audit.json","manifest.json","summary.txt"]:
        paths[result_prefix+name]=results/name
    for path in results.glob("folds/*/runs/*.json"):
        paths[result_prefix+path.relative_to(results).as_posix()]=path
    # Deliberately include only LinkedIn creation, integration, evaluation and reproducibility code.
    for folder in ["src","prep","configs","requirements","integration","reproduction","scripts","tests","notebooks"]:
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix not in [".pyc"]:
                paths[path.relative_to(ROOT).as_posix()]=path
    for rel in ["pyproject.toml","README.md","LICENSE",".gitignore",".gitattributes"]:
        paths[rel]=ROOT/rel
    for name in ["LINKEDIN_DATASET_CARD.md","MULTABENCH_COMPLIANCE_AUDIT.md","NATIVE_MULTABENCH.md","SUBMISSION_REQUIREMENTS.md"]:
        paths["docs/"+name]=ROOT/"docs"/name
    missing=[str(p) for p in paths.values() if not p.is_file()]
    if missing: raise FileNotFoundError("Missing delivery files: "+str(missing))
    package=dict(code_revision=source_revision,original_experiment_revision=status["original_project_revision"],
        complete_evaluations=120,paper_format_check="User must verify four main-text pages, NeurIPS template and appendix limits",
        files={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sorted(paths.items())})
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): raise FileExistsError("Choose a new output instead of overwriting an existing submission ZIP")
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(paths.items()): archive.write(path,name)
        archive.writestr("package_manifest.json",json.dumps(package,indent=2)+"\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None: raise ValueError("ZIP integrity failure")
    print(f"Prepared {destination}; {len(paths)} files, source {source_revision}. No Moodle upload performed.")
    return destination


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default=str(ROOT/"results/linkedin_batch256/linkedin_salary"))
    parser.add_argument("--paper",required=True)
    parser.add_argument("--output",default=str(ROOT/"artifacts/submission/linkedin_submission.zip"))
    args=parser.parse_args()
    build(args.results,args.paper,args.output)

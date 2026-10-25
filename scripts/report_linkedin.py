"""Audit/report original VM-v1 LinkedIn results without changing their provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from tt_curation.criteria import curation_verdict, evaluation_report

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "bfe0ba3b846cb8bd719d43a586dc4d25b36088cd"
PROTOCOL = "linkedin_vm_physical256_v1"
MODELS = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM",
          "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}
CONDITIONS = ["structured", "unstructured", "joint_frozen", "joint_tar"]


def audit(output):
    output = Path(output)
    reference = json.loads((ROOT / "tests/fixtures/vm256_manifest.json").read_text(encoding="utf-8"))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != reference:
        raise ValueError("Expected the exact original VM-v1 experiment manifest; never mix experiments")
    for path, key in [(ROOT / "data/processed/linkedin_salary.csv", "csv_sha256_lf"),
                      (ROOT / "configs/linkedin_salary.yaml", "config_sha256_lf")]:
        if hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != manifest[key]:
            raise ValueError("Selected data/config differs: " + str(path))
    expected = {}
    for fold in range(6):
        for short in MODELS:
            for condition in CONDITIONS:
                name = f"folds/fold{fold}/runs/{short}_{condition}_10k_fold{fold}.json"
                expected[name] = (short, condition, fold)
    actual = {p.relative_to(output).as_posix(): p for p in output.glob("folds/*/runs/*.json")}
    if set(actual) - set(expected):
        raise ValueError("Unexpected/duplicate run files: " + str(sorted(set(actual)-set(expected))))
    records, hashes, failures = [], {}, []
    for name, path in sorted(actual.items()):
        short, condition, fold = expected[name]
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "ok":
            failures.append(dict(path=name, error=row.get("error", row.get("status"))))
            continue
        identity = dict(dataset="linkedin_salary", short_name=short, model=MODELS[short], fold=fold,
            condition=condition, protocol=PROTOCOL, git=SOURCE, multabench_commit=reference["multabench_commit"],
            task="regression", size="10k", n_train=10000, n_test=5000,
            n_features=12 if condition=="structured" else 2 if condition=="unstructured" else 14,
            metric="root_mean_squared_error", text_encoder=None if condition=="structured" else
                "e5-small-tar" if condition=="joint_tar" else "e5-small")
        if any(row.get(k) != v for k,v in identity.items()):
            raise ValueError("Result identity/provenance differs: " + name)
        for key in ["r2", "test_error", "test_score", "runtime_s"]:
            value=row.get(key)
            if not isinstance(value,(float,int)) or isinstance(value,bool) or not math.isfinite(value):
                raise ValueError(f"Nonfinite/missing {key}: {name}")
        if (row["r2"] > 1+1e-8 or row["test_error"] < 0 or row["runtime_s"] < 0
                or not str(row.get("device", "")).startswith("cuda")
                or not math.isclose(row["test_score"], -row["test_error"], abs_tol=1e-12)):
            raise ValueError("Invalid metric/device: " + name)
        if condition == "joint_tar":
            stats=row.get("tar_finetune") or {}
            if (row.get("tar_parameters") != reference["tar_parameters"]
                    or row.get("training_arguments") != reference["training_arguments"]
                    or stats.get("rows") != 10000
                    or not isinstance(stats.get("epochs"),(int,float)) or not 0 < stats["epochs"] <= 50
                    or not isinstance(stats.get("best_val_loss"),(int,float)) or not math.isfinite(stats["best_val_loss"])):
                raise ValueError("TAR parameters/budget differ: " + name)
        records.append(row)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    counts={f:sum(r["fold"]==f for r in records) for f in range(6)}
    complete=[f for f,n in counts.items() if n==20]
    status=dict(dataset="linkedin_salary", protocol=PROTOCOL, original_project_revision=SOURCE,
        official_revision=manifest["multabench_commit"], successful_evaluations=len(records), expected_evaluations=120,
        evaluations_by_fold=counts, completed_folds=complete, complete=len(records)==120 and not failures,
        missing=sorted(set(expected)-set(actual)), failed=failures, raw_run_sha256=hashes,
        raw_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        primary_metric="r2", original_execution="tt_curation runner using pinned official components",
        native_loader_integration="provided separately; existing results not relabelled")
    return pd.DataFrame(records), status


def report(output):
    output=Path(output)
    rows,status=audit(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"final_audit.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
    if not rows.empty:
        rows[["model","short_name","condition","fold","r2","test_error","test_score","runtime_s","git","protocol"]].to_csv(output/"per_fold_results.csv",index=False)
    selected=rows[rows.fold.isin(status["completed_folds"])] if not rows.empty else rows
    lines=[f"{'FINAL' if status['complete'] else 'PRELIMINARY'}: {status['successful_evaluations']}/120 successful evaluations.",
        f"Means include fully completed folds only: {status['completed_folds']}.",
        "Metric: R2 on log salary; original RMSE and result provenance are preserved."]
    if not selected.empty:
        evaluation_report(selected).to_csv(output/"evaluation_report.csv",index=False)
        verdict=curation_verdict(selected.assign(test_score=selected.r2))
        verdict.to_csv(output/("curation_verdict.csv" if status["complete"] else "preliminary_verdict.csv"),index=False)
        columns={"score_structured":"Structured","score_unstructured":"Text only","score_joint_frozen":"Joint frozen",
                 "score_joint_tar":"Joint TAR","delta_frozen_vs_best_unimodal":"Joint delta","delta_tar_vs_frozen":"TAR delta"}
        lines.append(verdict.set_index("model")[list(columns)].rename(columns=columns).to_string(float_format=lambda v:f"{v:.6f}"))
        joint,tar=int(verdict.joint_signal.sum()),int(verdict.tar_gain.sum())
        lines.append(f"Joint Signal: {joint}/5; TAR Gain: {tar}/5; counted independently, raw deltas > 0.")
        lines.append("Final verdict: "+("PASS" if joint>=3 and tar>=3 else "FAIL") if status["complete"] else "Final verdict pending all six folds.")
    if not status["complete"]:
        (output/"curation_verdict.csv").unlink(missing_ok=True)
    else:
        (output/"preliminary_verdict.csv").unlink(missing_ok=True)
    (output/"summary.txt").write_text("\n".join(line.rstrip() for line in "\n".join(lines).splitlines())+"\n",encoding="utf-8")
    print("\n".join(lines))
    return status


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",default=str(ROOT/"results/linkedin_batch256/linkedin_salary"))
    args=parser.parse_args()
    report(args.results)

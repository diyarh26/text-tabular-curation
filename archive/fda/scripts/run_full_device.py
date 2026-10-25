"""Full sequential FDA Device evaluation with multiclass audits and persistent TAR."""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path

from colab_tar_compare import write_json
from run_full_linkedin import full_tar, ACCUMULATION_LOSS
import subprocess
import time
import threading

DATA_SHA256_LF = "b005b5ecf1bd1026c649f5989861cc6b8d3338ac1d09e67fefdc59c3793a6ec7"
MODELS = ["tabm", "cat", "light", "tabpfnv2", "tabpfnv2p5"]
CONDITIONS = ["structured", "unstructured", "joint_frozen", "joint_tar"]


def text_digest(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def fold_dir(output, fold):
    return Path(output) / "folds" / f"fold{fold}"


def audit_runs(output, folds=range(6)):
    """Require every expected identity and finite multiclass metrics; negative log-loss scores are valid."""
    from tt_curation.conditions import MODELS as names
    problems = []
    for fold in folds:
        expected = {f"{m}_{c}_10k_fold{fold}.json" for m in MODELS for c in CONDITIONS}
        extras = {p.name for p in (fold_dir(output, fold) / "runs").glob("*.json")} - expected
        if extras:
            problems.append(f"fold{fold}: unexpected run files {sorted(extras)}")
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                label = f"fold{fold}/{path.name}"
                if not path.exists():
                    problems.append(f"{label}: missing")
                    continue
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                    finite = all(isinstance(result.get(k), (float, int)) and math.isfinite(result[k])
                                 for k in ("test_score", "test_error", "auroc"))
                    encoder = "e5-small-tar" if condition == "joint_tar" else (
                        None if condition == "structured" else "e5-small")
                    valid = (finite and result["test_error"] >= 0 and 0 <= result["auroc"] <= 1 and
                             math.isclose(result["test_score"], -result["test_error"], abs_tol=1e-8) and
                             result.get("metric") == "log_loss" and result.get("model") == names[model] and
                             result.get("dataset") == "fda_device_severity" and result.get("short_name") == model and
                             result.get("condition") == condition and result.get("fold") == fold and
                             result.get("task") == "multiclass" and result.get("size") == "10k" and
                             result.get("multabench_commit") == "ff0c9c8b40a2085a06a7771190e53a261341802f" and
                             result.get("n_train") == 10_000 and result.get("n_test") == 5_000 and
                             result.get("text_encoder") == encoder)
                    if result.get("status") != "ok":
                        problems.append(f"{label}: {result.get('error', result.get('status'))}")
                    elif not valid:
                        problems.append(f"{label}: unexpected metadata or multiclass metrics")
                except (ValueError, OSError, TypeError, KeyError) as exc:
                    problems.append(f"{label}: unreadable ({type(exc).__name__})")
    return problems


def collect_runs(output):
    import pandas as pd
    rows = []
    for fold in range(6):
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                if path.exists():
                    try:
                        rows.append(json.loads(path.read_text(encoding="utf-8")))
                    except ValueError:
                        pass  # Audit lists interrupted JSON writes as unreadable.
    return pd.DataFrame(rows)


def make_report(output):
    from tt_curation.criteria import curation_verdict, evaluation_report, summary_lines
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = audit_runs(output)
    runs = collect_runs(output)
    if not problems:
        runs.pivot_table(index="model", columns="condition", values="auroc", aggfunc="mean").to_csv(output / "supplementary_macro_ovr_auroc.csv")
    if not runs.empty and (runs.status == "ok").any():
        evaluation_report(runs).to_csv(output / "evaluation_report.csv", index=False)
    if problems:
        text = f"INCOMPLETE: {len(problems)} of 120 expected evaluations missing/failed/invalid.\n"
        text += "No final curation verdict. Resume the same experiment.\n" + "\n".join(problems)
        # A resumed partial report must never retain a stale final verdict.
        (output / "curation_verdict.csv").unlink(missing_ok=True)
    else:
        verdict = curation_verdict(runs)
        verdict.to_csv(output / "curation_verdict.csv", index=False)
        text = "COMPLETE: all 120 evaluations passed the audit.\n" + "\n".join(summary_lines(verdict, runs))
    (output / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return not problems



def validate_resume(output, fold):
    # Never let the runner skip an invalid status-ok file or silently retry an error.
    for path in (fold_dir(output, fold) / "runs").glob("*.json"):
        issues = audit_runs(output, [fold])
        own = [i for i in issues if path.name in i]
        if own:
            raise ValueError("Invalid existing result; inspect before resuming: " + str(own))


def save_fold_report(output, fold):
    from tt_curation.criteria import curation_verdict
    runs = collect_runs(output)
    runs = runs[runs.fold == fold]
    destination = fold_dir(output, fold)
    curation_verdict(runs).to_csv(destination / "preliminary_scores_and_deltas.csv", index=False)
    runs.pivot(index="short_name", columns="condition", values="auroc").to_csv(destination / "supplementary_macro_ovr_auroc.csv")
    timing = json.loads((destination / "timing.json").read_text())
    tar = runs[runs.condition == "joint_tar"].iloc[0].tar_finetune
    write_json(destination / "preliminary_summary.json", {"preliminary": True, "fold": fold,
        "tar": tar, **timing, "estimated_remaining_seconds": timing["wall_seconds"] * (5-fold)})

def verify_prepared_data(root):
    import pandas as pd
    import yaml
    import pathlib
    expected={'data/processed/fda_device_severity.csv':'b005b5ecf1bd1026c649f5989861cc6b8d3338ac1d09e67fefdc59c3793a6ec7','configs/fda_device_severity.yaml':'34442d58c3d3f2803612aa9810718aa229e63bc3ec4f25ec51278b7d144eb590','data/provenance/fda_device_severity/selected_rows.csv':'834509665ae2d960e4b7ebac6da8891c39236e64f52d20776e64314a4bdebc10','data/provenance/fda_device_severity/text_changes.json':'8bb70c197895b322e6c07f3c74c2a930ddeac45866e6e2f303fe1bf8775fbf8a'}
    for p,h in expected.items():
     assert hashlib.sha256((root / p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,p
     print('SHA256 PASS',p)
    df=pd.read_csv(root / 'data/processed/fda_device_severity.csv'); trace=pd.read_csv(root / 'data/provenance/fda_device_severity/selected_rows.csv'); cfg=yaml.safe_load((root / 'configs/fda_device_severity.yaml').read_text())
    assert len(df)==15000 and df.classification.value_counts().to_dict()=={'Class I':988,'Class II':13426,'Class III':586}
    assert trace.event_id.nunique()==15000 and trace.classification.equals(df.classification)
    assert list(df.columns)==cfg['text_columns']+cfg['categorical_columns']+cfg['numerical_columns']+[cfg['target']] or set(df.columns)==set(cfg['text_columns']+cfg['categorical_columns']+cfg['numerical_columns']+[cfg['target']])
    changes=json.loads((root / 'data/provenance/fda_device_severity/text_changes.json').read_text()); assert len(changes)==12
    for c in changes: assert df.loc[c['output_row'],c['field']]==c['after']
    print('Rows/classes/provenance/text edits PASS'); print(df.isna().sum().to_string())


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--micro_batch", type=int, choices=[256], default=256,
                        help="Official default 256; smaller micro-batches require an approved memory adjustment.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare_only", action="store_true")
    mode.add_argument("--report_only", action="store_true")
    mode.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    if not folds or set(folds) - set(range(6)):
        parser.error("folds must be selected from 0..5")
    output = Path(args.results_root).resolve() / "fda_device_severity"
    if args.report_only:
        make_report(output)
        return
    if args.audit_only:
        problems = audit_runs(output)
        if problems:
            raise SystemExit("\n".join(problems))
        print("PASS: all 120 Device evaluations audited.")
        return

    from dotenv import load_dotenv
    from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
    load_dotenv(REPO_ROOT / ".env")
    from multabench.e5.constants import E5_LORA, E5_SMALL_V2
    from tt_curation.conditions import Condition, MODELS as upstream_models
    from tt_curation.data import load_dataset, describe
    from tt_curation.spec import load_spec

    assert set(MODELS) == set(upstream_models) and CONDITIONS == [str(c) for c in Condition]
    verify_prepared_data(REPO_ROOT)
    config = REPO_ROOT / "configs/fda_device_severity.yaml"
    spec = load_spec(config)
    if text_digest(spec.csv) != DATA_SHA256_LF:
        raise ValueError("Device CSV differs from the agreed 15K sample. Do not resample.")
    data = load_dataset(spec)
    if len(data.y) != 15_000 or not spec.is_cls or data.y.nunique() != 3:
        raise ValueError("Expected 15K Device multiclass rows")
    paths = [*sorted((REPO_ROOT / "src/tt_curation").glob("*.py")), Path(__file__).resolve(),
             Path(__file__).resolve().with_name("colab_tar_compare.py"),
             Path(__file__).resolve().with_name("run_full_linkedin.py")]
    paths += [REPO_ROOT / "requirements/constraints.txt"]
    from importlib.metadata import distribution
    installed_source = json.loads(distribution("multabench").read_text("direct_url.json"))
    if installed_source.get("vcs_info", {}).get("commit_id") != MULTABENCH_COMMIT:
        raise ValueError("Installed MulTaBench commit mismatch")
    settings = dict(dataset=spec.name, csv_sha256_lf=DATA_SHA256_LF, config_sha256_lf=text_digest(config),
                    multabench_commit=MULTABENCH_COMMIT, models=MODELS, conditions=CONDITIONS,
                    folds=list(range(6)), expected_evaluations=120, size="10k", frozen_encoder="e5-small",
                    source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                    tar_settings=E5_LORA.to_dict(), encoder_model=E5_SMALL_V2, checkpoint_version="persistent_epoch_adapter_restore_callbacks_v1",
                    tar_max_epochs=50, micro_batch=args.micro_batch, effective_batch=256, precision="float32",
                    accumulation_loss=ACCUMULATION_LOSS if args.micro_batch < 256 else "reference_single_batch_mean",
                    restore_early_stopping_state=True,
                    code={p.relative_to(REPO_ROOT).as_posix(): text_digest(p) for p in paths},
                    packages={n: version(n) for n in ["multabench", "torch", "transformers", "peft", "catboost",
                                                    "lightgbm", "pytabkit", "tabpfn", "scikit-learn", "pandas",
                                                    "numpy", "skrub"]})
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Saved experiment settings differ. Do not mix results or checkpoints.")
    elif args.prepare_only and (not output.exists() or not any(output.iterdir())):
        write_json(manifest, settings)
    else:
        raise ValueError("Initialize an empty output using --prepare_only before launching workers.")
    print("\n".join(describe(data)), flush=True)
    if args.prepare_only:
        print("Manifest verified. Workers can now start in separate fold directories.", flush=True)
        return

    import torch
    from tt_curation.runner import run_all
    import tt_curation.runner as runner
    original_run_one = runner.run_one
    def checked_run_one(*a, **kw):
        result = original_run_one(*a, **kw)
        if result["status"] != "ok":
            destination = fold_dir(output, result["fold"]) / "runs" / f'{result["short_name"]}_{result["condition"]}_10k_fold{result["fold"]}.json'
            write_json(destination, result)
            raise RuntimeError(result["error"])
        return result
    runner.run_one = checked_run_one
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one GPU. Set CUDA_VISIBLE_DEVICES before starting Python.")
    print(f"Assigned GPU: {torch.cuda.get_device_name(0)}; folds {folds}; micro-batch {args.micro_batch}, "
          "effective batch 256, cap 50.", flush=True)
    for fold in folds:
        destination = fold_dir(output, fold)
        validate_resume(output, fold)
        started = time.time()
        samples = []
        stop = threading.Event()
        def sample():
            while not stop.is_set():
                try:
                    samples.append(int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()))
                except Exception:
                    pass
                stop.wait(2)
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        with full_tar(destination / "checkpoints", micro_batch=args.micro_batch):
            run_all(data, destination, models=MODELS, conditions=list(Condition), folds=[fold],
                    size="10k", device="cuda", frozen_encoder="e5-small")
        stop.set()
        sampler.join()
        write_json(destination / "timing.json", {"wall_seconds": time.time()-started, "sampled_peak_gpu_mib": max(samples, default=0), "sampling_interval_s": 2})
        problems = audit_runs(output, [fold])
        if problems:
            raise RuntimeError("Fold incomplete; saved work can be resumed.\n" + "\n".join(problems))
        save_fold_report(output, fold)
        print(f"Fold {fold}: all 20 evaluations audited successfully.", flush=True)

    if not make_report(output):
        raise RuntimeError("Final audit failed")


if __name__ == "__main__":
    main()

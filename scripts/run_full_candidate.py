"""Resumable full-budget LendingClub/FDA Food workers; one visible GPU per process."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib.metadata import distribution, version
import json
import math
import os
from pathlib import Path
import re
import time

from colab_tar_compare import write_json
from run_full_linkedin import ACCUMULATION_LOSS, CONDITIONS, MODELS, full_tar, text_digest


@dataclass(frozen=True)
class Candidate:
    key: str
    name: str
    title: str
    run_name: str
    rows: int
    sha256: str
    counts: dict[str, int]
    structured: int
    metric: str

    @property
    def task(self):
        return "binary" if len(self.counts) == 2 else "multiclass"


CANDIDATES = {
    "lendingclub": Candidate("lendingclub", "lendingclub_longdesc_no_purpose", "LendingClub",
        "lendingclub_full", 15_000, "46d3b72cefc9634840fc5556b80deb860f7d07f956bbc3f00f0363ead8d9af24",
        {"0": 12_822, "1": 2_178}, 9, "roc_auc"),
    "food": Candidate("food", "fda_food_severity", "FDA Food", "fda_food_full", 7_918,
        "0a8ff4ce3ff48fd2beb16edb9ff197b4f38fc026a87cc82308be220baaee539d",
        {"Class I": 3_182, "Class II": 4_065, "Class III": 671}, 8, "log_loss"),
}


def fold_dir(output, fold):
    return Path(output) / "folds" / f"fold{fold}"


def verified_data(candidate):
    """Never silently substitute an earlier LendingClub sample or change FDA missing-value rules."""
    from tt_curation import REPO_ROOT
    from tt_curation.data import load_dataset
    from tt_curation.spec import load_spec
    config = REPO_ROOT / "configs" / f"{candidate.name}.yaml"
    spec = load_spec(config)
    if text_digest(spec.csv) != candidate.sha256 or spec.name != candidate.name or not spec.is_cls:
        raise ValueError("Data/config differs from the agreed prepared candidate. Do not resample.")
    data = load_dataset(spec)
    if (len(data.y) != candidate.rows or data.y.value_counts().to_dict() != candidate.counts or
            len(data.text_columns) != 3 or len(data.structured_columns) != candidate.structured):
        raise ValueError("Prepared sample counts or feature types differ")
    if candidate.key == "lendingclub":
        if data.x.isna().any().any() or "purpose" in data.x or "title" not in data.text_columns:
            raise ValueError("Expected complete LendingClub inputs, title retained, purpose removed")
        prefix = r"\bBorrower\s+added\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\s*>?"
        if not data.x.desc.str.replace(prefix, " ", regex=True, flags=re.I).str.split().str.len().gt(50).all():
            raise ValueError("Expected strictly more than 50 description content words")
    return data, config


def official_splits(data):
    """Record the actual pinned splits, including Food's smaller and unequal fold sizes."""
    import numpy as np
    from multabench.benchmark.splits import get_split
    result = {}
    for fold in range(6):
        train, test = get_split(data.y.to_numpy(), is_cls=True, split=fold, size="10k")
        if set(train) & set(test) or len(train) + len(test) != len(data.y):
            raise ValueError("Unexpected outer split")
        if set(data.y.iloc[train]) != set(data.y) or set(data.y.iloc[test]) != set(data.y):
            raise ValueError("An outer fold is missing a class")
        result[str(fold)] = dict(n_train=len(train), n_test=len(test),
            train_indices_sha256=hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
            test_indices_sha256=hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest())
    return result


def experiment_settings(candidate, data, config, micro_batch):
    from tt_curation import MULTABENCH_COMMIT, REPO_ROOT
    direct = distribution("multabench").read_text("direct_url.json")
    if not direct or json.loads(direct).get("vcs_info", {}).get("commit_id") != MULTABENCH_COMMIT:
        raise ValueError("Installed MulTaBench is not verified at the required Git commit")
    revision_path = REPO_ROOT / "SOURCE_REVISION"
    if revision_path.exists():
        revision = revision_path.read_text().strip()
    else:
        from tt_curation.runner import _git_commit
        revision = _git_commit()
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Cannot establish exact source revision")
    scripts = ["run_full_candidate.py", "kaggle_candidates.py", "run_full_linkedin.py",
               "colab_tar_compare.py", "kaggle_linkedin.py"]
    paths = [*sorted((REPO_ROOT / "src/tt_curation").glob("*.py")),
             *(REPO_ROOT / "scripts" / name for name in scripts)]
    return dict(protocol_version="candidate_full_v1", dataset=candidate.name, rows=candidate.rows,
        task=candidate.task, class_counts=candidate.counts, csv_sha256_lf=candidate.sha256,
        config_sha256_lf=text_digest(config), multabench_commit=MULTABENCH_COMMIT,
        source_revision=revision, models=MODELS, conditions=CONDITIONS, folds=list(range(6)),
        expected_evaluations=120, size="10k", metric=candidate.metric, frozen_encoder="e5-small",
        tar_max_epochs=50, micro_batch=micro_batch, effective_batch=256, precision="float32",
        accumulation_loss=ACCUMULATION_LOSS if micro_batch < 256 else "reference_single_batch_mean",
        restore_early_stopping_state=True, splits=official_splits(data),
        code={p.relative_to(REPO_ROOT).as_posix(): text_digest(p) for p in paths},
        packages={n: version(n) for n in ["multabench", "torch", "transformers", "accelerate", "peft",
            "catboost", "lightgbm", "pytabkit", "tabpfn", "scikit-learn", "pandas", "numpy", "skrub"]})


def initialize_manifest(output, settings, *, prepare=False):
    output = Path(output)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != settings:
            raise ValueError("Saved experiment settings differ. Do not mix results or checkpoints.")
    elif prepare and (not output.exists() or not any(output.iterdir())):
        write_json(manifest, settings)
    else:
        raise ValueError("Initialize an empty experiment with --prepare_only first")


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def audit_runs(output, candidate, folds=range(6)):
    from tt_curation import MULTABENCH_COMMIT
    from tt_curation.conditions import MODELS as names
    output = Path(output)
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["dataset"] != candidate.name or manifest["csv_sha256_lf"] != candidate.sha256 or
                manifest["rows"] != candidate.rows or manifest["metric"] != candidate.metric):
            return ["Manifest describes a different candidate"]
        splits = manifest["splits"]
    except (OSError, ValueError, KeyError, TypeError):
        return ["Missing or invalid experiment manifest"]
    problems = []
    for fold in folds:
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                label = f"fold{fold}/{path.name}"
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if row.get("status") != "ok":
                        problems.append(f"{label}: {row.get('error', row.get('status'))}")
                        continue
                    encoder = "e5-small-tar" if condition == "joint_tar" else (
                        None if condition == "structured" else "e5-small")
                    features = candidate.structured if condition == "structured" else (
                        3 if condition == "unstructured" else candidate.structured + 3)
                    identity = dict(dataset=candidate.name, model=names[model], short_name=model,
                        condition=condition, fold=fold, task=candidate.task, size="10k",
                        n_train=splits[str(fold)]["n_train"], n_test=splits[str(fold)]["n_test"],
                        n_features=features, text_encoder=encoder, metric=candidate.metric,
                        multabench_commit=MULTABENCH_COMMIT)
                    valid = all(row.get(k) == v for k, v in identity.items())
                    valid &= all(_finite(row.get(k)) for k in ("test_score", "test_error", "auroc", "runtime_s"))
                    if valid:
                        valid &= row["test_error"] >= 0 and row["runtime_s"] >= 0 and 0 <= row["auroc"] <= 1
                        if candidate.metric == "log_loss":
                            valid &= math.isclose(row["test_score"], -row["test_error"], abs_tol=1e-8)
                        else:
                            valid &= 0 <= row["test_score"] <= 1
                            valid &= math.isclose(row["test_score"], row["auroc"], abs_tol=1e-8)
                            valid &= math.isclose(row["test_error"], 1-row["test_score"], abs_tol=1e-8)
                    if not valid:
                        problems.append(f"{label}: unexpected metadata or metrics")
                except FileNotFoundError:
                    problems.append(f"{label}: missing")
                except (OSError, ValueError, TypeError, KeyError):
                    problems.append(f"{label}: unreadable or invalid")
    return problems


def collect_runs(output, folds=range(6)):
    import pandas as pd
    rows = []
    for fold in folds:
        for model in MODELS:
            for condition in CONDITIONS:
                path = fold_dir(output, fold) / "runs" / f"{model}_{condition}_10k_fold{fold}.json"
                try:
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    pass
    return pd.DataFrame(rows)


def score_table(runs, value):
    from tt_curation.conditions import MODELS as names
    table = runs.pivot_table(index="model", columns="condition", values=value).reindex(list(names.values()))
    table = table[CONDITIONS].copy()
    table["Joint delta"] = table.joint_frozen - table[["structured", "unstructured"]].max(axis=1)
    table["TAR delta"] = table.joint_tar - table.joint_frozen
    return table


def progress_text(output, candidate, folds):
    """A live, preliminary comparison using only fully audited folds, never partial baselines."""
    runs = collect_runs(output, folds)
    table = score_table(runs, "test_score")
    text = [f"{candidate.title} | completed folds {list(folds)} | {20*len(folds)}/120 evaluations in this table",
            "PRELIMINARY until all six folds complete. No performance-based early exit.",
            "Official metric: " + ("AUROC (higher is better)" if candidate.metric == "roc_auc" else
                                  "NEGATIVE LOG LOSS (higher is better; closer to zero)"),
            table.to_string(float_format=lambda x: f"{x:.6f}")]
    joint, tar = int(table["Joint delta"].gt(0).sum()), int(table["TAR delta"].gt(0).sum())
    text += [f"Joint Signal: {joint}/5 | TAR Gain: {tar}/5 | need >=3/5 independently on six-fold means.",
             "Joint delta = joint frozen - better unimodal; TAR delta = joint TAR - joint frozen."]
    if candidate.metric == "log_loss":
        text += ["Supplementary macro OvR AUROC (not the official log-loss verdict):",
                 score_table(runs, "auroc").to_string(float_format=lambda x: f"{x:.6f}")]
    return "\n".join(text) + "\n"


def save_fold_report(output, candidate, fold, elapsed):
    output = Path(output)
    text = f"===== FOLD {fold} COMPLETE: ALL 20 EVALUATIONS AUDITED =====\n"
    text += progress_text(output, candidate, [fold])
    text += f"Worker time for this fold in this session: {elapsed/60:.1f} minutes (may include reused runs).\n"
    rows = collect_runs(output, [fold])
    stats = next((r for r in rows.tar_finetune if isinstance(r, dict)), {}) if "tar_finetune" in rows else {}
    if stats:
        text += "TAR training metadata: " + json.dumps(stats, default=str) + "\n"
    score_table(rows, "test_score").to_csv(fold_dir(output, fold) / "scores.csv")
    write_json(fold_dir(output, fold) / "timing.json", dict(fold=fold, session_elapsed_seconds=elapsed, tar=stats))
    destination = output / f"fold{fold}_report.txt"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(destination)  # Supervisor prints only a complete, atomically published table.
    print(text, flush=True)


def make_report(output, candidate):
    from tt_curation.criteria import curation_verdict, evaluation_report, summary_lines
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    problems = audit_runs(output, candidate)
    runs = collect_runs(output)
    if not runs.empty and (runs.status == "ok").any():
        evaluation_report(runs).to_csv(output / "evaluation_report.csv", index=False)
    if problems:
        text = f"INCOMPLETE: {len(problems)} audit problems. No final curation verdict.\n" + "\n".join(problems)
        (output / "curation_verdict.csv").unlink(missing_ok=True)
    else:
        verdict = curation_verdict(runs)
        verdict.to_csv(output / "curation_verdict.csv", index=False)
        text = "COMPLETE: all 120 evaluations audited.\n" + "\n".join(summary_lines(verdict, runs))
        if candidate.metric == "log_loss":
            text += "\nOfficial scores/deltas are negative log loss; macro AUROC is supplementary.\n"
    (output / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return not problems


def main():
    os.environ["MPLBACKEND"] = "Agg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CANDIDATES, required=True)
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4,5")
    parser.add_argument("--micro_batch", type=int, choices=[16, 32, 256], default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare_only", action="store_true")
    mode.add_argument("--report_only", action="store_true")
    mode.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    candidate = CANDIDATES[args.dataset]
    output = Path(args.results_root).resolve() / candidate.name
    folds = list(dict.fromkeys(int(f) for f in args.folds.split(",")))
    if not folds or set(folds) - set(range(6)):
        parser.error("folds must be selected from 0..5")
    if args.report_only:
        raise SystemExit(0 if make_report(output, candidate) else 1)
    if args.audit_only:
        problems = audit_runs(output, candidate)
        if problems:
            raise SystemExit("\n".join(problems))
        print("PASS: all 120 evaluations audited.", flush=True)
        return
    from dotenv import load_dotenv
    from tt_curation import REPO_ROOT
    load_dotenv(REPO_ROOT / ".env")
    from tt_curation.conditions import Condition
    from tt_curation.data import describe
    data, config = verified_data(candidate)
    settings = experiment_settings(candidate, data, config, args.micro_batch)
    initialize_manifest(output, settings, prepare=args.prepare_only)
    print("\n".join(describe(data)), flush=True)
    if args.prepare_only:
        print("Manifest and all six official splits verified. Ready for workers.", flush=True)
        return
    import torch
    from tt_curation.runner import run_all
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Each worker must see exactly one CUDA GPU")
    print(f"Assigned GPU: {torch.cuda.get_device_name(0)}; folds {folds}; micro batch {args.micro_batch}; "
          "effective batch 256; cap 50 with early stopping.", flush=True)
    for fold in folds:
        started = time.monotonic()
        with full_tar(fold_dir(output, fold) / "checkpoints", micro_batch=args.micro_batch):
            run_all(data, fold_dir(output, fold), models=MODELS, conditions=list(Condition),
                    folds=[fold], size="10k", device="cuda", frozen_encoder="e5-small")
        problems = audit_runs(output, candidate, [fold])
        if problems:
            raise RuntimeError("Fold incomplete; resume saved work.\n" + "\n".join(problems))
        save_fold_report(output, candidate, fold, time.monotonic()-started)


if __name__ == "__main__":
    main()

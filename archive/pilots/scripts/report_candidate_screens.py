"""Audit all 54 approved candidate-screen evaluations and compare both FDA metrics.

This only reads completed results; it never fits a model or changes experiment settings.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "lendingclub_default": "LendingClub",
    "fda_device_severity": "FDA Device",
    "fda_food_severity": "FDA Food",
}
MODELS = {"cat": "CatBoost", "light": "LightGBM"}
CONDITIONS = ["structured", "unstructured", "joint_frozen"]
FOLDS = [0, 1, 2]


def audited_runs(name):
    output = ROOT / "results" / f"{name}_screen_tfidf"
    manifest = json.loads((output / "screen_manifest.json").read_text())
    csv = ROOT / "data" / "processed" / f"{name}.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == manifest["csv_sha256"], f"Changed data: {name}"
    config = ROOT / "configs" / f"{name}.yaml"
    assert hashlib.sha256(config.read_bytes()).hexdigest() == manifest["config_sha256"], f"Changed config: {name}"
    for file, expected_sha in manifest["pipeline_files_sha256"].items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected_sha, f"Changed runner: {file}"
    files = sorted((output / "runs").glob("*.json"))
    runs = [json.loads(p.read_text()) for p in files]
    expected = set(itertools.product(MODELS, CONDITIONS, FOLDS))
    keys = [(r["short_name"], r["condition"], r["fold"]) for r in runs]
    assert len(keys) == len(expected) and set(keys) == expected, f"Incomplete grid: {name}: {len(keys)}/18"
    assert max(Counter(keys).values()) == 1, f"Duplicate evaluations: {name}"
    for run in runs:
        assert run["status"] == "ok", f"Failed evaluation: {name}: {run.get('error')}"
        assert run["device"] == "cpu", run
        assert run["dataset"] == name and run["size"] == "10k", run
        assert run["multabench_commit"] == manifest["multabench_commit"], run
        if run["condition"] != "structured":
            assert run["text_encoder"] == "tfidf", run
        assert 0 <= run["auroc"] <= 1, run
        assert math.isfinite(run["test_score"]), run
        assert run["n_train"] <= 10_000 and run["n_test"] <= 5_000, run
        assert run["n_train"] + run["n_test"] == manifest["selected_rows"], run
    return pd.DataFrame(runs)


def comparisons(runs, name, metric, column):
    rows = []
    for short_name, label in MODELS.items():
        group = runs[runs.short_name == short_name]
        folded = group.pivot(index="fold", columns="condition", values=column).loc[FOLDS, CONDITIONS]
        means, std = folded.mean(), folded.std()
        wins = (folded.joint_frozen > folded.structured) & (folded.joint_frozen > folded.unstructured)
        rows.append({
            "dataset": name, "dataset_label": DATASETS[name], "learner": label, "metric": metric,
            "n_rows": json.loads((ROOT / "data" / "processed" / f"{name}.summary.json").read_text())["counts"]["selected_rows"],
            "structured": means.structured, "text_only": means.unstructured, "joint": means.joint_frozen,
            "delta_joint_vs_best_unimodal": means.joint_frozen - max(means.structured, means.unstructured),
            "structured_std": std.structured, "text_only_std": std.unstructured, "joint_std": std.joint_frozen,
            "joint_fold_wins": int(wins.sum()), "n_folds": len(folded),
        })
    return rows


def markdown_table(rows):
    lines = ["| Dataset | Learner | Structured | Text TF-IDF | Joint TF-IDF | Joint delta | Fold wins |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset_label']} | {r['learner']} | {r['structured']:.4f} | {r['text_only']:.4f} | "
                     f"{r['joint']:.4f} | {r['delta_joint_vs_best_unimodal']:+.6f} | {r['joint_fold_wins']}/{r['n_folds']} |")
    return lines


def main():
    rows, audit = [], []
    for name in DATASETS:
        runs = audited_runs(name)
        metric = "binary AUROC" if name == "lendingclub_default" else "macro one-vs-rest AUROC"
        rows.extend(comparisons(runs, name, metric, "auroc"))
        if name != "lendingclub_default":
            rows.extend(comparisons(runs, name, "negative log loss", "test_score"))
        audit.append({"dataset": name, "expected": 18, "successful": len(runs), "failed": 0,
                      "learner_runtime_seconds": float(runs.runtime_s.sum())})
    output = ROOT / "results"
    pd.DataFrame(rows).to_csv(output / "new_candidate_screen_comparison.csv", index=False)
    lines = ["# New candidate CPU screening results", "",
             "All 54 evaluations succeeded: 3 datasets x 2 learners x 3 conditions x 3 folds.", "",
             "Actual pinned MulTaBench wrappers, preprocessing, hyperparameters and outer splits. Text uses TF-IDF; no E5 or TAR.", "",
             "The joint delta is the unrounded mean joint score minus the better unrounded unimodal mean score. Fold wins require beating both unimodal scores within a fold.", "",
             "## AUROC", "",
             "Binary AUROC for LendingClub; macro one-vs-rest AUROC for the three-class FDA tasks. These measure different prediction problems, so raw gains are not a universal cross-dataset ranking.", ""]
    lines.extend(markdown_table([r for r in rows if r["metric"] != "negative log loss"]))
    lines.extend(["", "## FDA official multiclass metric: negative log loss", "",
                  "Higher is better. A positive joint delta is a reduction in log loss relative to the better unimodal model. Its scale is different from AUROC.", ""])
    lines.extend(markdown_table([r for r in rows if r["metric"] == "negative log loss"]))
    lines.extend(["", "## Prepared data and interpretation", "",
                  "- LendingClub: 15,000 resolved accepted loans with descriptions; 3 text and 10 structured features. The original missing-value rule and sample are retained.",
                  "- FDA Device: 15,000 events; FDA Food: 7,918 events. One randomly selected product per event, original three severity classes, 3 text and 8 structured features.",
                  "- Explicit FDA recall-severity text has been masked, and the audited pasted administrative template removed. Legitimate device regulatory classes remain.",
                  "- Seed 42; no class balancing, minimum word count or text truncation.",
                  "- This screen measures possible joint signal. It does not establish TAR gain or final acceptance with the five-learner GPU pipeline.", ""])
    (output / "new_candidate_screen_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "new_candidate_screen_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

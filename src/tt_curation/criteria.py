"""Course curation criteria, following staff forum clarifications supplied on 2026-10-23.

Use unrounded mean test scores over folds (MulTaBench's metric, higher is better):
  1. Joint signal: joint_frozen - max(structured, unstructured) > 0
  2. TAR gain:     joint_tar - joint_frozen > 0
Each criterion needs at least 3 of the 5 learners, counted independently; the learners may differ.
This course rule supersedes the paper analysis's rounded-score convention used by earlier reports.
"""
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from tt_curation.conditions import Condition

MIN_PASSING_LEARNERS = 3
METRIC_COLUMNS = ["test_score", "test_error", "auroc", "r2"]


def load_runs(results_dir: Path) -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted((results_dir / "runs").glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"No runs in {results_dir / 'runs'}")
    return pd.DataFrame(rows)


def evaluation_report(runs: pd.DataFrame) -> pd.DataFrame:
    """The deliverable table: one row per (learner, condition), mean and std over the successful folds."""
    ok = runs[runs["status"] == "ok"]
    metrics = [c for c in METRIC_COLUMNS if c in ok.columns and ok[c].notna().any()]
    grouped = ok.groupby(["dataset", "model", "condition"])
    report = grouped[metrics].agg(["mean", "std"])
    report.columns = [f"{m}_{s}" for m, s in report.columns]
    report["n_folds"] = grouped.size()
    report["metric"] = grouped["metric"].first()
    failed = runs[runs["status"] != "ok"].groupby(["dataset", "model", "condition"]).size()
    report["n_failed"] = failed.reindex(report.index).fillna(0).astype(int)
    report = report.reset_index()
    order = {c.value: i for i, c in enumerate(Condition)}
    return report.sort_values(["model", "condition"], key=lambda s: s.map(order) if s.name == "condition" else s)


def curation_verdict(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-learner raw means/deltas and fold wins. 'passes' diagnoses both on that learner only;
    dataset acceptance counts joint_signal and tar_gain independently, not this diagnostic column.
    """
    ok = runs[runs["status"] == "ok"]
    rows = []
    for model, group in ok.groupby("model"):
        per_fold = group.pivot_table(index="fold", columns="condition", values="test_score")
        mean = per_fold.mean()
        row = {"model": model, **{f"score_{c.value}": mean.get(c.value, np.nan) for c in Condition}}
        row["folds"] = len(per_fold)
        s, u, f, t = (per_fold.get(c.value) for c in Condition)
        if s is not None and u is not None and f is not None:
            delta = float(mean[Condition.JOINT_FROZEN] - max(mean[Condition.STRUCTURED], mean[Condition.UNSTRUCTURED]))
            row["delta_frozen_vs_best_unimodal"] = delta
            row["joint_signal"] = delta > 0
            row["joint_signal_fold_wins"] = int(((f > s) & (f > u)).sum())
        if f is not None and t is not None:
            delta = float(mean[Condition.JOINT_TAR] - mean[Condition.JOINT_FROZEN])
            row["delta_tar_vs_frozen"] = delta
            row["tar_gain"] = delta > 0
            row["tar_gain_fold_wins"] = int((t > f).sum())
        row["passes"] = bool(row.get("joint_signal", False) and row.get("tar_gain", False))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_lines(verdict: pd.DataFrame, runs: pd.DataFrame) -> List[str]:
    n_joint = int(verdict.get("joint_signal", pd.Series(dtype=bool)).fillna(False).sum())
    n_tar = int(verdict.get("tar_gain", pd.Series(dtype=bool)).fillna(False).sum())
    lines = ["Decision uses unrounded fold-mean test scores, strict improvement (> 0), and independent criterion counts.",
             f"{'learner':<13}{'struct':>11}{'text':>11}{'frozen':>11}{'TAR':>11}  "
             f"{'joint delta':>13} {'TAR delta':>13}  joint  TAR-gain  folds"]
    for _, r in verdict.iterrows():
        def mark(key):
            return "  ?  " if pd.isna(r.get(key)) else ("  ✅ " if r[key] else "  ❌ ")
        lines.append(f"{r['model']:<13}" + "".join(f"{r[f'score_{c.value}']:>11.6f}" for c in Condition)
                     + f"  {r.get('delta_frozen_vs_best_unimodal', np.nan):>13.9f}"
                     + f" {r.get('delta_tar_vs_frozen', np.nan):>13.9f}"
                     + f"  {mark('joint_signal')}  {mark('tar_gain')}    {r['folds']}")
    n_failed = int((runs["status"] != "ok").sum())
    passed = n_joint >= MIN_PASSING_LEARNERS and n_tar >= MIN_PASSING_LEARNERS
    lines.append(f"Joint Signal: {n_joint}/{len(verdict)} learners; needs {MIN_PASSING_LEARNERS} of 5.")
    lines.append(f"TAR Gain: {n_tar}/{len(verdict)} learners; needs {MIN_PASSING_LEARNERS} of 5.")
    verdict_text = "PASSES" if passed else "does NOT pass (yet)"
    lines.append(f"Dataset {verdict_text}: each criterion must pass independently; the qualifying learners may differ.")
    if n_failed:
        lines.append(f"⚠️ {n_failed} runs failed; see their JSONs")
    return lines

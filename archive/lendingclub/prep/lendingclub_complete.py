"""Filter the full accepted-loan source for complete chosen inputs, then sample 15K.

python prep/lendingclub_complete.py

Creates a new dataset/config, retaining the original LendingClub sample and all results.
The full cleaned pool and row identifiers stay under the ignored data/raw directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import candidate_screens as prep

NAME = "lendingclub_default_complete"
FEATURES = prep.LENDING_TEXT + prep.LENDING_CAT + prep.LENDING_NUM
TARGET = "charged_off"
LABELS = ["Fully Paid", "Charged Off"]


def status_counts(values):
    return {str(k): int(v) for k, v in values.value_counts().items()}


def distribution(stage, counts):
    total = sum(counts.values())
    return {"stage": stage, "rows": total,
            "class_counts": counts,
            "class_percentages": {k: 100 * v / total for k, v in counts.items()}}


def write_config():
    config = f"""# Complete cases from the full accepted-loan source, selected before random sampling.
name: {NAME}
csv: data/processed/{NAME}.csv
target: charged_off
task: classification

text_columns: [desc, title, emp_title]
categorical_columns: [term, emp_length, home_ownership, purpose, grade]
numerical_columns: [loan_amnt, annual_inc, dti, fico_score, int_rate]
"""
    (prep.ROOT / "configs" / f"{NAME}.yaml").write_text(config, encoding="utf-8")


def write_report(summary, stages, original_statuses):
    lines = ["# Complete-case LendingClub preparation", "",
             "A separate dataset. Filtering was applied to the full accepted-loan source before sampling. No models were fitted for this new version.", "",
             "## Filters, in order", "",
             "1. Read the accepted-loan table once; its compressed duplicate and the rejected-applicant table are not added.",
             "2. Keep only Fully Paid and Charged Off loans. Exclude CSV footer rows and all other statuses.",
             "3. Remove HTML markup, decode entities and normalize text whitespace. Require a nonempty description.",
             "4. Derive origination FICO as the mean of fico_range_low/high and parse numeric inputs before checking completeness.",
             "5. Require every one of the 13 chosen inputs and the target to be nonmissing; numeric values must be finite. Retain the valid home_ownership code NONE.",
             "6. Randomly select 15,000 eligible loans without replacement, seed 42. No class balancing or minimum word count.", "",
             "Completeness applies to the chosen 3 text and 10 structured inputs, not the 151 original columns. No values are imputed.", "",
             "Text: " + ", ".join(prep.LENDING_TEXT) + ".", "",
             "Categorical: " + ", ".join(prep.LENDING_CAT) + ".", "",
             "Numerical: " + ", ".join(prep.LENDING_NUM) + ".", "",
             "## Original accepted loans: all original statuses", "",
             "The denominator excludes the 33 nondata CSV footer rows. Rejected applications have no repayment outcome and are a separate table.", "",
             "| Status | Count | Percentage of original accepted loans |", "|---|---:|---:|"]
    original_total = sum(original_statuses.values())
    for label, count in sorted(original_statuses.items(), key=lambda item: -item[1]):
        lines.append(f"| {label} | {count:,} | {100 * count / original_total:.4f}% |")
    lines.extend([f"| **Total** | **{original_total:,}** | **100%** |", "",
                  "## Binary outcome comparison", "",
                  "The original resolved-loan row below uses only the same two target classes, before description/completeness filters. Original all-status percentages above use a different denominator.", "",
                  "| Stage | Rows | Fully Paid count | Fully Paid % | Charged Off count | Charged Off % |",
                  "|---|---:|---:|---:|---:|---:|"])
    for stage in stages:
        counts, percentages = stage["class_counts"], stage["class_percentages"]
        lines.append(f"| {stage['stage']} | {stage['rows']:,} | {counts.get('Fully Paid', 0):,} | "
                     f"{percentages.get('Fully Paid', 0):.4f}% | {counts.get('Charged Off', 0):,} | "
                     f"{percentages.get('Charged Off', 0):.4f}% |")
    lines.extend(["", "## Checks and outputs", "",
                  "- The full cleaned pool and the new 15K sample have zero missing values in every chosen input and target.",
                  "- Selected loan IDs are unique; identifiers are stored separately and excluded from model inputs.",
                  "- The new sample contains loans drawn from the full clean pool, not just the previous sample's complete rows.",
                  "- The previous LendingClub dataset and its model results are byte-for-byte unchanged.",
                  f"- New sample: `{summary['csv']}`.",
                  f"- Full clean pool: `{summary['full_clean_pool']['csv']}` (local, ignored by Git).",
                  f"- Config: `configs/{NAME}.yaml`.",
                  f"- Sample SHA256: `{summary['csv_sha256']}`.", ""])
    (prep.ROOT / "docs" / "LENDINGCLUB_COMPLETE_PREPARATION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads" / "archive (5).zip")
    args = parser.parse_args()
    prep.OUT.mkdir(parents=True, exist_ok=True)
    prep.TRACE.mkdir(parents=True, exist_ok=True)
    protected = [prep.OUT / "lendingclub_default.csv", prep.OUT / "lendingclub_default.summary.json",
                 prep.ROOT / "configs" / "lendingclub_default.yaml"]
    protected.extend(p for p in (prep.ROOT / "results" / "lendingclub_default_screen_tfidf").rglob("*") if p.is_file())
    before = {p: prep.digest(p) for p in protected if p.exists()}

    frame, member, counts = prep.lending_pool(args.archive)
    original_statuses = counts.pop("original_loan_status_counts")
    description_counts = status_counts(frame.loan_status)
    frame["fico_score"] = (pd.to_numeric(frame.fico_range_low, errors="coerce")
                           + pd.to_numeric(frame.fico_range_high, errors="coerce")) / 2
    for col in prep.LENDING_NUM:
        frame[col] = pd.to_numeric(frame[col].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
    for col in prep.LENDING_CAT:
        frame[col] = frame[col].map(prep.clean_category)
        # Keep the valid NONE ownership code, but treat placeholder values in other fields as missing.
        if col != "home_ownership":
            placeholder = frame[col].fillna("").str.lower().isin(prep.MISSING)
            frame.loc[placeholder, col] = None
    frame[TARGET] = frame.loan_status.map({"Fully Paid": 0, "Charged Off": 1})
    missing = frame[FEATURES + [TARGET]].isna().any(axis=1)
    nonfinite = ~np.isfinite(frame[prep.LENDING_NUM].to_numpy(dtype=float)).all(axis=1)
    keep = ~missing & ~nonfinite
    counts["description_pool_rows_with_missing_inputs"] = int(missing.sum())
    counts["description_pool_rows_with_nonfinite_values_only"] = int((nonfinite & ~missing).sum())
    clean = frame.loc[keep].reset_index(drop=True)
    if len(clean) < prep.MAX_ROWS:
        raise ValueError(f"Only {len(clean)} complete rows, fewer than the requested 15K")
    clean_model = clean[FEATURES + [TARGET]]
    assert not clean_model.isna().any().any()
    counts["full_clean_pool_rows"] = len(clean)
    counts["full_clean_pool_rows_removed_by_sampling"] = len(clean) - prep.MAX_ROWS

    pool_csv = prep.TRACE / f"{NAME}_pool.csv"
    pool_trace = prep.TRACE / f"{NAME}_pool.selected_rows.csv"
    clean_model.to_csv(pool_csv, index=False, lineterminator="\n")
    clean[["source_row", "id", "loan_status"]].to_csv(pool_trace, index=False, lineterminator="\n")
    chosen = prep.sample_rows(clean)
    trace = chosen[["source_row", "id", "loan_status"]].copy()
    model = chosen[FEATURES + [TARGET]].copy()
    assert trace.id.is_unique and len(model) == prep.MAX_ROWS
    assert not model.isna().any().any()
    summary = prep.save(NAME, model, trace, args.archive, member, counts,
                        prep.LENDING_TEXT, prep.LENDING_CAT, prep.LENDING_NUM, TARGET, {
                            "target": "0 = Fully Paid; 1 = Charged Off. Other statuses excluded.",
                            "row_selection": "Filter the FULL accepted source for resolved loans, nonempty desc and complete chosen inputs BEFORE the random 15K sample.",
                            "missing_fields": "Zero missing chosen inputs or targets; no imputation.",
                            "fico_score": "Mean of origination fico_range_low/high, derived before completeness filtering.",
                            "complete_case_filter_before_sampling": True,
                        })
    summary["preparation_script_sha256"] = prep.digest(__file__)
    summary["shared_preparation_helpers_sha256"] = prep.digest(prep.__file__)
    summary["full_clean_pool"] = {"csv": str(pool_csv.relative_to(prep.ROOT)), "rows": len(clean),
                                  "csv_sha256": prep.digest(pool_csv),
                                  "trace": str(pool_trace.relative_to(prep.ROOT)), "trace_sha256": prep.digest(pool_trace)}
    stages = [distribution("Original resolved loans: before feature filters", {label: original_statuses[label] for label in LABELS}),
              distribution("Resolved loans with a description", description_counts),
              distribution("Full clean pool: all chosen inputs complete", status_counts(clean.loan_status)),
              distribution("NEW complete-case random 15K sample", status_counts(chosen.loan_status))]
    old_trace = prep.TRACE / "lendingclub_default.selected_rows.csv"
    if old_trace.exists():
        stages.append(distribution("Previous 15K sample: missing values allowed", status_counts(pd.read_csv(old_trace).loan_status)))
    summary["original_status_distribution"] = distribution("Original accepted table: all statuses", original_statuses)
    summary["binary_class_distributions"] = stages
    for p, checksum in before.items():
        assert prep.digest(p) == checksum, f"Unexpected change to previous dataset/results: {p}"
    summary["previous_dataset_and_results_unchanged"] = True
    (prep.OUT / f"{NAME}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_config()

    from tt_curation.spec import load_spec
    from tt_curation.data import load_dataset
    loaded = load_dataset(load_spec(prep.ROOT / "configs" / f"{NAME}.yaml"))
    assert not loaded.x.isna().any().any() and not loaded.y.isna().any()
    assert len(loaded.x) == 15_000 and set(loaded.y) == {"0", "1"}
    write_report(summary, stages, original_statuses)
    print(json.dumps({"original": summary["original_status_distribution"], "binary_stages": stages,
                      "missing_cells_in_new_sample": int(model.isna().sum().sum())}, indent=2), flush=True)


if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()

"""Prepare the three agreed CPU-screen candidates; this script never trains models.

python prep/candidate_screens.py --downloads /path/to/downloads

Install requirements/prep_candidates.txt in addition to the project's base dependencies.
Original archives are read in place. Row identifiers are written separately under data/raw/,
never included in the model CSVs. FDA selection is uniform within each eligible event,
then a simple random sample of events if there are more than 15,000.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import random
import re
import sys
import zipfile

import ijson
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
TRACE = ROOT / "data" / "raw" / "candidate_screens"
SEED = 42
MAX_ROWS = 15_000
MIN_ROWS = 1_000
MISSING = {"", "null", "none", "nan", "n/a"}
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
CLASS_MENTION = re.compile(r"\bclass\s+(?:III|II|I|[123])\b", re.I)
RECALL_SEVERITY = re.compile(
    r"\bclass\s+(?:III|II|I|[123])\b(?=\s*,?\s*(?:voluntary\s+)?recall\b)", re.I)
RECLASSIFIED_RECALL = re.compile(
    r"(\bthis recall is now being reclassified by the Agency as a\s+)class\s+(?:III|II|I|[123])\b", re.I)
ADMIN_TEMPLATE = re.compile(
    r"\bPre-Review Reminders\b.*?\bDoc ID 03534\.05\.06 2 of 8\b", re.I | re.S)

LENDING_TEXT = ["desc", "title", "emp_title"]
LENDING_CAT = ["term", "emp_length", "home_ownership", "purpose", "grade"]
LENDING_NUM = ["loan_amnt", "annual_inc", "dti", "fico_score", "int_rate"]
FDA_TEXT = ["product_description", "reason_for_recall", "distribution_pattern"]
FDA_CAT = ["recalling_firm", "city", "state", "country", "voluntary_mandated", "initial_firm_notification"]
FDA_NUM = ["recall_year", "recall_month"]
FDA_CLASSES = {"Class I", "Class II", "Class III"}


def clean_string(value, *, text=False):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in MISSING:
        return None
    if text:
        # Strip markup before decoding entities so encoded comparisons remain literal text.
        value = html.unescape(HTML_TAG.sub(" ", value))
    value = " ".join(value.split())
    return value or None


def clean_category(value):
    # NONE is a real LendingClub home_ownership code, not a blank cell.
    if value is None or pd.isna(value):
        return None
    return " ".join(str(value).split()) or None


def remove_recall_leakage(value):
    """Remove audited recall-severity references, without reading the row's target.

    Device regulatory class, laser class and clinical class remain legitimate inputs.
    The narrow reclassification/template patterns cover the other audited source defects.
    """
    if value is None:
        return None
    value = ADMIN_TEMPLATE.sub(" ", value)
    value = RECALL_SEVERITY.sub("[recall severity]", value)
    value = RECLASSIFIED_RECALL.sub(lambda match: match.group(1) + "[recall severity]", value)
    return " ".join(value.split()) or None


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 2**20):
            hasher.update(block)
    return hasher.hexdigest()


def sample_rows(frame):
    # Shuffle even a smaller pool: ordering is never based on outcome or model performance.
    return frame.sample(n=min(MAX_ROWS, len(frame)), random_state=SEED).reset_index(drop=True)


def choose_event_records(rows, seed=SEED):
    """Reservoir sample one eligible product per event, without aggregating its severity."""
    rng = random.Random(seed)
    chosen, sizes = {}, Counter()
    classes = defaultdict(set)
    counts = Counter()
    recall_numbers = set()
    for index, raw in enumerate(rows):
        counts["source_rows"] += 1
        label = raw.get("classification")
        if label not in FDA_CLASSES:
            counts["excluded_unknown_severity"] += 1
            continue
        event = clean_string(raw.get("event_id"))
        if not event:
            counts["excluded_missing_event_id"] += 1
            continue
        recall_number = clean_string(raw.get("recall_number"))
        if not recall_number or recall_number in recall_numbers:
            raise ValueError(f"Missing or repeated recall_number: {recall_number!r}")
        recall_numbers.add(recall_number)
        row = {c: clean_string(raw.get(c), text=c in FDA_TEXT) for c in FDA_TEXT + FDA_CAT}
        row.update(event_id=event, recall_number=recall_number, source_row=index,
                   classification=label, recall_initiation_date=raw.get("recall_initiation_date"))
        sizes[event] += 1
        classes[event].add(label)
        if rng.randrange(sizes[event]) == 0:
            chosen[event] = row
    counts["eligible_product_rows"] = sum(sizes.values())
    counts["eligible_events"] = len(chosen)
    counts["events_with_multiple_products"] = sum(v > 1 for v in sizes.values())
    counts["events_with_multiple_severities"] = sum(len(v) > 1 for v in classes.values())
    counts["product_rows_removed_by_one_per_event"] = sum(sizes.values()) - len(chosen)
    return pd.DataFrame(chosen.values()), dict(counts)


def prepare_fda(kind, archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if n.endswith(".json"))
        with archive.open(member) as stream:
            frame, counts = choose_event_records(ijson.items(stream, "results.item"))
    dates = pd.to_datetime(frame.recall_initiation_date, format="%Y%m%d", errors="coerce")
    frame["recall_year"] = dates.dt.year
    frame["recall_month"] = dates.dt.month
    counts["missing_or_invalid_initiation_date_in_event_pool"] = int(dates.isna().sum())
    chosen = sample_rows(frame)
    counts["events_removed_by_row_cap"] = len(frame) - len(chosen)
    trace = chosen[["source_row", "event_id", "recall_number", "classification"]].copy()
    model = chosen[[*FDA_TEXT, *FDA_CAT, *FDA_NUM, "classification"]].copy()
    text_changes = []
    for col in FDA_TEXT:
        for index, original in model[col].items():
            cleaned = remove_recall_leakage(original)
            if cleaned != original:
                text_changes.append({"output_row": int(index), "field": col,
                                     "recall_number": trace.at[index, "recall_number"],
                                     "before": original, "after": cleaned})
                model.at[index, col] = cleaned
    assert trace.event_id.is_unique and trace.recall_number.is_unique
    return save(f"fda_{kind}_severity", model, trace, archive_path, member, counts,
                FDA_TEXT, FDA_CAT, FDA_NUM, "classification", {
                    "target": "Original Class I / Class II / Class III, without merging or ordering as a numeric target.",
                    "row_selection": "Exclude unknown severities; uniform random product within each event; then random events up to 15000.",
                    "mixed_severity_events": "Retain the selected product's own label; do not assign a single severity to all products in an event.",
                    "date_features": "Year and month of recall_initiation_date only; missing dates remain missing.",
                    "recall_severity_text": "Mask explicit recall severity; remove the audited pasted review template. Preserve device/laser/clinical classes. Labels are never inspected by the text cleaner.",
                }, text_changes=text_changes)


def lending_pool(archive_path):
    fields = ["id", *LENDING_TEXT, *LENDING_CAT,
              "loan_amnt", "annual_inc", "dti", "int_rate", "fico_range_low", "fico_range_high", "loan_status"]
    frames, counts, original_statuses = [], Counter(), Counter()
    with zipfile.ZipFile(archive_path) as archive:
        member = next(n for n in archive.namelist() if "accepted" in n.lower() and n.lower().endswith(".csv"))
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, usecols=fields, chunksize=100_000, low_memory=False):
                offset = counts["source_csv_rows"]
                counts["source_csv_rows"] += len(chunk)
                chunk["source_row"] = np.arange(offset, offset + len(chunk))
                counts["footer_rows_missing_loan_status"] += int(chunk.loan_status.isna().sum())
                original_statuses.update(chunk.loan_status.dropna())
                resolved = chunk.loan_status.isin(["Fully Paid", "Charged Off"])
                counts["resolved_paid_or_charged_off_rows"] += int(resolved.sum())
                chunk = chunk.loc[resolved].copy()
                original_desc = chunk.desc.map(clean_string)
                counts["resolved_rows_with_nonempty_raw_description"] += int(original_desc.notna().sum())
                for col in LENDING_TEXT:
                    chunk[col] = chunk[col].map(lambda v: clean_string(v, text=True))
                chunk = chunk[chunk.desc.notna()].copy()
                counts["eligible_rows_after_text_cleanup"] += len(chunk)
                frames.append(chunk)
                if counts["source_csv_rows"] % 500_000 == 0:
                    print(f"LendingClub: read {counts['source_csv_rows']:,} source rows", flush=True)
    frame = pd.concat(frames, ignore_index=True)
    if frame.id.isna().any() or not frame.id.is_unique:
        raise ValueError("Eligible LendingClub loan IDs are missing or repeated; review before selecting rows.")
    return frame, member, {**dict(counts), "original_loan_status_counts": dict(original_statuses)}


def prepare_lending(archive_path):
    frame, member, counts = lending_pool(archive_path)
    chosen = sample_rows(frame)
    trace = chosen[["source_row", "id", "loan_status"]].copy()
    chosen["fico_score"] = (pd.to_numeric(chosen.fico_range_low, errors="coerce")
                             + pd.to_numeric(chosen.fico_range_high, errors="coerce")) / 2
    # Values are already percentages, not fractions: 13.5 means a 13.5% annual rate.
    for col in LENDING_NUM:
        values = chosen[col].astype(str).str.replace("%", "", regex=False).str.strip()
        chosen[col] = pd.to_numeric(values, errors="coerce")
    for col in LENDING_CAT:
        chosen[col] = chosen[col].map(clean_category)
    chosen["charged_off"] = chosen.loan_status.map({"Fully Paid": 0, "Charged Off": 1})
    model = chosen[[*LENDING_TEXT, *LENDING_CAT, *LENDING_NUM, "charged_off"]].copy()
    counts["eligible_rows_removed_by_row_cap"] = len(frame) - len(chosen)
    return save("lendingclub_default", model, trace, archive_path, member, counts,
                LENDING_TEXT, LENDING_CAT, LENDING_NUM, "charged_off", {
                    "target": "0 = Fully Paid; 1 = Charged Off. All other loan statuses excluded.",
                    "row_selection": "Accepted-loan table only, resolved loans with a nonempty cleaned desc, simple random sample up to 15000.",
                    "fico_score": "Mean of origination fico_range_low and fico_range_high, never last_fico_range values.",
                    "term_and_employment_length": "Retained as categorical strings (for example 36 months and 10+ years).",
                    "interest_rate": "Numeric percentage points; int_rate and grade are lender-assigned origination features.",
                    "missing_fields": "Keep missing ancillary features; the existing learner preprocessing handles them.",
                })


def save(name, model, trace, archive_path, member, counts, text, categorical, numerical, target, rules,
         text_changes=None):
    if not MIN_ROWS <= len(model) <= MAX_ROWS:
        raise ValueError(f"{name}: {len(model)} rows is outside the approved 1000..15000 range")
    if model[target].isna().any():
        raise ValueError(f"{name}: missing target")
    if set(model) != set([*text, *categorical, *numerical, target]):
        raise ValueError(f"{name}: undeclared or missing columns")
    if np.isinf(model[numerical].to_numpy(dtype=float)).any():
        raise ValueError(f"{name}: infinite numeric values")
    if model[target].value_counts().min() < 6:
        raise ValueError(f"{name}: too few examples in a class for a reliable screen")
    csv_path = OUT / f"{name}.csv"
    trace_path = TRACE / f"{name}.selected_rows.csv"
    model.to_csv(csv_path, index=False, lineterminator="\n")
    trace.to_csv(trace_path, index=False, lineterminator="\n")
    mentions = []
    for col in text:
        mask = model[col].fillna("").str.contains(CLASS_MENTION)
        for i in model.index[mask]:
            mentions.append({"output_row": int(i), "field": col, "target": str(model.at[i, target]),
                             "text": model.at[i, col], "source": trace.iloc[i].to_dict()})
    audit_path = TRACE / f"{name}.class_mentions.json"
    audit_path.write_text(json.dumps(mentions, indent=2, default=str) + "\n", encoding="utf-8")
    summary = {
        "dataset": name, "preparation_version": 2, "seed": SEED,
        "preparation_script_sha256": digest(__file__),
        "source_archive": str(archive_path.resolve()), "source_member": member,
        "source_archive_sha256": digest(archive_path),
        "counts": {**counts, "selected_rows": len(model)},
        "csv": str(csv_path.relative_to(ROOT)), "csv_sha256": digest(csv_path),
        "selection_trace": str(trace_path.relative_to(ROOT)), "selection_trace_sha256": digest(trace_path),
        "text_columns": text, "categorical_columns": categorical, "numerical_columns": numerical,
        "target": target, "class_counts": {str(k): int(v) for k, v in model[target].value_counts().items()},
        "missing_counts": {k: int(v) for k, v in model.isna().sum().items()},
        "constant_features": [c for c in model.drop(columns=target) if model[c].nunique(dropna=False) <= 1],
        "text_profiles": {c: {"nonempty": int(model[c].notna().sum()),
                              "distinct": int(model[c].nunique()),
                              "median_words_nonempty": float(model[c].dropna().str.split().str.len().median()),
                              "max_words": int(model[c].dropna().str.split().str.len().max())}
                          for c in text},
        "duplicate_complete_feature_rows": int(model.drop(columns=target).duplicated().sum()),
        "duplicate_combined_text_rows": int(model[text].duplicated().sum()),
        "explicit_class_phrase_occurrences": len(mentions),
        "class_phrase_audit": str(audit_path.relative_to(ROOT)),
        "rules": {**rules, "text_cleanup": "Remove HTML tags, decode entities, normalize whitespace; retain date prefixes. Additional FDA leakage edits are documented separately.",
                  "no_word_length_filter": True, "no_text_truncation": True,
                  "no_class_balancing": True, "no_performance_based_selection": True},
    }
    if "country" in model:
        summary["country_counts"] = {str(k): int(v) for k, v in model.country.value_counts(dropna=False).items()}
    if text_changes is not None:
        changes_path = TRACE / f"{name}.text_changes.json"
        changes_path.write_text(json.dumps(text_changes, indent=2) + "\n", encoding="utf-8")
        summary["text_cleanup"] = {
            "changed_fields": len(text_changes), "changed_rows": len({c["output_row"] for c in text_changes}),
            "change_log": str(changes_path.relative_to(ROOT)), "change_log_sha256": digest(changes_path),
        }
    (OUT / f"{name}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {name}: {len(model):,} rows; classes {summary['class_counts']}", flush=True)
    return summary


def write_review(summaries):
    lines = ["# Candidate preparation for approval", "",
             "Prepared data only. No model fitting, TF-IDF fitting, or evaluation has been run.", "",
             "Seed 42. Keep between 1,000 and 15,000 rows per dataset. No class balancing, minimum word count, or text truncation.", "",
             "The original downloaded archives are unchanged. Identifiers and source row positions are stored separately under data/raw/candidate_screens/, outside the model CSVs.", "",
             "## Prepared samples", "",
             "| Dataset | Rows | Classes | Text columns | Structured columns |", "|---|---:|---|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['dataset']} | {s['counts']['selected_rows']:,} | "
                     + ", ".join(f"{k}: {v:,}" for k, v in s['class_counts'].items())
                     + f" | {len(s['text_columns'])} | {len(s['categorical_columns']) + len(s['numerical_columns'])} |")
    for s in summaries:
        lines.extend(["", f"## {s['dataset']}", "", f"- Target: {s['rules']['target']}",
                      f"- Text: {', '.join(s['text_columns'])}.",
                      f"- Categorical: {', '.join(s['categorical_columns'])}.",
                      f"- Numerical: {', '.join(s['numerical_columns'])}.",
                      f"- Selection: {s['rules']['row_selection']}",
                      f"- Counts: `{json.dumps(s['counts'], sort_keys=True)}`.",
                      f"- Constant features: {', '.join(s['constant_features']) or 'none'}.",
                      f"- Duplicate complete feature rows: {s['duplicate_complete_feature_rows']}.",
                      f"- Duplicate combined text rows: {s['duplicate_combined_text_rows']}.",
                      f"- Remaining class-phrase field values: {s['explicit_class_phrase_occurrences']}; device regulatory class is different from recall severity.",
                      f"- FDA leakage cleanup: {s.get('text_cleanup', {}).get('changed_rows', 0)} rows edited; row selection and labels unchanged.",
                      f"- CSV SHA256: `{s['csv_sha256']}`.", "",
                      "| Text | Nonempty | Distinct | Median words | Maximum words |", "|---|---:|---:|---:|---:|"])
        for c, p in s["text_profiles"].items():
            lines.append(f"| {c} | {p['nonempty']:,} | {p['distinct']:,} | {p['median_words_nonempty']:g} | {p['max_words']:,} |")
        lines.extend(["", "Missing values by column:", "",
                      ", ".join(f"{c}: {v}" for c, v in s["missing_counts"].items() if v) or "None."])
    lines.extend(["", "## Approved FDA text cleanup", "",
                  "Explicit recall-severity references have been masked with the same [recall severity] placeholder, while retaining legitimate device regulatory class, laser class, and clinical class descriptions.",
                  "The audited pasted FDA administrative review template has been removed. Row selection, labels, class proportions, and structured fields are unchanged.",
                  "Before/after edits and trace identifiers are recorded under data/raw/candidate_screens/. No models have been fitted.",
                  "", "## Proposed screen (awaiting approval)", "",
                  "- CPU only: CatBoost and LightGBM, the existing pinned MulTaBench wrappers and hyperparameters.",
                  "- Three conditions: structured, unstructured with TF-IDF, joint with TF-IDF.",
                  "- Three official outer folds (0, 1, 2); 18 evaluations per dataset, 54 total.",
                  "- LendingClub: AUROC. FDA: official negative log loss plus macro one-vs-rest AUROC.",
                  "- Joint delta is joint minus the better unimodal condition, computed separately for each metric.",
                  "- This cannot establish TAR gain or final dataset acceptance; no E5 or GPU run is included.",
                  "- Do not compare a log-loss delta numerically against an AUROC delta.", ""])
    (ROOT / "docs" / "CANDIDATE_SCREEN_PREPARATION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path.home() / "Downloads")
    parser.add_argument("--datasets", nargs="+", choices=["lendingclub", "device", "food"],
                        default=["lendingclub", "device", "food"])
    args = parser.parse_args()
    for path in [OUT, TRACE, ROOT / "docs"]:
        path.mkdir(parents=True, exist_ok=True)
    summaries = []
    for kind in ["lendingclub", "device", "food"]:
        name = "lendingclub_default" if kind == "lendingclub" else f"fda_{kind}_severity"
        if kind in args.datasets:
            if kind == "lendingclub":
                summary = prepare_lending(args.downloads / "archive (5).zip")
            else:
                summary = prepare_fda(kind, args.downloads / f"{kind}-enforcement-0001-of-0001.json.zip")
        else:
            existing = OUT / f"{name}.summary.json"
            if not existing.exists():
                continue
            summary = json.loads(existing.read_text(encoding="utf-8"))
        summaries.append(summary)
    write_review(summaries)


if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()

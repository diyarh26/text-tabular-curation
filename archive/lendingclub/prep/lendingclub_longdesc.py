"""Sample complete LendingClub loans with >50 content words; omit purpose.

Reads the FULL verified complete-case pool, not an earlier 15K sample.
The date prefixes are excluded from the length calculation and retained in model text.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

import candidate_screens as prep

NAME = "lendingclub_longdesc_no_purpose"
PARENT = "lendingclub_default_complete"
CATEGORICAL = [c for c in prep.LENDING_CAT if c != "purpose"]
FEATURES = prep.LENDING_TEXT + CATEGORICAL + prep.LENDING_NUM
TARGET = "charged_off"
DATE_PREFIX = re.compile(r"\bBorrower\s+added\s+on\s+\d{1,2}/\d{1,2}/\d{2,4}\s*>?", re.I)


def distribution(target):
    names = {0: "Fully Paid", 1: "Charged Off"}
    return {names[int(k)]: {"count": int(v), "percentage": 100 * int(v) / len(target)}
            for k, v in target.value_counts().items()}


def main():
    parent = json.loads((prep.OUT / f"{PARENT}.summary.json").read_text())
    pool_path = prep.ROOT / parent["full_clean_pool"]["csv"]
    trace_path = prep.ROOT / parent["full_clean_pool"]["trace"]
    assert prep.digest(pool_path) == parent["full_clean_pool"]["csv_sha256"]
    assert prep.digest(trace_path) == parent["full_clean_pool"]["trace_sha256"]
    protected = [p for stem in ["lendingclub_default", PARENT]
                 for p in [prep.OUT / f"{stem}.csv", prep.OUT / f"{stem}.summary.json"]]
    before = {p: prep.digest(p) for p in protected}
    frame = pd.read_csv(pool_path, low_memory=False)
    trace = pd.read_csv(trace_path)
    assert len(frame) == len(trace) == parent["full_clean_pool"]["rows"]
    assert not frame.isna().any().any() and trace.id.is_unique
    assert trace.loan_status.map({"Fully Paid": 0, "Charged Off": 1}).equals(frame[TARGET])

    content = frame.desc.map(lambda value: " ".join(DATE_PREFIX.sub(" ", value).split()))
    lengths = content.str.split().str.len()
    eligible = frame.loc[lengths.gt(50), FEATURES + [TARGET]].copy()
    eligible_trace = trace.loc[eligible.index].copy()
    eligible_trace["description_content_words"] = lengths.loc[eligible.index]
    if len(eligible) < prep.MAX_ROWS:
        raise ValueError(f"Only {len(eligible)} eligible rows; cannot sample 15K")
    assert "purpose" not in eligible and "title" in eligible
    assert not eligible.isna().any().any()
    assert np.isfinite(eligible[prep.LENDING_NUM].to_numpy(dtype=float)).all()
    long_pool_path = prep.TRACE / f"{NAME}_pool.csv"
    long_trace_path = prep.TRACE / f"{NAME}_pool.selected_rows.csv"
    eligible.to_csv(long_pool_path, index=False, lineterminator="\n")
    eligible_trace.to_csv(long_trace_path, index=False, lineterminator="\n")

    selected = eligible.sample(n=prep.MAX_ROWS, random_state=prep.SEED)
    selected_trace = eligible_trace.loc[selected.index].reset_index(drop=True)
    model = selected.reset_index(drop=True)
    assert selected_trace.id.is_unique and selected_trace.description_content_words.gt(50).all()
    counts = {"full_complete_parent_pool": len(frame), "long_description_eligible_pool": len(eligible),
              "excluded_at_most_50_content_words": len(frame) - len(eligible),
              "eligible_rows_removed_by_sampling": len(eligible) - len(model)}
    summary = prep.save(NAME, model, selected_trace, Path(parent["source_archive"]), parent["source_member"],
                        counts, prep.LENDING_TEXT, CATEGORICAL, prep.LENDING_NUM, TARGET, {
                            "target": "0 = Fully Paid; 1 = Charged Off.",
                            "row_selection": "From the FULL verified complete pool, require >50 description content words BEFORE random sampling of 15000 loans, seed 42.",
                            "missing_fields": "Zero missing selected inputs or targets; no imputation.",
                            "description_minimum": "Strictly greater than 50 whitespace-separated words after removing Borrower added on date prefixes for counting only.",
                            "date_prefixes_retained_in_model_text": True,
                            "purpose_removed_from_csv_and_config": True,
                            "title_retained_as_text": True,
                        })
    summary["preparation_script_sha256"] = prep.digest(__file__)
    summary["shared_preparation_helpers_sha256"] = prep.digest(prep.__file__)
    summary["rules"]["no_word_length_filter"] = False
    summary["parent_complete_pool_sha256"] = parent["full_clean_pool"]["csv_sha256"]
    summary["full_long_description_pool"] = {
        "csv": str(long_pool_path.relative_to(prep.ROOT)), "csv_sha256": prep.digest(long_pool_path),
        "trace": str(long_trace_path.relative_to(prep.ROOT)), "trace_sha256": prep.digest(long_trace_path),
        "class_distribution": distribution(eligible[TARGET]),
    }
    summary["sample_class_distribution"] = distribution(model[TARGET])
    summary["description_content_words"] = {
        "min": int(selected_trace.description_content_words.min()),
        "median": float(selected_trace.description_content_words.median()),
        "max": int(selected_trace.description_content_words.max()),
    }
    for path, checksum in before.items():
        assert prep.digest(path) == checksum, f"Previous data changed: {path}"
    summary["earlier_samples_unchanged"] = True
    (prep.OUT / f"{NAME}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    config = f"""# Complete loans with >50 description content words; purpose deliberately omitted.
name: {NAME}
csv: data/processed/{NAME}.csv
target: charged_off
task: classification

text_columns: [desc, title, emp_title]
categorical_columns: [term, emp_length, home_ownership, grade]
numerical_columns: [loan_amnt, annual_inc, dti, fico_score, int_rate]
"""
    (prep.ROOT / "configs" / f"{NAME}.yaml").write_text(config, encoding="utf-8")
    from tt_curation.spec import load_spec
    from tt_curation.data import load_dataset
    loaded = load_dataset(load_spec(prep.ROOT / "configs" / f"{NAME}.yaml"))
    assert len(loaded.x) == 15000 and loaded.x.shape[1] == 12
    assert not loaded.x.isna().any().any() and "purpose" not in loaded.x
    assert set(loaded.y) == {"0", "1"}
    print(json.dumps({"eligible_pool": len(eligible), "sample": len(model),
                      "sample_class_distribution": summary["sample_class_distribution"],
                      "content_words": summary["description_content_words"],
                      "inputs": len(FEATURES), "missing_input_cells": int(loaded.x.isna().sum().sum())}, indent=2))


if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()

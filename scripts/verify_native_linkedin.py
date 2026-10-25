"""Verify native loading matches the selected CSV and the original six splits.

Run with PYTHONPATH pointing at the isolated, dataset-patched MulTaBench checkout.
This validates integration; it does not rewrite or retrain any saved experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify(data_dir):
    from multabench.benchmark.load import load_multabench_dataset, load_split, load_target
    from multabench.baselines.preprocessing.feature_types import detect_feature_types
    from multabench.datasets.linkedin_salary import DATASETS, FULL, STRUCTURED, TEXT
    from multabench.datasets.info import list_datasets
    from tt_curation.data import load_dataset
    from tt_curation.spec import load_spec

    original = load_dataset(load_spec(ROOT / "configs/linkedin_salary.yaml"))
    manifest = json.loads((ROOT / "tests/fixtures/vm256_manifest.json").read_text(encoding="utf-8"))
    views = {FULL: original.x, STRUCTURED: original.x.drop(columns=original.text_columns),
             TEXT: original.x[original.text_columns]}
    registered = {info.id for info in list_datasets(tier="candidate")}
    assert registered == set(DATASETS), "Native registry is missing a view"
    report = {}
    for dataset_id, expected in views.items():
        dataset = load_multabench_dataset(dataset_id, data_dir=str(data_dir))
        pd.testing.assert_frame_equal(dataset.x, expected, check_exact=True)
        pd.testing.assert_series_equal(dataset.y, original.y, check_exact=True)
        target, task = load_target(dataset_id, data_dir=str(data_dir))
        pd.testing.assert_series_equal(target, original.y, check_exact=True)
        assert task == dataset.task_type
        types = detect_feature_types(dataset.x, image_column=None)
        assert types.text_features == (set() if dataset_id == STRUCTURED else set(original.text_columns))
        splits = {}
        for fold in range(6):
            split = load_split(dataset_id, fold=fold, size="10k", data_dir=str(data_dir))
            saved = manifest["splits"][str(fold)]
            for label, values in [("train", split.x_train), ("test", split.x_test)]:
                digest = hashlib.sha256(np.asarray(values.index, dtype="<i8").tobytes()).hexdigest()
                assert digest == saved[f"{label}_indices_sha256"]
            pd.testing.assert_frame_equal(split.x_train, expected.loc[split.x_train.index], check_exact=True)
            pd.testing.assert_frame_equal(split.x_test, expected.loc[split.x_test.index], check_exact=True)
            splits[str(fold)] = dict(n_train=len(split.y_train), n_test=len(split.y_test), matches_vm_split=True)
        report[dataset_id.name] = dict(rows=len(dataset.y), columns=list(dataset.x),
            exact_values_dtypes_row_order_and_target=True, splits=splits)
    result = dict(loader_parity_passed=True, gpu_training_performed=False,
        original_results_relabelled=False, views=report)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = verify(Path(args.data_dir).resolve())
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

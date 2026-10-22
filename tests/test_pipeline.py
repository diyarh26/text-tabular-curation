import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tt_curation import REPO_ROOT
from tt_curation.conditions import Condition, condition_features
from tt_curation.criteria import curation_verdict, evaluation_report
from tt_curation.data import load_dataset
from tt_curation.spec import DatasetSpec, load_spec

sys.path.insert(0, str(REPO_ROOT / "prep"))
import make_toy_dataset  # noqa: E402


@pytest.fixture(scope="module")
def toy():
    make_toy_dataset.main()
    return load_dataset(load_spec(REPO_ROOT / "configs" / "toy.yaml"))


def test_types_follow_the_spec(toy):
    assert list(toy.x.columns) == ["price", "category", "review"]
    assert toy.x["category"].dtype == "category"
    assert toy.x["review"].dtype == object
    assert toy.x["price"].dtype == float
    assert toy.structured_columns == ["price", "category"]


def test_conditions_split_the_modalities(toy):
    assert list(condition_features(toy.x, toy.text_columns, Condition.STRUCTURED).columns) == ["price", "category"]
    assert list(condition_features(toy.x, toy.text_columns, Condition.UNSTRUCTURED).columns) == ["review"]
    assert condition_features(toy.x, toy.text_columns, Condition.JOINT_TAR).shape == toy.x.shape


def test_spec_rejects_target_as_feature():
    with pytest.raises(ValueError):
        DatasetSpec(name="x", csv=Path("x.csv"), target="y", task="regression", text_columns=["y"])


def _fake_runs(scores):
    rows = []
    for model, by_condition in scores.items():
        for condition, values in by_condition.items():
            for fold, v in enumerate(values):
                rows.append(dict(dataset="d", model=model, condition=condition, fold=fold, status="ok",
                                 test_score=v, test_error=1 - v, auroc=v, metric="roc_auc"))
    return pd.DataFrame(rows)


def test_verdict():
    passing = dict(structured=[.70, .71], unstructured=[.65, .66], joint_frozen=[.75, .74], joint_tar=[.78, .77])
    no_tar = dict(passing, joint_tar=[.74, .73])
    verdict = curation_verdict(_fake_runs({"A": passing, "B": no_tar})).set_index("model")
    assert verdict.loc["A", "passes"] and verdict.loc["A", "tar_gain_fold_wins"] == 2
    assert verdict.loc["B", "joint_signal"] and not verdict.loc["B", "tar_gain"]
    # As in the MulTaBench paper: means rounded to 3 decimals, so a 0.0004 edge is a tie and 0.001 is a pass.
    tie = dict(structured=[.7400, .7400], unstructured=[.6, .6], joint_frozen=[.7404, .7404],
               joint_tar=[.7414, .7414])
    verdict = curation_verdict(_fake_runs({"C": tie})).set_index("model")
    assert not verdict.loc["C", "joint_signal"] and verdict.loc["C", "tar_gain"]
    report = evaluation_report(_fake_runs({"A": passing}))
    assert list(report["condition"]) == ["structured", "unstructured", "joint_frozen", "joint_tar"]


@pytest.mark.bench
@pytest.mark.skipif(importlib.util.find_spec("multabench") is None, reason="MulTaBench not installed")
def test_end_to_end_on_cpu(toy, tmp_path):
    """The CPU conditions; TAR needs a GPU."""
    from tt_curation.runner import run_all
    cpu_conditions = [Condition.STRUCTURED, Condition.UNSTRUCTURED, Condition.JOINT_FROZEN]
    run_all(toy, tmp_path, models=["light", "cat"], conditions=cpu_conditions, folds=[0], device="cpu")
    runs = [json.loads(p.read_text()) for p in (tmp_path / "runs").glob("*.json")]
    assert len(runs) == 6
    assert all(r["status"] == "ok" for r in runs), [r.get("error") for r in runs]


@pytest.mark.bench
@pytest.mark.skipif(importlib.util.find_spec("multabench") is None, reason="MulTaBench not installed")
def test_embedding_cache_matches_plain_encoding(toy):
    """Each distinct text is encoded once, and its embedding equals plain encoding up to float rounding (batch
    composition changes the rounding as much in MulTaBench's own code)."""
    import torch
    from multabench.baselines.preprocessing import text_embeddings
    from multabench.e5.e5_finetune import get_vanilla_e5
    from tt_curation.runner import embedding_cache

    texts = toy.x["review"].tolist()[:120]
    model, tokenizer = get_vanilla_e5(torch.device("cpu"))
    plain = text_embeddings.encode_texts_with_e5(texts, "review", model, tokenizer, "cpu")
    with embedding_cache():
        cached_fn = text_embeddings.encode_texts_with_e5
        first = cached_fn(texts, "review", model, tokenizer, "cpu")
        again = cached_fn(texts[::-1], "review", model, tokenizer, "cpu")[::-1]
    assert text_embeddings.encode_texts_with_e5.__name__ == "encode_texts_with_e5"
    np.testing.assert_allclose(first, plain, atol=1e-5)
    np.testing.assert_array_equal(first, again)


@pytest.mark.bench
@pytest.mark.skipif(importlib.util.find_spec("multabench") is None, reason="MulTaBench not installed")
def test_tar_refuses_cpu(toy, tmp_path):
    from tt_curation.runner import run_all
    with pytest.raises(RuntimeError, match="needs a GPU"):
        run_all(toy, tmp_path, models=["light"], conditions=[Condition.JOINT_TAR], folds=[0], device="cpu")

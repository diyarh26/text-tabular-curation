"""A full run is complete only when the expected 120 successful evaluations are present."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_full_steam import audit_runs  # noqa: E402
from tt_curation.conditions import Condition, MODELS  # noqa: E402


def make_runs(directory):
    for fold in range(6):
        for model in MODELS:
            for condition in Condition:
                encoder = ("e5-small-tar" if condition == Condition.JOINT_TAR else
                           None if condition == Condition.STRUCTURED else "e5-small")
                result = {"dataset": "steam_recommended", "status": "ok", "test_score": .9,
                          "short_name": model, "condition": str(condition), "fold": fold,
                          "n_train": 10_000, "n_test": 5_000, "text_encoder": encoder}
                path = directory / f"{model}_{condition}_10k_fold{fold}.json"
                path.write_text(json.dumps(result), encoding="utf-8")


def test_audit_all_120_and_rejects_failed_missing_or_wrong_encoder(tmp_path):
    make_runs(tmp_path)
    assert audit_runs(tmp_path, range(6), MODELS, Condition) == []
    (tmp_path / "cat_structured_10k_fold0.json").unlink()
    (tmp_path / "unrelated.json").write_text("{}")  # Still 120 files: counting alone is insufficient.
    failed = tmp_path / "light_joint_tar_10k_fold2.json"
    result = json.loads(failed.read_text())
    result.update(status="error", error="model access denied")
    failed.write_text(json.dumps(result))
    wrong = tmp_path / "cat_joint_frozen_10k_fold4.json"
    result = json.loads(wrong.read_text())
    result["text_encoder"] = "tfidf"
    wrong.write_text(json.dumps(result))
    assert len(list(tmp_path.glob("*.json"))) == 120
    problems = audit_runs(tmp_path, range(6), MODELS, Condition)
    assert len(problems) == 3
    assert any("missing" in message for message in problems)
    assert any("model access denied" in message for message in problems)
    assert any("unexpected experiment metadata" in message for message in problems)

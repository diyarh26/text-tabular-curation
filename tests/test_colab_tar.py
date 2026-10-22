"""Colab comparison semantics and a real, tiny CPU checkpoint/reload test; no model downloads."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("colab_tar_compare", Path(__file__).resolve().parents[1] /
                                             "scripts/colab_tar_compare.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_summary_matches_folds_and_does_not_call_tfidf_difference_tar_gain(tmp_path, monkeypatch):
    import tt_curation
    monkeypatch.setattr(tt_curation, "REPO_ROOT", tmp_path / "project")
    results = tmp_path / "results"
    dataset = results / "steam_experiment"
    pilot.write_json(dataset / "manifest.json", {"dataset": "steam_recommended", "task": "classification",
                                               "max_epochs": 10})
    common = {"model": "CatBoost", "status": "ok"}
    pilot.write_json(dataset / "runs/tar.json", {**common, "fold": 0, "condition": "joint_tar", "auroc": .90})
    pilot.write_json(dataset / "runs/frozen_other_fold.json",
                     {**common, "fold": 1, "condition": "joint_frozen", "auroc": .75})
    baseline = tt_curation.REPO_ROOT / "results/steam_recommended_screen_tfidf/runs"
    for fold, score in enumerate([.70, .80, .90]):
        pilot.write_json(baseline / f"fold{fold}.json",
                         {**common, "fold": fold, "condition": "joint_frozen", "auroc": score})
    row = pilot.build_summary(results).iloc[0]
    assert row["TFIDF_joint_same_folds"] == .70
    assert row["TAR_minus_TFIDF_not_isolated_TAR_gain"] == pytest.approx(.20)
    assert "isolated_TAR_gain" not in row
    pilot.write_json(dataset / "runs/frozen_same_fold.json",
                     {**common, "fold": 0, "condition": "joint_frozen", "auroc": .85})
    row = pilot.build_summary(results).iloc[0]
    assert row["isolated_TAR_gain"] == pytest.approx(.05)


@pytest.mark.bench
@pytest.mark.skipif(importlib.util.find_spec("multabench") is None, reason="MulTaBench not installed")
def test_epoch_resume_and_completed_adapter_reload(tmp_path, monkeypatch):
    import pandas as pd
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast, TrainerCallback
    from multabench.baselines.preprocessing import text_embeddings
    from multabench.e5.e5_finetune import encode_texts_with_e5
    from multabench.finetune import lora

    torch.set_num_threads(2)
    base = tmp_path / "tiny_bert"
    base.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "passage", ":", "review",
                  "good", "bad", "game", "story", "fun", "slow"]
    (base / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"))
    tokenizer.save_pretrained(str(base))
    BertModel(BertConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=3, num_attention_heads=2, max_position_embeddings=512)).save_pretrained(str(base))
    x = pd.DataFrame({"review": ["good game story fun", "bad game slow"] * 20})
    y = pd.Series(["yes", "no"] * 20)
    cache = tmp_path / "cache"
    original_trainer = lora.Trainer

    class InterruptAfterSave(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            raise RuntimeError("simulated Colab interruption")

    class InterruptingTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(InterruptAfterSave())

    monkeypatch.setattr(lora, "Trainer", InterruptingTrainer)
    with pytest.raises(RuntimeError, match="simulated Colab interruption"):
        with pilot.persistent_tar(cache, epochs=2, micro_batch=16):
            text_embeddings.finetune_e5(x, y, ["review"], is_cls=True, device="cpu", model_name=str(base))
    assert not (cache / "complete.json").exists()
    assert list((cache / "trainer").glob("checkpoint-*/trainer_state.json"))

    monkeypatch.setattr(lora, "Trainer", original_trainer)
    with pilot.persistent_tar(cache, epochs=2, micro_batch=16):
        model, tokenizer, stats = text_embeddings.finetune_e5(x, y, ["review"], is_cls=True,
                                                           device="cpu", model_name=str(base))
    assert stats["epochs"] == 2
    expected = encode_texts_with_e5(["good game"], "review", model, tokenizer, "cpu")
    with pilot.persistent_tar(cache, epochs=2, micro_batch=16):
        loaded, loaded_tokenizer, saved_stats = text_embeddings.finetune_e5(
            x, y, ["review"], is_cls=True, device="cpu", model_name=str(base))
    actual = encode_texts_with_e5(["good game"], "review", loaded, loaded_tokenizer, "cpu")
    assert saved_stats["adapter_reused"]
    np.testing.assert_allclose(expected, actual, atol=1e-6)
    assert lora.Trainer is original_trainer

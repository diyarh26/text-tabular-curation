"""Read saved experiment results only. Never starts training or changes original artifacts."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import zipfile

import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DOWNLOADS = Path(r"C:\Users\Diyar\Downloads")
sys.path.insert(0, str(ROOT / "scripts"))
from tt_curation import MULTABENCH_COMMIT
from tt_curation.conditions import MODELS
from tt_curation.data import load_dataset
from tt_curation.spec import load_spec
from multabench.benchmark.splits import get_split
import numpy as np

CONDITIONS = ["structured", "unstructured", "joint_frozen", "joint_tar"]
ORDER = ["CatBoost", "LightGBM", "TabM", "TabPFN-v2", "TabPFN-v2p5"]
ALL_FOLDS = set(range(6))
EXPECTED = {(m, c, f) for m in MODELS for c in CONDITIONS for f in ALL_FOLDS}


def sha(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def save_source(path, content):
    path = OUT / "sources" / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Preserving existing source: " + str(path))
    path.write_bytes(content)


def zip_source(key, filename):
    path = DOWNLOADS / filename
    files = {}
    states = []
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            assert not member.is_absolute() and ".." not in member.parts and "\\" not in info.filename
            if info.is_dir():
                continue
            if "/checkpoints/" in info.filename:
                if info.filename.endswith("/trainer_state.json"):
                    state = json.loads(archive.read(info))
                    states.append(dict(path=info.filename, epoch=state.get("epoch"),
                                       step=state.get("global_step"), best_metric=state.get("best_metric")))
                continue
            assert info.file_size < 30_000_000
            content = archive.read(info)  # Checks the CRC of every imported result/metadata/log file.
            name = "/".join(member.parts[1:])
            files[name] = content
            save_source(Path(key) / name, content)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return files, dict(type="downloaded_zip", file=str(path), sha256=digest, trainer_states=states)


def local_source(folder):
    folder = ROOT / folder
    files = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*")
             if p.is_file() and "checkpoints" not in p.relative_to(folder).parts}
    return files, dict(type="local_results", folder=str(folder))


def github_device():
    revision = subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip()
    prefix = "results/fda_device_full_e5/fda_device_severity/"
    data = subprocess.check_output(["git", "archive", revision, prefix.rstrip("/")], cwd=ROOT)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            assert member.name.startswith(prefix)
            name = member.name[len(prefix):]
            content = archive.extractfile(member).read()
            files[name] = content
            save_source(Path("device_vm") / "fda_device_severity" / name, content)
    audit = json.loads(files["final_audit.json"])
    checked = 0
    for name, expected in audit["run_json_sha256"].items():
        content = files[name]
        assert hashlib.sha256(content).hexdigest() == expected or sha(content) == expected
        checked += 1
    assert checked == 120
    return files, dict(type="github_commit", commit=revision,
                       folder=prefix, original_final_audit_run_hashes_verified=checked)


def frame(files):
    rows = [json.loads(value) for name, value in files.items()
            if "/runs/" in "/" + name and name.endswith(".json")]
    return pd.DataFrame(rows)


def table(runs, metric, conditions=CONDITIONS):
    result = runs.pivot_table(index="model", columns="condition", values=metric)
    result = result.reindex([m for m in ORDER if m in result.index])[conditions].copy()
    result["Joint delta"] = result.joint_frozen - result[["structured", "unstructured"]].max(axis=1)
    if "joint_tar" in conditions:
        result["TAR delta"] = result.joint_tar - result.joint_frozen
    return result


def markdown(result):
    labels = dict(structured="Structured", unstructured="Text only", joint_frozen="Joint frozen",
                  joint_tar="Joint TAR", **{"Joint delta": "Joint Δ", "TAR delta": "TAR Δ"})
    lines = ["| Learner | " + " | ".join(labels[c] for c in result) + " |",
             "|---|" + "---:|" * len(result.columns)]
    for name, row in result.iterrows():
        values = [f"{row[c]:+.6f}" if c.endswith("delta") else f"{row[c]:.6f}" for c in result]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    return "\n".join(lines)


def analyze(key, title, dataset, files, source, *, legacy=False):
    runs = frame(files)
    spec = load_spec(ROOT / "configs" / (dataset + ".yaml"))
    data = load_dataset(spec)
    metric = "auroc" if spec.is_cls else "r2"
    raw_manifest = next((value for name, value in files.items() if name.endswith("manifest.json")), None)
    manifest = json.loads(raw_manifest) if raw_manifest else {}
    assert not runs.duplicated(["short_name", "condition", "fold"]).any()
    assert set(runs.dataset) == {dataset}
    assert set(runs.status) == {"ok"}
    assert all(math.isfinite(value) for value in runs[metric])
    actual = set(zip(runs.short_name, runs.condition, runs.fold))
    assert actual <= EXPECTED
    source["dataset_csv_verified"] = bool(manifest and sha(spec.csv.read_bytes()) == manifest["csv_sha256_lf"])
    source["config_verified"] = bool(manifest and sha((ROOT / "configs" / (dataset + ".yaml")).read_bytes())
                                     == manifest["config_sha256_lf"])
    if manifest:
        assert source["dataset_csv_verified"] and source["config_verified"]
        assert manifest["multabench_commit"] == MULTABENCH_COMMIT
        revision = manifest.get("source_revision") or runs["git"].dropna().iloc[0]
        matched = 0
        for path, expected in manifest["code"].items():
            content = subprocess.check_output(["git", "show", revision + ":" + path], cwd=ROOT)
            assert sha(content) == expected, path
            matched += 1
        source.update(runtime_source_revision=revision, source_file_hashes_verified=matched)
    splits = {}
    for fold in sorted(runs.fold.unique()):
        train, test = get_split(data.y.to_numpy(), is_cls=spec.is_cls, split=int(fold), size="10k")
        splits[str(fold)] = dict(n_train=len(train), n_test=len(test),
            train_indices_sha256=hashlib.sha256(np.asarray(train, dtype="<i8").tobytes()).hexdigest(),
            test_indices_sha256=hashlib.sha256(np.asarray(test, dtype="<i8").tobytes()).hexdigest())
        if manifest.get("splits"):
            assert manifest["splits"][str(fold)] == splits[str(fold)]
    for row in runs.to_dict("records"):
        assert MODELS[row["short_name"]] == row["model"]
        assert row["multabench_commit"] == MULTABENCH_COMMIT
        encoder = (None if row["condition"] == "structured" else
                   "e5-small-tar" if row["condition"] == "joint_tar" else "e5-small")
        assert row["text_encoder"] == encoder
        assert (row["n_train"], row["n_test"]) == (splits[str(row["fold"])]["n_train"],
                                                   splits[str(row["fold"])]["n_test"])
        assert all(math.isfinite(row[c]) for c in ("test_score", "test_error", metric))
        if spec.is_cls:
            assert 0 <= row[metric] <= 1
        else:
            assert row[metric] <= 1
    complete = [f for f in range(6) if {(m, c, f) for m in MODELS for c in CONDITIONS} <= actual]
    selected = runs[runs.fold.isin(complete)]
    scores = table(selected, metric)
    joint, tar = scores["Joint delta"], scores["TAR delta"]
    rounded = scores[CONDITIONS].round(3).mul(1000).round().astype("int64")
    paper_joint = rounded.joint_frozen - rounded[["structured", "unstructured"]].max(axis=1)
    paper_tar = rounded.joint_tar - rounded.joint_frozen
    both = int((paper_joint.ge(1) & paper_tar.ge(1)).sum())
    stats = dict(key=key, title=title, dataset=dataset, rows=len(data.y), metric=metric,
        successful_evaluations=len(runs), expected_evaluations=120, completed_folds=complete,
        missing=[dict(model=m, condition=c, fold=f) for m, c, f in sorted(EXPECTED - actual)],
        complete=len(actual) == 120,
        micro_batch=manifest.get("micro_batch"),
        accumulation=256 // manifest["micro_batch"] if manifest.get("micro_batch") else None,
        joint_count=int(joint.gt(0).sum()), tar_count=int(tar.gt(0).sum()),
        joint_mean_passing=float(joint[joint.gt(0)].mean()), tar_mean_passing=float(tar[tar.gt(0)].mean()),
        joint_mean_all=float(joint.mean()), tar_mean_all=float(tar.mean()),
        course_acceptance=(bool(joint.gt(0).sum() >= 3 and tar.gt(0).sum() >= 3) if len(actual) == 120 else None),
        preliminary_course_counts_meet_rule=bool(joint.gt(0).sum() >= 3 and tar.gt(0).sum() >= 3),
        paper_both_count=both, paper_rule_acceptance=(both >= 3 if len(actual) == 120 else None),
        source=source)
    per_fold = []
    for fold in complete:
        t = table(runs[runs.fold.eq(fold)], metric)
        per_fold.extend(dict(fold=fold, model=name, **row.to_dict()) for name, row in t.iterrows())
    stats["per_model"] = scores.reset_index().to_dict("records")
    stats["per_fold"] = per_fold
    (OUT / (key + ".json")).write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    status = "COMPLETE" if stats["complete"] else "INCOMPLETE: matched completed folds only"
    text = [f"## {title}", "", f"{status}; {len(runs)}/120 successful evaluations; metric {metric}; "
            f"table averages folds {complete}."]
    if manifest.get("micro_batch"):
        text += [f"Physical batch {stats['micro_batch']}, accumulation {stats['accumulation']}, effective batch 256."]
    text += ["", markdown(scores), "",
             f"Joint Signal {stats['joint_count']}/5; TAR Gain {stats['tar_count']}/5.",
             f"Passing-only means, independently by criterion: Joint {stats['joint_mean_passing']:+.6f}; "
             f"TAR {stats['tar_mean_passing']:+.6f}.",
             f"All-five means: Joint {stats['joint_mean_all']:+.6f}; TAR {stats['tar_mean_all']:+.6f}.",
             f"Paper rule on these matched folds: {both}/5 satisfy both rounded-condition-mean margins >=0.001."]
    if not stats["complete"]:
        text += ["No final acceptance verdict. Partial next-fold baselines were excluded from all four condition means."]
    if key == "linkedin":
        text += ["Fold 5 has its 15 baselines, but all five joint_tar evaluations are missing. "
                 "The saved fold-5 Trainer checkpoint is at epoch 30. This archive is a snapshot, not proof of the live session's current state."]
    if key == "device":
        text += ["The original GitHub final audit declares failure using negative log loss; "
                 "the AUROC curation calculation above passes. Original files are preserved unchanged."]
    if legacy:
        text += ["Historical stopped experiment; not a completed six-fold candidate evaluation."]
    return stats, "\n".join(text)


def main():
    blocks, experiments = [], []
    specifications = [
        ("linkedin", "LinkedIn salary — provisional", "linkedin_salary", "linkedin_full_checkpoint.zip"),
        ("food256", "FDA Food — physical batch 256", "fda_food_severity", "fda_food_full_batch256_checkpoint.zip"),
        ("food64", "FDA Food — batch 64 x 4", "fda_food_severity", "fda_food_full_checkpoint.zip"),
        ("lendingclub", "LendingClub long descriptions, purpose removed", "lendingclub_longdesc_no_purpose", "lendingclub_full_checkpoint.zip")]
    for key, title, dataset, filename in specifications:
        files, source = zip_source(key, filename)
        stats, text = analyze(key, title, dataset, files, source)
        experiments.append(stats); blocks.append(text)
    files, source = github_device()
    stats, text = analyze("device", "FDA Device — Azure VM, physical batch 256", "fda_device_severity", files, source)
    experiments.append(stats); blocks.append(text)
    files, source = local_source("results/steam_full_e5/steam_recommended")
    stats, text = analyze("steam", "Steam recommendations — physical batch 256", "steam_recommended", files, source)
    experiments.append(stats); blocks.append(text)
    files, source = local_source("results/la_airbnb_price")
    stats, text = analyze("airbnb", "LA Airbnb log price — historical partial evaluation", "la_airbnb_price", files, source, legacy=True)
    experiments.append(stats); blocks.append(text)

    lines = ["# All saved experiment results — 2026-10-24", "",
        "This report reads existing JSON results only. No model was trained or rerun.",
        "Classification curation uses AUROC (macro one-vs-rest for multiclass); regression uses R². "
        "Repository negative log loss/RMSE are diagnostic and are not substituted for those curation metrics.",
        "Joint Δ = mean(joint frozen) − max(mean(structured), mean(text)). "
        "TAR Δ = mean(joint TAR) − mean(joint frozen). Decisions use unrounded values.",
        "Course counts are independent: Joint >0 in >=3/5 learners and TAR >0 in >=3/5. "
        "The paper rule is shown separately using rounded condition means, >=0.001 and the same three learners.",
        "All comparisons average the SAME fully completed folds across all four conditions. "
        "A partial fold's baselines are never compared with fewer TAR folds.",
        "Passing-only means follow the user's requested definition: average each delta only over "
        "learners positive on that criterion. They exclude negative results; counts and all-five means "
        "are also shown so they are not mistaken for whole-pool averages.", "",
        "## Project ranking", "",
        "This is a project recommendation based on both criteria, gain sizes and completion. "
        "AUROC and R² are different metrics; this is not a numerical leaderboard mixing their scales.", "",
        "| Rank | Dataset | Status | Joint/TAR learners | Passing-only Joint mean | Passing-only TAR mean |",
        "|---|---|---|---:|---:|---:|"]
    by_key = {item["key"]: item for item in experiments}
    for rank, key in [("1 (provisional)", "linkedin"), ("2; best completed", "device"),
                      ("3", "food256"), ("4", "steam"), ("5", "lendingclub"), ("6; historical partial", "airbnb")]:
        item = by_key[key]
        status = ("PASSES" if item["course_acceptance"] else "FAILS") if item["complete"] else "INCOMPLETE"
        lines.append(f"| {rank} | {item['title']} | {status} | {item['joint_count']}/5; {item['tar_count']}/5 | "
                     f"{item['joint_mean_passing']:+.6f} | {item['tar_mean_passing']:+.6f} |")
    lines += ["", "Food batch 64 x 4 is the same prepared dataset under a separate execution protocol, "
              "not an additional candidate; it also passes. Food batch 256 is the reference-setting run used in the ranking.",
              "LinkedIn is the leading provisional candidate: all five learners exceed +0.01 on both "
              "five-fold R² mean deltas. Finish the five missing fold-5 TAR evaluations before issuing a final verdict.",
              "Device is the strongest completed option under the requested passing-only summary; "
              "its positive Joint/TAR means exceed +0.01, although fewer than three individual learners "
              "have Joint gains >=+0.01. Passing-only averages do not imply that every learner meets that target.", ""]
    lines.extend(block + "\n" for block in blocks)

    lines += ["", "## Earlier CPU TF-IDF screens", "",
              "Each screen has two learners, three conditions and three folds (18 runs), "
              "with no TAR evaluation. These are screening indicators, not final acceptance evidence."]
    screens = []
    for folder in sorted((ROOT / "results").glob("*screen_tfidf")):
        files, _ = local_source(folder.relative_to(ROOT))
        runs = frame(files)
        metric = "r2" if "r2" in runs and runs.r2.notna().all() else "auroc"
        scores = table(runs, metric, CONDITIONS[:3])
        assert len(runs) == 18 and runs.status.eq("ok").all()
        screens.append(dict(experiment=folder.name, metric=metric, rows=len(runs), per_model=scores.reset_index().to_dict("records")))
        lines += ["", "### " + folder.name, "", "Metric: " + metric, "", markdown(scores)]
    lines += ["", "## Earlier Colab TAR pilots", "",
              "One fold, CatBoost and LightGBM, ten-epoch pilot cap, TAR only. "
              "The historical microbatch-loss implementation predates the corrected full-run loss normalization. "
              "Do not use these pilots for a final verdict or compare their TAR scores with TF-IDF to claim isolated TAR gain.", "",
              "| Dataset | Learner | Metric | Pilot joint TAR |", "|---|---|---|---:|"]
    pilots = []
    for folder in sorted((ROOT / "results/colab_tar_pilot").glob("*/runs")):
        for path in sorted(folder.glob("*.json")):
            row = json.loads(path.read_text())
            metric = "r2" if row["task"] == "regression" else "auroc"
            pilots.append(dict(dataset=row["dataset"], learner=row["model"], metric=metric, joint_tar=row[metric]))
            lines.append(f"| {row['dataset']} | {row['model']} | {metric} | {row[metric]:.6f} |")
    lines += ["", "Glassdoor has CPU screens only. No completed MulTaBench GPU evaluation was found for "
              "Indiegogo or FDA Drug in these sources; they are not assigned a final-result rank.",
              "The failed Kaggle Device-256 attempt is not an extra completed experiment. "
              "Its OOM does not alter the independent completed Azure VM Device results.", "",
              "## Audit and sources", "",
              "Every imported successful run was checked for dataset, model, condition, fold, encoder, "
              "pinned MulTaBench version, train/test sizes, finite metric and duplicate identity. "
              "Available manifest data/config hashes, source-code hashes and saved split hashes were verified. "
              "Downloaded model/optimizer weight files were not extracted or executed.",
              "Device was read from GitHub origin/main; its 120 run-file hashes also match the uploaded final audit. "
              "Other originals remain in Downloads/local results; imported small artifacts are preserved under sources/.",
              "Legacy Airbnb has no experiment manifest; its provenance evidence is weaker than the modern full runs. "
              "Its two-fold table is diagnostic only.", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "all_results.json").write_text(json.dumps(dict(experiments=experiments, screens=screens, pilots=pilots), indent=2) + "\n", encoding="utf-8")
    print("Verified and summarized", len(experiments), "GPU experiment variants,", len(screens), "CPU screens,", len(pilots), "pilot scores.")
    for item in experiments:
        print(item["key"], item["successful_evaluations"], item["completed_folds"],
              item["joint_count"], item["tar_count"], item["joint_mean_passing"], item["tar_mean_passing"])
    print("Report:", OUT / "summary.md")


if __name__ == "__main__":
    main()

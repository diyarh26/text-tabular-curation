"""Local LinkedIn salary candidate and its ablation views, inside MulTaBench.

Only dataset preparation and lookup live here. Embedding creation and evaluation
use the unchanged multabench.embeddings.embed and benchmark.py entry points.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from multabench.datasets.all_datasets import MulTaBenchDatasetID

FULL = MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY
STRUCTURED = MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY_STRUCTURED
TEXT = MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY_TEXT
DATASETS = (FULL, STRUCTURED, TEXT)
ENV = "MULTABENCH_LINKEDIN_DATA_DIR"
SOURCE = "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings"
CSV_SHA256_LF = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
TARGET = "log_salary"
TEXT_COLUMNS = ["title", "description"]
CATEGORIES = ["experience_level", "work_type", "state", "application_type", "company_industry", "job_industry"]
NUMERICAL = ["remote_allowed", "company_size", "company_employees", "company_followers", "n_skills", "n_benefits"]


def local_root(data_dir=None) -> Path:
    value = data_dir if data_dir is not None else os.environ.get(ENV)
    if not value:
        raise ValueError(f"Set {ENV} to the prepared local LinkedIn data/cache directory")
    return Path(value).expanduser().resolve()


def dataset_dir(dataset_id, data_dir=None) -> str:
    if dataset_id not in DATASETS:
        raise ValueError("Not a local LinkedIn dataset")
    root = local_root(data_dir)
    manifest_path = root / "linkedin_preparation.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepare the exact LinkedIn CSV first: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("csv_sha256_lf") != CSV_SHA256_LF or manifest.get("rows") != 15000:
        raise ValueError("LinkedIn preparation provenance differs")
    path = root / dataset_id.value
    for name in ("data.parquet", "metadata.json"):
        file = path / name
        expected = manifest.get("files", {}).get(f"{dataset_id.value}/{name}")
        if not file.is_file():
            raise FileNotFoundError(file)
        if expected is None or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared LinkedIn file differs: {file}")
    return str(path)


def typed_csv(csv) -> pd.DataFrame:
    csv = Path(csv)
    if hashlib.sha256(csv.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != CSV_SHA256_LF:
        raise ValueError("Expected the unchanged, selected 15,000-row LinkedIn CSV")
    df = pd.read_csv(csv, low_memory=False)
    if len(df) != 15000 or set(df) != set(TEXT_COLUMNS + CATEGORIES + NUMERICAL + [TARGET]):
        raise ValueError("Unexpected LinkedIn rows or columns")
    for col in TEXT_COLUMNS + CATEGORIES:
        values = df[col].astype(object).where(df[col].notna(), None)
        df[col] = values.map(lambda v: v if v is None else str(v))
        if col in CATEGORIES:
            df[col] = df[col].astype("category")
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df[TARGET] = df[TARGET].astype(float)
    if not np.isfinite(df[TARGET]).all() or df[TEXT_COLUMNS].isna().any().any():
        raise ValueError("Expected finite targets and nonmissing text")
    return df


def prepare(csv, output):
    """Export the same ordered sample as three typed views, without resampling."""
    df = typed_csv(csv)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        for dataset in DATASETS:
            dataset_dir(dataset, output)
        print(f"Preserving verified prepared views at {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="linkedin_prepare_", dir=output.parent) as temporary:
        stage = Path(temporary)
        columns = {FULL: list(df), STRUCTURED: [c for c in df if c not in TEXT_COLUMNS],
                   TEXT: [c for c in df if c in TEXT_COLUMNS or c == TARGET]}
        files = {}
        for dataset, selected in columns.items():
            path = stage / dataset.value
            path.mkdir()
            df[selected].to_parquet(path / "data.parquet", index=False)
            metadata = dict(target=TARGET, task_type="reg", image_col=None,
                source=SOURCE, license="CC BY-SA 4.0", csv_sha256_lf=CSV_SHA256_LF,
                parent_dataset=FULL.name, view=dataset.name, rows=len(df))
            (path / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
            for file in path.iterdir():
                files[file.relative_to(stage).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        manifest = dict(csv_sha256_lf=CSV_SHA256_LF, rows=len(df), target=TARGET,
                        datasets=[d.name for d in DATASETS], files=files)
        (stage / "linkedin_preparation.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        if output.exists():
            if any(output.iterdir()):
                raise ValueError("Output changed during preparation")
            output.rmdir()
        stage.rename(output)
    print(f"Prepared 15,000 unchanged rows in three typed views at {output}")

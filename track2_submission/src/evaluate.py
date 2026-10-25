"""Run LinkedIn inside official MulTaBench and print/save the six-fold results.

Setup:  python src/evaluate.py --install
Run:    python src/evaluate.py
Check:  python src/evaluate.py --check-only

Python 3.11 and Git are required. GPU runs keep physical batch 256, accumulation
1 and the pinned official learner/preprocessing/splitting/LoRA source unchanged.
The dataset-registration patch and recipe are included visibly in this file so
no other project source files are needed. Third-party source/caches are created
in --work-dir (by default outside the submission, under the user's cache).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COMMIT = "ff0c9c8b40a2085a06a7771190e53a261341802f"
ORIGINAL_SOURCE = "bfe0ba3b846cb8bd719d43a586dc4d25b36088cd"
CSV_SHA256_LF = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
MODELS = {"tabm": "TabM", "cat": "CatBoost", "light": "LightGBM",
          "tabpfnv2": "TabPFN-v2", "tabpfnv2p5": "TabPFN-v2p5"}
CONDITIONS = ("structured", "unstructured", "joint_frozen", "joint_tar")
FULL = "REG_TEXT_LINKEDIN_SALARY"
STRUCTURED = FULL + "_STRUCTURED"
TEXT = FULL + "_TEXT"
TAR = dict(learning_rate=0.0001, epochs=50, lora_rank=16, layers=3,
           patience=3, weight_decay=0.001, batch_size=256)

# Filled from the same dependency pins used for the submitted experiment.
DEPENDENCIES = {'accelerate': '1.12.0',
 'autogluon-common': '1.5.0',
 'autogluon-core': '1.5.0',
 'autogluon-features': '1.5.0',
 'catboost': '1.2.8',
 'huggingface-hub': '0.36.0',
 'lightgbm': '4.6.0',
 'numpy': '2.3.5',
 'pandas': '2.3.3',
 'peft': '0.18.0',
 'pillow': '11.3.0',
 'psutil': '7.0.0',
 'pyarrow': '20.0.0',
 'pytabkit': '1.7.0',
 'python-dotenv': '1.2.1',
 'pyyaml': '6.0.3',
 'scikit-learn': '1.6.1',
 'scipy': '1.16.3',
 'setuptools': '80.9.0',
 'skrub': '0.10.1',
 'tabpfn': '9.2.0',
 'torch': '2.7.1',
 'torchvision': '0.22.1',
 'transformers': '4.57.3'}
EXPECTED_SPLITS = {
    "0": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "07ba06a6f4fbe10b89f7129cb7a712373d76208fb9f5445e2595f7eb3fe1aaa7",
        "test_indices_sha256": "77a427c3c601b8cba69e1883656b9c2d1a98f3f192ccebb7a1d32d2cdb6dde2c"
    },
    "1": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "2e5950a2a3e65b2b7cedd9803289e3160c7f4f1f57111952cc30e06d3bfc7557",
        "test_indices_sha256": "93541ab6094c5da950bfd18770c2ef08a6331cec2cfa0637daf63122424d6175"
    },
    "2": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "0f4fc1fba48d33b62770f4aa6b5d115b9e2f88e0dde9eb7d1dcb737b161038ff",
        "test_indices_sha256": "d793084ec59167625c9abc71f420f11022e25860c5a6b0965a6dfdfe507d4293"
    },
    "3": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "7d71284e5f7663f35d078d76c016ccf8f7cd99f882e9e34d075fbcd4d05a7202",
        "test_indices_sha256": "32f296db8f3fff8a88a847ddec95fdf06fef41db7fad3f8cdbcbb65269b31daa"
    },
    "4": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "b9bb6f7e49ec0ad849b20f072c08adb4dab1cbb7d366dc09a29f147e53f919bd",
        "test_indices_sha256": "5deb2e26356a289e728b9db966fcb55c9fba1002ea2ab89ca4045b25ab07c529"
    },
    "5": {
        "n_train": 10000,
        "n_test": 5000,
        "train_indices_sha256": "f8fc54c688fe0b60c0eaf7bdc5ecf46fbec3008ee436538254d43878d8d31d4c",
        "test_indices_sha256": "a2a4cab5ee88ffbbfbce310a8dd0c8d2bf502b59ba1cf00781d942167ccf9848"
    }
}

# Only registration, candidate metadata, and local dataset/cache routing change.
# No evaluation, preprocessing, split, learner, embedding-computation or LoRA
# implementation is patched. The official benchmark.py is executed directly.
DATASET_PATCH = r'''diff --git a/multabench/benchmark/load.py b/multabench/benchmark/load.py
index 5181b1a..81eb252 100644
--- a/multabench/benchmark/load.py
+++ b/multabench/benchmark/load.py
@@ -22,6 +22,7 @@ from multabench.benchmark.splits import DEFAULT_SIZE, SPLITS, Size, get_split
 from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
 from multabench.datasets.hub import DATA_PARQUET, IMAGES_DIR, extract_images, hf_repo_id, local_repo_dir
 from multabench.datasets.revisions import revision
+from multabench.datasets.linkedin_salary import DATASETS as LINKEDIN_DATASETS, dataset_dir as linkedin_dir
 from multabench.datasets.objects import DatasetSplit, MultimodalDataset, SupervisedTask
 from multabench.e5.constants import DEFAULT_TEXT_ENCODER
 from multabench.embeddings.hub import (CACHED_ENCODERS, EMBEDDINGS_NPZ, META_JSON, CachedEmbeddings, CachedEncoder,
@@ -49,6 +50,8 @@ def _missing_as_nan(df: pd.DataFrame) -> pd.DataFrame:
 def download_dataset(dataset_id: MulTaBenchDatasetID, data_dir: Optional[str] = None,
                      with_image_files: bool = True) -> str:
     """The dataset's folder, downloaded at its pinned revision or under data_dir; without image files if asked."""
+    if dataset_id in LINKEDIN_DATASETS:
+        return linkedin_dir(dataset_id, data_dir)
     if not with_image_files:
         return _download_table(dataset_id, data_dir=data_dir)
     repo_id = hf_repo_id(dataset_id)
@@ -166,6 +169,8 @@ def _load_columns(dataset_id: MulTaBenchDatasetID, columns: List[str], data_dir:
 
 
 def _download_table(dataset_id: MulTaBenchDatasetID, data_dir: Optional[str] = None) -> str:
+    if dataset_id in LINKEDIN_DATASETS:
+        return linkedin_dir(dataset_id, data_dir)
     repo_id = hf_repo_id(dataset_id)
     if data_dir is not None:
         return local_repo_dir(data_dir, repo_id, [DATA_PARQUET, METADATA_JSON])
diff --git a/multabench/datasets/all_datasets.py b/multabench/datasets/all_datasets.py
index bacd076..9a61888 100644
--- a/multabench/datasets/all_datasets.py
+++ b/multabench/datasets/all_datasets.py
@@ -47,6 +47,11 @@ class MulTaBenchDatasetID(Enum):
     REG_TEXT_VIDEO_GAMES_SALES = "core-text-reg-video-games-sales"
     REG_TEXT_ZOMATO_RESTAURANTS = "core-text-reg-zomato-restaurants"
 
+    # Local course candidate: one dataset, three feature-ablation views.
+    REG_TEXT_LINKEDIN_SALARY = "candidate-text-reg-linkedin-salary"
+    REG_TEXT_LINKEDIN_SALARY_STRUCTURED = "candidate-text-reg-linkedin-salary-structured"
+    REG_TEXT_LINKEDIN_SALARY_TEXT = "candidate-text-reg-linkedin-salary-text"
+
     # Extended text datasets
     MUL_TEXT_CONSUMER_COMPLAINT = "extended-text-cls-consumer-complaint"
     MUL_TEXT_BOX_OFFICE = "extended-text-cls-box-office"
@@ -152,6 +157,9 @@ def dataset_modality(dataset_id: MulTaBenchDatasetID) -> Modality:
 
 
 MULTABENCH_SOURCES: dict[MulTaBenchDatasetID, str] = {
+    MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY: "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings",
+    MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY_STRUCTURED: "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings",
+    MulTaBenchDatasetID.REG_TEXT_LINKEDIN_SALARY_TEXT: "https://www.kaggle.com/datasets/arshkon/linkedin-job-postings",
     MulTaBenchDatasetID.BIN_IMAGE_CELEB_ATTRACTIVENESS: "https://www.kaggle.com/datasets/jessicali9530/celeba-dataset",
     MulTaBenchDatasetID.BIN_IMAGE_HATEFUL_MEME: "https://www.kaggle.com/datasets/parthplc/facebook-hateful-meme-dataset",
     MulTaBenchDatasetID.BIN_IMAGE_MAMMOGRAPHY_CMMD: "https://www.kaggle.com/datasets/nguynththanhho/cmmd-mammography",
diff --git a/multabench/datasets/hub.py b/multabench/datasets/hub.py
index ed24040..23e01bc 100644
--- a/multabench/datasets/hub.py
+++ b/multabench/datasets/hub.py
@@ -7,6 +7,7 @@ from os.path import join
 from typing import List
 
 from multabench.datasets.all_datasets import MulTaBenchDatasetID
+from multabench.datasets.linkedin_salary import DATASETS as LINKEDIN_DATASETS
 
 HF_ORG = "multabench"
 DATA_PARQUET = "data.parquet"
@@ -15,6 +16,8 @@ IMAGE_SHARDS = "images-*.zip"
 
 
 def hf_repo_id(dataset_id: MulTaBenchDatasetID) -> str:
+    if dataset_id in LINKEDIN_DATASETS:
+        return f"local/{dataset_id.value}"
     return f"{HF_ORG}/{dataset_id.value}"
 
 
diff --git a/multabench/datasets/info.py b/multabench/datasets/info.py
index 80a0bda..26d4e81 100644
--- a/multabench/datasets/info.py
+++ b/multabench/datasets/info.py
@@ -13,6 +13,7 @@ PREFIX2TASK = {"BIN": SupervisedTask.BINARY, "MUL": SupervisedTask.MULTICLASS, "
 class Tier(StrEnum):
     CORE = "core"
     EXTENDED = "extended"
+    CANDIDATE = "candidate"
 
 
 @dataclass(frozen=True)
diff --git a/multabench/embeddings/hub.py b/multabench/embeddings/hub.py
index 92b321c..66aed29 100644
--- a/multabench/embeddings/hub.py
+++ b/multabench/embeddings/hub.py
@@ -23,6 +23,8 @@ from multabench.benchmark.splits import Size
 from multabench.datasets.all_datasets import Modality, MulTaBenchDatasetID, dataset_modality
 from multabench.datasets.hub import HF_ORG, local_repo_dir
 from multabench.datasets.revisions import revision
+from multabench.datasets.linkedin_salary import (DATASETS as LINKEDIN_DATASETS, FULL as LINKEDIN_FULL,
+                                                STRUCTURED as LINKEDIN_STRUCTURED, TEXT as LINKEDIN_TEXT, local_root)
 from multabench.dino.constants import IMAGE_ENCODERS, ImageEncoder
 from multabench.e5.constants import TEXT_ENCODERS, TextEncoder, format_e5_passage
 from multabench.utils.encoders import Encoder
@@ -70,6 +72,8 @@ def cached_encoder(model_name: str, tuned: bool) -> Optional[CachedEncoder]:
 
 def embeds_dataset(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID) -> bool:
     """Whether the benchmark's runs use the encoder on the dataset; datasets with images embed text with e5-small."""
+    if dataset_id == LINKEDIN_STRUCTURED:
+        return False
     modality = dataset_modality(dataset_id)
     if isinstance(encoder, ImageEncoder):
         return modality.has_images
@@ -78,6 +82,8 @@ def embeds_dataset(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID) -> b
 
 def embeddings_dir(dataset_id: MulTaBenchDatasetID, size: Optional[Size] = None, fold: Optional[int] = None) -> str:
     """<dataset> for a frozen encoder, <dataset>/<size>/fold_<k> for a tuned one."""
+    if dataset_id == LINKEDIN_TEXT:
+        dataset_id = LINKEDIN_FULL  # Same text/target values; share raw embeddings, never fitted PCA.
     if fold is None:
         return dataset_id.value
     return join(dataset_id.value, str(Size(size)), f"fold_{fold}")
@@ -145,6 +151,8 @@ def download_embeddings(encoder: CachedEncoder, dataset_id: MulTaBenchDatasetID,
     """The directory of the dataset's cached embeddings, the fold's for a tuned encoder; under data_dir if given."""
     if is_tuned(encoder) != (fold is not None):
         raise ValueError(f"{encoder} embeddings are cached {'per fold' if is_tuned(encoder) else 'once per dataset'}")
+    if dataset_id in LINKEDIN_DATASETS:
+        data_dir = str(local_root(data_dir))
     path_in_repo = embeddings_dir(dataset_id, size=size, fold=fold)
     files = [EMBEDDINGS_NPZ, META_JSON]
     repo_id = embeddings_repo_id(encoder)
'''
DATASET_MODULE = r'''"""Local LinkedIn salary candidate and its ablation views, inside MulTaBench.

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
'''
PATCH_FILES = {"multabench/datasets/all_datasets.py", "multabench/datasets/info.py",
               "multabench/datasets/hub.py", "multabench/benchmark/load.py",
               "multabench/embeddings/hub.py"}
ADDED_MODULE = "multabench/datasets/linkedin_salary.py"


def git(checkout, *arguments, data=None):
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=false", "-C", str(checkout), *arguments],
        input=data, stderr=subprocess.PIPE)


def prepare_official_source(checkout):
    """Register LinkedIn in a pinned upstream checkout; reject unrelated edits."""
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout",
                        "https://github.com/alanarazi7/MulTaBench.git", str(checkout)], check=True)
        git(checkout, "checkout", "--detach", OFFICIAL_COMMIT)
    if git(checkout, "rev-parse", "HEAD").decode().strip() != OFFICIAL_COMMIT:
        raise ValueError("MulTaBench checkout is not the required pinned official commit")
    patch = DATASET_PATCH.encode("utf-8")
    diff = git(checkout, "diff", "--no-ext-diff", "--binary", "HEAD")
    if not diff:
        if git(checkout, "status", "--porcelain", "--untracked-files=all").strip():
            raise ValueError("Refusing unrelated existing changes in the upstream checkout")
        git(checkout, "apply", "--check", "-", data=patch)
        git(checkout, "apply", "-", data=patch)
        (checkout / ADDED_MODULE).write_bytes(DATASET_MODULE.encode("utf-8"))
    elif diff != patch:
        raise ValueError("Existing upstream changes differ from the dataset-only patch")
    if (checkout / ADDED_MODULE).read_bytes() != DATASET_MODULE.encode("utf-8"):
        raise ValueError("Native dataset recipe differs")
    changed = set(git(checkout, "diff", "--name-only", "HEAD").decode().splitlines())
    added = set(git(checkout, "ls-files", "--others", "--exclude-standard").decode().splitlines())
    if changed != PATCH_FILES or added != {ADDED_MODULE}:
        raise ValueError("Unexpected changes to official source")


def run(checkout, env, *arguments):
    subprocess.run([sys.executable, *map(str, arguments)], cwd=checkout, env=env, check=True)


def check_versions():
    for package, expected in DEPENDENCIES.items():
        actual = version(package).split("+")[0]
        if actual != expected:
            raise ValueError(f"{package}: expected {expected}, found {actual}; use --install in Python 3.11")


def validate_native_data(csv, prepared):
    """Validate every native view and every official split, before training."""
    import numpy as np
    import pandas as pd
    from multabench.benchmark.load import load_multabench_dataset, load_split
    from multabench.datasets.linkedin_salary import DATASETS, FULL, STRUCTURED, TEXT, prepare, typed_csv
    from multabench.e5.constants import E5_LORA
    from multabench.finetune.train_args import DATALOADER_WORKERS

    assert asdict(E5_LORA) == TAR and DATALOADER_WORKERS == 4
    prepare(csv, prepared)
    original = typed_csv(csv)
    y = original["log_salary"]
    x = original.drop(columns="log_salary")
    views = {FULL: x, STRUCTURED: x.drop(columns=["title", "description"]),
             TEXT: x[["title", "description"]]}
    for dataset in DATASETS:
        loaded = load_multabench_dataset(dataset)
        pd.testing.assert_frame_equal(loaded.x, views[dataset], check_exact=True)
        pd.testing.assert_series_equal(loaded.y, y, check_exact=True)
        for fold in range(6):
            split = load_split(dataset, fold=fold, size="10k")
            reference = EXPECTED_SPLITS[str(fold)]
            for label, frame, target in (("train", split.x_train, split.y_train),
                                          ("test", split.x_test, split.y_test)):
                digest = hashlib.sha256(np.asarray(frame.index, dtype="<i8").tobytes()).hexdigest()
                assert len(frame) == reference[f"n_{label}"]
                assert digest == reference[f"{label}_indices_sha256"]
                pd.testing.assert_frame_equal(frame, views[dataset].loc[frame.index], check_exact=True)
                pd.testing.assert_series_equal(target, y.loc[frame.index], check_exact=True)
    print("Verified exact CSV, all three native views, all six official splits and batch-256 defaults.", flush=True)


def report_results(results, csv, destination):
    """Print means/deltas and save the performance CSV; never mix experiments."""
    import numpy as np
    import pandas as pd
    from multabench.benchmark.splits import get_split

    if hashlib.sha256(csv.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != CSV_SHA256_LF:
        raise ValueError("Expected the unchanged curated CSV")
    y = pd.read_csv(csv, usecols=["log_salary"])["log_salary"].to_numpy()
    variances = {}
    for fold in range(6):
        train, test = get_split(y, False, fold, size="10k")
        for label, indices in (("train", train), ("test", test)):
            digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            if digest != EXPECTED_SPLITS[str(fold)][f"{label}_indices_sha256"]:
                raise ValueError("Official report split differs from the original experiment")
        variances[fold] = float(np.var(y[test], ddof=0))
    native = sorted((results / "runs").glob("*.json"))
    original = sorted(results.glob("folds/*/runs/*.json"))
    if native and original:
        raise ValueError("Do not mix native CLI results with the original VM experiment")
    paths = native or original
    if not paths:
        raise ValueError("No result JSONs found")
    records = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        fold = row.get("fold")
        if (row.get("status") != "ok" or type(fold) is not int or fold not in range(6)
                or row.get("n_train") != 10000 or row.get("n_test") != 5000
                or row.get("metric") != "root_mean_squared_error" or row.get("size") != "10k"
                or row.get("model") not in MODELS.values()):
            raise ValueError("Failed/unexpected evaluation: " + str(path))
        if native:
            dataset, encoder = row.get("dataset"), row.get("text_encoder")
            condition = {STRUCTURED: "structured", TEXT: "unstructured"}.get(dataset)
            if dataset == FULL:
                condition = {"e5-small": "joint_frozen", "e5-small-tar": "joint_tar"}.get(encoder)
            # The unchanged official CLI writes a seven-character Git revision.
            if row.get("git") not in {OFFICIAL_COMMIT, OFFICIAL_COMMIT[:7]} or encoder not in {"e5-small", "e5-small-tar"}:
                raise ValueError("Native result source/encoder differs: " + str(path))
        else:
            condition = row.get("condition")
            if (row.get("dataset") != "linkedin_salary" or row.get("git") != ORIGINAL_SOURCE
                    or row.get("protocol") != "linkedin_vm_physical256_v1"
                    or row.get("multabench_commit") != OFFICIAL_COMMIT
                    or row.get("short_name") not in MODELS
                    or MODELS[row["short_name"]] != row["model"]):
                raise ValueError("Original experiment provenance differs: " + str(path))
            if condition == "joint_tar" and (row.get("tar_parameters") != TAR
                    or row.get("training_arguments", {}).get("per_device_train_batch_size") != 256
                    or row.get("training_arguments", {}).get("gradient_accumulation_steps") != 1):
                raise ValueError("Original TAR settings differ: " + str(path))
        if condition not in CONDITIONS:
            raise ValueError("Unexpected condition: " + str(path))
        rmse = row.get("test_error")
        if isinstance(rmse, bool) or not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0:
            raise ValueError("Invalid RMSE: " + str(path))
        r2 = 1 - rmse**2 / variances[fold]
        if original and not math.isclose(row.get("r2", math.nan), r2, abs_tol=1e-10):
            raise ValueError("Saved R-squared disagrees with RMSE/test target: " + str(path))
        records.append(dict(model=row["model"], condition=condition, fold=fold, r2=r2, rmse=rmse))
    frame = pd.DataFrame(records)
    if frame.duplicated(["model", "condition", "fold"]).any():
        raise ValueError("Duplicate evaluations")
    expected = {(model, condition) for model in MODELS.values() for condition in CONDITIONS}
    completed = [int(fold) for fold, group in frame.groupby("fold")
                 if set(zip(group.model, group.condition)) == expected]
    if not completed:
        print(f"PRELIMINARY: {len(frame)}/120 evaluations; no completed fold yet.", flush=True)
        return False
    selected = frame[frame.fold.isin(completed)]
    aggregate = selected.groupby(["model", "condition"]).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        test_error_mean=("rmse", "mean"), test_error_std=("rmse", "std"), n_folds=("fold", "size")).reset_index()
    final = len(frame) == 120 and sorted(completed) == list(range(6))
    aggregate.insert(0, "dataset", "linkedin_salary")
    aggregate["metric"] = "root_mean_squared_error"
    aggregate["experiment"] = "native_official_cli" if native else "original_vm_physical256_v1"
    aggregate["status"] = "final" if final else "preliminary"
    destination.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(destination, index=False)
    table = selected.groupby(["model", "condition"]).r2.mean().unstack().reindex(
        index=list(MODELS.values()), columns=list(CONDITIONS))
    table["Joint delta"] = table.joint_frozen - table[["structured", "unstructured"]].max(axis=1)
    table["TAR delta"] = table.joint_tar - table.joint_frozen
    joint, tar = int((table["Joint delta"] > 0).sum()), int((table["TAR delta"] > 0).sum())
    print(f"\n{'FINAL' if final else 'PRELIMINARY'}: {len(frame)}/120 evaluations; completed folds {completed}.", flush=True)
    print("R-squared on log salary; means use fully completed folds only.")
    print(table.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Joint Signal: {joint}/5; TAR Gain: {tar}/5; need >=3/5 independently, raw deltas >0.")
    print("Final verdict: " + ("PASS" if joint >= 3 and tar >= 3 else "FAIL") if final else "Final verdict pending all six folds.")
    print(f"Saved {destination}", flush=True)
    return final


def execute(args):
    csv, output = args.csv.resolve(), args.output.resolve()
    results = args.results.resolve() if args.results else args.work_dir.resolve() / "evaluation"
    if args.report_only:
        report_results(results, csv, output)
        return
    if sys.version_info[:2] != (3, 11):
        raise ValueError("Use Python 3.11 for the pinned environment")
    if args.install:
        packages = [f"{name}=={value}" for name, value in DEPENDENCIES.items()]
        packages.append("multabench[bench,catboost,lightgbm,pytabkit,tabpfn] @ "
                        f"git+https://github.com/alanarazi7/MulTaBench@{OFFICIAL_COMMIT}")
        subprocess.run([sys.executable, "-m", "pip", "install", *packages], check=True)
        return
    check_versions()
    checkout = args.work_dir.resolve() / "MulTaBench"
    prepared = args.work_dir.resolve() / "dataset"
    prepare_official_source(checkout)
    env = {**os.environ, "PYTHONPATH": str(checkout), "MPLBACKEND": "Agg",
           "MULTABENCH_LINKEDIN_DATA_DIR": str(prepared), "CUDA_VISIBLE_DEVICES": str(args.gpu),
           "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    # The child imports this one file solely for validation; it imports no tt_curation helpers.
    check = ("import runpy,sys; m=runpy.run_path(sys.argv[1]); "
             "m['validate_native_data'](sys.argv[2],sys.argv[3])")
    run(checkout, env, "-c", check, Path(__file__).resolve(), csv, prepared)
    if args.check_only:
        return
    run(checkout, env, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU required'; "
        "assert torch.cuda.device_count()==1, 'Expose one GPU'; print('GPU:',torch.cuda.get_device_name(0))")
    results.mkdir(parents=True, exist_ok=True)
    manifest = dict(protocol="linkedin_native_official_batch256_v1", official_commit=OFFICIAL_COMMIT,
        csv_sha256_lf=CSV_SHA256_LF, tar=TAR, folds=list(range(6)), models=MODELS,
        conditions=CONDITIONS, evaluate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path = results / "experiment.json"
    # JSON round-trip makes tuples/lists identical when checking an existing experiment.
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing evaluation directory belongs to a different experiment")
    else:
        if any(results.iterdir()):
            raise ValueError("Use an empty results directory for a new experiment")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    run(checkout, env, "-m", "multabench.embeddings.embed", "--encoder", "e5-small",
        "--dataset_name", FULL, "--output_dir", prepared, "--device", "cuda")
    for fold in range(6):
        print(f"\n===== FOLD {fold}: official batch 256; 20 evaluations =====", flush=True)
        run(checkout, env, "-m", "multabench.embeddings.embed", "--encoder", "e5-small-tar",
            "--dataset_name", FULL, "--fold", fold, "--size", "10k", "--output_dir", prepared, "--device", "cuda")
        for model in MODELS:
            for condition, dataset, encoder in (("structured", STRUCTURED, "e5-small"),
                    ("unstructured", TEXT, "e5-small"), ("joint_frozen", FULL, "e5-small"),
                    ("joint_tar", FULL, "e5-small-tar")):
                path = results / "runs" / f"{model}_{dataset}_{encoder}_none_10k_{fold}.json"
                run(checkout, env, "benchmark.py", "--model", model, "--dataset_name", dataset,
                    "--fold", fold, "--size", "10k", "--text_encoder", encoder,
                    "--device", "cuda", "--output_dir", results / "runs")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") != "ok":
                    raise ValueError("Evaluation failed: " + str(path))
        # Reporting is separate from training and runs in the same pinned native environment.
        code = ("import runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); "
                "m['report_results'](Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))")
        run(checkout, env, "-c", code, Path(__file__).resolve(), results, csv, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results.csv")
    parser.add_argument("--work-dir", type=Path, default=Path.home() / ".cache/linkedin_track2")
    parser.add_argument("--results", type=Path, help="Native evaluation directory; for --report-only also accepts original VM fold results")
    parser.add_argument("--gpu", type=int, default=0)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--install", action="store_true", help="Install the pinned dependencies in this Python environment and exit")
    modes.add_argument("--check-only", action="store_true", help="Prepare/validate native data and six splits without training")
    modes.add_argument("--report-only", action="store_true", help="Print/save already computed results without training")
    execute(parser.parse_args())

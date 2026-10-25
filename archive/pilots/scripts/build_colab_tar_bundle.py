"""Build the TAR-only Colab notebook and a portable bundle of current code/data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def build_notebook():
    cells = []

    def cell(kind, source):
        item = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell("markdown", """
        # Steam + LinkedIn: joint TAR only

        Upload this notebook to Google Colab and select **Runtime > Change runtime type > T4 GPU** (or another GPU).
        You also need the accompanying `steam_linkedin_tar_bundle.zip`; it contains the exact prepared datasets,
        code, configs, and CPU TF-IDF baselines. No GitHub or TabPFN token is needed.

        **Default pilot:** each dataset's existing 15,000 rows; outer fold 0 (10K train / 5K test);
        CatBoost + LightGBM; one shared target-tuned E5-small-v2 encoder per dataset/fold.
        Only `joint_tar` runs. The cap is **10 epochs with early stopping**, versus 50 in the official protocol.
        This pilot is for deciding what to run next, not a full five-learner/six-fold curation result.

        **Comparison limitation:** the previous three conditions used **TF-IDF**, not frozen E5. A difference between
        those scores and E5-TAR combines an encoder change with tuning. A matching frozen-E5 joint control is
        available below but **off by default**, as requested.

        **Resuming:** results and checkpoints save to Google Drive. Completed fits are skipped; a completed tuned
        adapter is reused; interrupted tuning resumes from the latest saved epoch checkpoint (work after that
        checkpoint must be repeated). Run setup again after reconnecting, with the same settings and results folder.
        Settings and dataset checksums separate incompatible experiments automatically.

        Colab limits vary and are not guaranteed; see [Google's resource-limit FAQ](https://research.google.com/colaboratory/faq.html#resource-limits).
        Allow roughly 1–3 hours for both default pilots on a T4, with considerable uncertainty. Run Steam first,
        inspect its epoch time, then decide whether to start LinkedIn. No GPU training was run when preparing this notebook.
    """)
    cell("code", """
        from pathlib import Path

        # Start small. To expand later, use FOLDS = [0, 1, 2]. Finished folds are skipped.
        FOLDS = [0]
        MODELS = ["light", "cat"]
        MAX_EPOCHS = 10        # 5 for a shorter exploratory pilot; 50 for the official epoch cap.
        MICRO_BATCH = 16       # Effective training batch stays 256 through gradient accumulation.
        INCLUDE_FROZEN_E5 = False  # Optional matching joint_frozen control; default is TAR ONLY.

        RESULTS_ROOT = Path("/content/drive/MyDrive/text-tabular-curation/colab_tar_pilot")
        PROJECT = Path("/content/text_tabular_tar_project")
        BUNDLE = Path("/content/steam_linkedin_tar_bundle.zip")
    """)
    cell("markdown", """
        ## 1. Mount Drive and upload the bundle once

        Select the accompanying ZIP when the upload picker appears. It is saved to Drive so reconnecting does not
        require another upload. Datasets are extracted to the Colab VM for fast reads; results/checkpoints stay in Drive.
    """)
    cell("code", """
        import hashlib
        import json
        import os
        import shutil
        import zipfile
        from google.colab import drive, files

        drive.mount("/content/drive")
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        saved_bundle = RESULTS_ROOT.parent / "inputs" / BUNDLE.name
        if not BUNDLE.exists():
            if saved_bundle.exists():
                shutil.copy2(saved_bundle, BUNDLE)
            else:
                uploads = files.upload()
                selected = [(name, content) for name, content in uploads.items() if name.endswith(".zip")]
                if len(selected) != 1:
                    raise ValueError("Please upload exactly one project bundle ZIP.")
                BUNDLE.write_bytes(selected[0][1])
                del uploads, selected
        saved_bundle.parent.mkdir(parents=True, exist_ok=True)
        if not saved_bundle.exists() or hashlib.sha256(saved_bundle.read_bytes()).digest() != hashlib.sha256(BUNDLE.read_bytes()).digest():
            shutil.copy2(BUNDLE, saved_bundle)
        PROJECT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BUNDLE) as archive:
            for member in archive.infolist():
                if not (PROJECT / member.filename).resolve().is_relative_to(PROJECT.resolve()):
                    raise ValueError("Invalid archive path")
            archive.extractall(PROJECT)
        manifest = json.loads((PROJECT / "bundle_manifest.json").read_text())
        for relative, expected in manifest["sha256"].items():
            actual = hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch: {relative}")
        os.chdir(PROJECT)
        print("Verified bundle:", len(manifest["sha256"]), "files. Project:", PROJECT)
    """)
    cell("markdown", """
        ## 2. Install the pinned environment and check the GPU

        This uses an isolated Python 3.11 environment; it does not replace Colab's notebook packages.
        It installs only the E5 stack and the two tree learners. Setup may take several minutes on first use.
    """)
    cell("code", """
        import subprocess
        import sys

        def run_checked(command, label):
            print(f"--- {label} ---", flush=True)
            child_env = os.environ.copy()
            child_env["MPLBACKEND"] = "Agg"  # The isolated environment has no Colab inline backend.
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1, env=child_env)
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"{label} failed. The underlying error is printed immediately above.")

        PYTHON = Path("/content/tt_tar_venv/bin/python")
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("No NVIDIA runtime found. Select Runtime > Change runtime type > a GPU.")
        run_checked(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], "GPU hardware")
        run_checked([sys.executable, "-m", "pip", "install", "-q", "uv"], "Install uv")
        if not PYTHON.exists():
            run_checked([sys.executable, "-m", "uv", "venv", "--python", "3.11", str(PYTHON.parent.parent)], "Create Python 3.11 environment")
        run_checked([sys.executable, "-m", "uv", "pip", "install", "--python", str(PYTHON),
                        "-c", str(PROJECT / "requirements/constraints.txt"),
                        "-r", str(PROJECT / "requirements/colab_tar.txt"), "-e", str(PROJECT)], "Install pinned packages")
        run_checked([str(PYTHON), "-u", "-c",
                        "import torch; print('PyTorch:', torch.__version__); "
                        "print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); "
                        "assert torch.cuda.is_available(), 'Choose a GPU runtime before training'; "
                        "print('GPU:', torch.cuda.get_device_name(0))"], "PyTorch GPU check")
        run_checked([str(PYTHON), "-u", "-c",
                        "from tt_curation.runner import _model_class; _model_class('light'); _model_class('cat'); "
                        "print('Runner and both learners imported successfully')"], "Project imports")
    """)
    cell("code", """
        import signal

        def run_dataset(config):
            command = [str(PYTHON), "-u", str(PROJECT / "scripts/colab_tar_compare.py"),
                       "--config", str(PROJECT / config), "--results_root", str(RESULTS_ROOT),
                       "--folds", ",".join(map(str, FOLDS)), "--models", ",".join(MODELS),
                       "--max_epochs", str(MAX_EPOCHS), "--micro_batch", str(MICRO_BATCH)]
            if INCLUDE_FROZEN_E5:
                command.append("--include_frozen")
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            env["PYTHONUNBUFFERED"] = "1"
            env["HF_HOME"] = str(RESULTS_ROOT.parent / "hf_cache")
            process = subprocess.Popen(command, cwd=PROJECT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                if process.wait() != 0:
                    raise RuntimeError("Run stopped. Inspect the error above; completed work is saved to Drive.")
            except KeyboardInterrupt:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=10)
                raise
    """)
    cell("markdown", """
        ## 3. Steam: run joint TAR

        One review text column. Watch the first epoch's elapsed time to refine the time estimate.
        Both learners share the same tuned encoder. You can stop after this cell and inspect the summary.
    """)
    cell("code", 'run_dataset("configs/steam_recommended.yaml")')
    cell("markdown", """
        ## 4. LinkedIn: run joint TAR

        Two text columns: title and salary-masked description. This is likely to take longer than Steam.
        Run this cell only when you want to spend the remaining GPU time on LinkedIn.
    """)
    cell("code", 'run_dataset("configs/linkedin_salary.yaml")')
    cell("markdown", """
        ## 5. Inspect the comparison (also works after Steam alone)

        The TF-IDF comparison uses **the same folds** that have completed for TAR, not the earlier three-fold mean
        when only fold 0 has run. Only `isolated_TAR_gain`, when a matching frozen-E5 control was enabled, measures
        the gain from fine-tuning alone. R² and AUROC are different scales; compare within each dataset.
    """)
    cell("code", """
        import pandas as pd
        from IPython.display import display

        run_checked([str(PYTHON), str(PROJECT / "scripts/colab_tar_compare.py"),
                     "--results_root", str(RESULTS_ROOT), "--summary_only"], "Build comparison")
        table = pd.read_csv(RESULTS_ROOT / "tar_comparison.csv")
        display(table)
        print("Saved results and checkpoints:", RESULTS_ROOT)
    """)
    cell("markdown", """
        ## 6. Download the small results ZIP

        This exports reports, run JSONs, and manifests, leaving the large training checkpoints in Drive.
        After downloading, use **Runtime > Disconnect and delete runtime** to release the GPU.
    """)
    cell("code", """
        export = Path("/content/steam_linkedin_tar_results.zip")
        with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in RESULTS_ROOT.rglob("*"):
                if path.is_file() and "checkpoints" not in path.relative_to(RESULTS_ROOT).parts:
                    archive.write(path, path.relative_to(RESULTS_ROOT))
        files.download(str(export))
    """)
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "notebooks/compare_steam_linkedin_tar_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def build_bundle():
    notebook = build_notebook()
    output = ROOT / "artifacts/steam_linkedin_tar_bundle.zip"
    output.parent.mkdir(exist_ok=True)
    paths = [ROOT / name for name in ["pyproject.toml", "README.md", "LICENSE", "requirements/constraints.txt",
             "requirements/colab_tar.txt", "scripts/colab_tar_compare.py", "configs/steam_recommended.yaml",
             "configs/linkedin_salary.yaml", "data/processed/steam_recommended.csv",
             "data/processed/linkedin_salary.csv", "data/processed/steam_recommended.summary.json"]]
    paths += sorted((ROOT / "src/tt_curation").glob("*.py"))
    for dataset in ["steam_recommended", "linkedin_salary"]:
        paths += sorted((ROOT / "results" / f"{dataset}_screen_tfidf" / "runs").glob("*.json"))
    manifest = {"purpose": "Steam and LinkedIn TAR-only Colab pilot", "sha256": {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Notebook: {notebook}")
    print(f"Bundle: {output} ({output.stat().st_size / 1e6:.1f} MB, {len(paths)} source files)")


if __name__ == "__main__":
    build_bundle()

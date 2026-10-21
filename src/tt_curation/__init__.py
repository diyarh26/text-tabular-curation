"""Curating text-tabular datasets with the MulTaBench curation pipeline (NLP 097215, Track 2)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The MulTaBench commit our runs use; keep in sync with pyproject.toml and requirements/constraints.txt.
MULTABENCH_COMMIT = "ff0c9c8b40a2085a06a7771190e53a261341802f"

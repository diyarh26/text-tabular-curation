"""A dataset spec: one YAML file per dataset, saying where its CSV is and what each column is."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from tt_curation import REPO_ROOT

TASKS = ("classification", "regression")


@dataclass
class DatasetSpec:
    name: str
    csv: Path
    target: str
    task: str
    text_columns: List[str]
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    date_columns: List[str] = field(default_factory=list)
    drop_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {self.task!r}")
        if not self.text_columns:
            raise ValueError("A text-tabular dataset needs at least one text column (text_columns)")
        declared = self.text_columns + self.categorical_columns + self.numerical_columns + self.date_columns
        duplicated = sorted({c for c in declared if declared.count(c) > 1})
        if duplicated:
            raise ValueError(f"Columns declared under more than one type: {duplicated}")
        if self.target in declared + self.drop_columns:
            raise ValueError(f"The target {self.target!r} can't also be a feature or dropped")

    @property
    def is_cls(self) -> bool:
        return self.task == "classification"


def load_spec(path: str | Path) -> DatasetSpec:
    """Read a spec; a relative csv path is relative to the repo root."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    csv = Path(raw.pop("csv"))
    if not csv.is_absolute():
        csv = REPO_ROOT / csv
    return DatasetSpec(csv=csv, **raw)

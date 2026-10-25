"""Load a dataset's CSV into the typed frame MulTaBench expects.

MulTaBench types a feature by its dtype alone: category -> categorical, numeric/bool -> numerical, datetime -> date and
anything else (object) -> free text. So every column must be cast explicitly here, otherwise a string category such as
"city" would be embedded as text, and would leak into the text-only condition.
"""
import warnings
from dataclasses import dataclass
from typing import List

import pandas as pd
from pandas.api.types import is_numeric_dtype

from tt_curation.spec import DatasetSpec

# TabPFN v2 and v2.5 take at most 10 classes.
TABPFN_MAX_CLASSES = 10


@dataclass
class LoadedDataset:
    x: pd.DataFrame
    y: pd.Series
    spec: DatasetSpec

    @property
    def text_columns(self) -> List[str]:
        return list(self.spec.text_columns)

    @property
    def structured_columns(self) -> List[str]:
        return [c for c in self.x.columns if c not in self.spec.text_columns]


def load_dataset(spec: DatasetSpec) -> LoadedDataset:
    df = pd.read_csv(spec.csv, low_memory=False)
    missing = [c for c in [spec.target, *spec.text_columns, *spec.categorical_columns, *spec.numerical_columns,
                           *spec.date_columns, *spec.drop_columns] if c not in df.columns]
    if missing:
        raise KeyError(f"Columns in the spec but not in {spec.csv.name}: {missing}")
    df = df.drop(columns=spec.drop_columns)

    n_null_target = int(df[spec.target].isna().sum())
    if n_null_target:
        warnings.warn(f"Dropping {n_null_target} rows with a missing target")
        df = df[df[spec.target].notna()].reset_index(drop=True)

    y = df.pop(spec.target)
    y = y.astype(float) if spec.task == "regression" else y.astype(str)

    undeclared = [c for c in df.columns
                  if c not in spec.text_columns + spec.categorical_columns + spec.numerical_columns + spec.date_columns]
    numerical = list(spec.numerical_columns) + [c for c in undeclared if is_numeric_dtype(df[c])]
    categorical = list(spec.categorical_columns) + [c for c in undeclared if not is_numeric_dtype(df[c])]
    if undeclared:
        warnings.warn(f"Undeclared columns typed by dtype (declare them in the spec to be sure): "
                      f"numerical={[c for c in undeclared if c in numerical]}, "
                      f"categorical={[c for c in undeclared if c in categorical]}")

    x = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col in spec.text_columns:
            x[col] = df[col].astype(object).where(df[col].notna(), None)
            x[col] = x[col].map(lambda v: v if v is None else str(v))
        elif col in categorical:
            x[col] = df[col].astype(object).where(df[col].notna(), None).map(
                lambda v: v if v is None else str(v)).astype("category")
        elif col in numerical:
            x[col] = _to_numeric(df[col], col)
        elif col in spec.date_columns:
            x[col] = pd.to_datetime(df[col], errors="coerce")
    return LoadedDataset(x=x, y=y, spec=spec)


def _to_numeric(values: pd.Series, col: str) -> pd.Series:
    if values.dtype == bool:
        return values.astype(float)
    converted = pd.to_numeric(values, errors="coerce")
    n_lost = int(converted.isna().sum() - values.isna().sum())
    if n_lost:
        warnings.warn(f"{col!r}: {n_lost} non-numeric values became missing")
    return converted


def describe(data: LoadedDataset) -> List[str]:
    """A summary of the dataset, with warnings for what tends to break the pipeline or the curation criteria."""
    spec, x, y = data.spec, data.x, data.y
    lines = [f"Dataset {spec.name}: {len(x)} rows, task={spec.task}",
             f"  text columns ({len(data.text_columns)}): {data.text_columns}",
             f"  structured columns ({len(data.structured_columns)}): {data.structured_columns}"]
    for col in data.text_columns:
        lengths = x[col].dropna().str.split().str.len()
        lines.append(f"  text {col!r}: {x[col].isna().mean():.0%} missing, {x[col].nunique()} unique, "
                     f"median {lengths.median():.0f} words")
    if spec.is_cls:
        counts = y.value_counts()
        lines.append(f"  classes ({len(counts)}): " + ", ".join(f"{k}={v}" for k, v in counts.head(10).items()))
        if len(counts) > TABPFN_MAX_CLASSES:
            lines.append(f"  ⚠️ {len(counts)} classes: TabPFN v2/v2.5 take at most {TABPFN_MAX_CLASSES}, so 2 of the 5 "
                         f"learners would fail")
        if counts.min() < 3:
            lines.append("  ⚠️ a class has fewer than 3 rows, so the folds can't be stratified")
    else:
        lines.append(f"  target: mean {y.mean():.4g}, std {y.std():.4g}, min {y.min():.4g}, max {y.max():.4g}")
    if not data.structured_columns:
        lines.append("  ⚠️ no structured columns: the structured-only condition can't run")
    if len(x) < 500:
        lines.append("  ⚠️ fewer than 500 rows: the differences between conditions will be noisy")
    if len(x) > 15_000:
        lines.append("  ℹ️ more than 15K rows: each fold is capped at 10K train / 5K test rows (MulTaBench's 10k size)")
    return lines

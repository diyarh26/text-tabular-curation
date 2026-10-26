"""Prepare LinkedIn Salary from the downloaded relational source CSVs.

The supplied submission CSV is immutable. By default a newly prepared candidate
must match it row-for-row and value-for-value before any output is written.
No language filtering, translation or structured missing-value imputation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

SEED = 42
MAX_ROWS = 15000
MAX_WORDS = 500
RARE_COUNT = 20
APPROVED_SHA256 = "8880f84563535c60eda01be5b52010dea6f80e318f228ade00166d7cb66d57fa"
TEXT_COLUMNS = ["title", "description"]
COLUMNS = [
    "title",
    "description",
    "experience_level",
    "work_type",
    "remote_allowed",
    "state",
    "application_type",
    "company_size",
    "company_employees",
    "company_followers",
    "company_industry",
    "job_industry",
    "n_skills",
    "n_benefits",
    "log_salary",
]

# Frozen dataset specification, confirmed against the previous preparation code
# after the user explicitly authorized that inspection. Evaluation uses only the
# official July benchmark implementation. Keep expression order: it affects text.
EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")
NUMBER = r"\d[\d,]*(?:\.\d+)?"
NUMBER_RANGE = NUMBER + r"(?:\s*(?:-|\u2013|to)\s*" + NUMBER + r")?"
PERIOD = (
    r"(?:hr|hour|hourly|yr|year|yearly|annum|annually|wk|week|weekly|mo|month|monthly)"
)
PAY_PERIOD = r"(?:hourly|yearly|annually|annual|monthly|weekly|biweekly)"
MONEY_PATTERNS = [
    re.compile(r"(?:US\s?)?\$\s*" + NUMBER + r"(?:\s*(?:mm|bn|[kKmMbB]))?"),
    re.compile(r"\bUSD\s*" + NUMBER + r"(?:\s*[kKmM])?", re.I),
    re.compile(NUMBER_RANGE + r"\s*(?:USD|dollars)", re.I),
    re.compile(NUMBER_RANGE + r"\s*(?:/|per|an|a)\s*" + PERIOD + r"\b", re.I),
    re.compile(NUMBER_RANGE + r"\s*" + PAY_PERIOD, re.I),
    re.compile(r"\b\d+\.\d{2}\s*(?:-|\u2013|to)\s*\d+\.\d{2}\b"),
    re.compile(r"\b" + NUMBER + r"\s*[kK]\b"),
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?"),
    re.compile(r"\b\d{5,7}(?:\.\d+)?\b"),
]


def clean_text(value) -> str:
    if not isinstance(value, str):
        return ""
    text = str(value)
    text = EMAIL.sub("[email]", text)
    text = PHONE.sub("[phone]", text)
    for expression in MONEY_PATTERNS:
        text = expression.sub("[amount]", text)
    return " ".join(text.split())


def read_source(root: Path, name: str) -> pd.DataFrame:
    return (
        pd.read_csv(root / name, low_memory=False)
        if name == "postings.csv"
        else pd.read_csv(root / name)
    )


def prepare(raw: Path) -> tuple[pd.DataFrame, dict]:
    """Maps preserve posting order; lookup ties retain their source file order."""
    postings = read_source(raw, "postings.csv")
    audit = {"raw_postings": len(postings)}
    salary = postings["normalized_salary"]
    if not pd.api.types.is_numeric_dtype(salary):
        raise ValueError("Source normalized_salary must be numeric")
    keep = (
        salary.notna()
        & postings["currency"].eq("USD")
        & salary.between(15000, 1000000, inclusive="both")
    )
    frame = postings.loc[keep].copy()
    audit["eligible_usd_salary"] = len(frame)
    frame["log_salary"] = np.log(salary.loc[keep])
    frame["experience_level"] = frame["formatted_experience_level"]
    frame["work_type"] = frame["formatted_work_type"]
    frame["remote_allowed"] = frame["remote_allowed"].fillna(0).astype(int)
    frame["state"] = frame["location"].str.extract(r",\s*([A-Z]{2})$", expand=False)

    companies = read_source(raw, "companies/companies.csv").drop_duplicates(
        "company_id", keep="first"
    )
    frame["company_size"] = frame["company_id"].map(
        companies.set_index("company_id")["company_size"]
    )
    employees = read_source(raw, "companies/employee_counts.csv")
    # Native source recipe selects the last nonmissing value per company/column
    # after sorting timestamps; it is not an arbitrary whole-row deduplication.
    employees = employees.sort_values("time_recorded").groupby("company_id").last()
    frame["company_employees"] = frame["company_id"].map(employees["employee_count"])
    frame["company_followers"] = frame["company_id"].map(employees["follower_count"])

    industries = (
        read_source(raw, "companies/company_industries.csv")
        .groupby("company_id")["industry"]
        .first()
    )
    frame["company_industry"] = frame["company_id"].map(industries)
    job_industries = (
        read_source(raw, "jobs/job_industries.csv")
        .groupby("job_id")["industry_id"]
        .first()
    )
    names = read_source(raw, "mappings/industries.csv").set_index("industry_id")[
        "industry_name"
    ]
    frame["job_industry"] = frame["job_id"].map(job_industries.map(names))
    for column, source in (
        ("n_skills", "jobs/job_skills.csv"),
        ("n_benefits", "jobs/benefits.csv"),
    ):
        counts = read_source(raw, source).groupby("job_id", sort=False).size()
        frame[column] = frame["job_id"].map(counts).fillna(0).astype(int)

    for column in TEXT_COLUMNS:
        frame[column] = frame[column].map(clean_text)
    frame["description"] = frame["description"].map(
        lambda text: " ".join(text.split()[:MAX_WORDS])
    )
    frame = frame.loc[frame["title"].ne("") & frame["description"].ne("")]
    audit["nonmissing_cleaned_text"] = len(frame)
    frame = frame.drop_duplicates("description", keep="first")
    audit["unique_cleaned_truncated_descriptions"] = len(frame)
    frame = frame.sample(n=min(MAX_ROWS, len(frame)), replace=False, random_state=SEED)
    for column in ("state", "company_industry", "job_industry"):
        frequencies = frame[column].value_counts(dropna=True)
        rare = frame[column].notna() & frame[column].map(frequencies).lt(RARE_COUNT)
        frame.loc[rare, column] = "Other"
    frame = frame[COLUMNS].reset_index(drop=True)
    audit.update(
        rows=len(frame),
        seed=SEED,
        language_filter=False,
        missing={c: int(frame[c].isna().sum()) for c in frame},
        description_max_words=int(frame.description.str.split().str.len().max()),
    )
    return frame, audit


def compare(candidate: pd.DataFrame, reference: Path) -> dict:
    data = reference.read_bytes().replace(b"\r\n", b"\n")
    reference_hash = hashlib.sha256(data).hexdigest()
    if reference_hash != APPROVED_SHA256:
        raise ValueError(
            "Reference CSV does not match the approved LF-normalized SHA-256"
        )
    csv_text = candidate.to_csv(index=False, lineterminator="\n")
    generated_hash = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    expected = pd.read_csv(reference)
    candidate = pd.read_csv(io.StringIO(csv_text))
    if list(candidate) != list(expected) or candidate.shape != expected.shape:
        return {
            "matches": False,
            "reason": "shape/columns",
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(expected.shape),
            "reference_sha256": reference_hash,
        }
    different = {}
    for column in candidate:
        actual, target = candidate[column], expected[column]
        equal = (actual.eq(target) | (actual.isna() & target.isna())).to_numpy()
        if not equal.all():
            different[column] = int((~equal).sum())
    return {
        "matches": not different,
        "differing_cells_by_column": different,
        "reference_sha256": reference_hash,
        "generated_lf_sha256": generated_hash,
        "byte_identical_after_lf_normalization": generated_hash == reference_hash,
        "numeric_comparison_absolute_tolerance": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Written only after parity passes; cannot replace the approved CSV",
    )
    parser.add_argument(
        "--audit", type=Path, help="Optional audit JSON outside the submission"
    )
    args = parser.parse_args()
    if pd.__version__ != "2.3.3" or np.__version__ != "2.3.5":
        raise RuntimeError(
            "Use historical pandas==2.3.3 and numpy==2.3.5 for preparation/parity"
        )
    candidate, audit = prepare(args.raw.resolve())
    audit["parity"] = compare(candidate, args.reference.resolve())
    print(json.dumps(audit, indent=2))
    if args.audit:
        if args.audit.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
            raise ValueError("Audit artifacts must stay outside the final submission")
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if not audit["parity"]["matches"]:
        raise SystemExit(
            "Preparation does not reproduce the supplied CSV. No output written; specification needs resolution."
        )
    if args.output:
        if args.output.resolve() == args.reference.resolve() or args.output.exists():
            raise ValueError(
                "Refusing to replace the approved CSV or an existing output"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        candidate.to_csv(args.output, index=False, lineterminator="\n")
        print(
            "GENERATED_LF_SHA256", hashlib.sha256(args.output.read_bytes()).hexdigest()
        )


if __name__ == "__main__":
    main()

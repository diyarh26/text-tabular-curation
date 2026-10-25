"""Create the 15,000-row LinkedIn salary dataset with the original cleaning rules.

Usage: python src/preprocess.py --raw-dir /path/to/linkedin-job-postings
Source: arshkon/LinkedIn Job Postings on Kaggle, CC BY-SA 4.0.
Original preparation logic: project revision bfe0ba3b846cb8bd719d43a586dc4d25b36088cd.
Only the command-line input/output paths are adapted for this submission layout.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "linkedin"
OUT = ROOT / "data" / "dataset.csv"

# Roughly full-time federal minimum wage to $1M; outside are unit errors ($0, $130/year, $535M).
MIN_SALARY, MAX_SALARY = 15_000, 1_000_000
# E5 reads 512 tokens (~380 words), so longer descriptions are cut, keeping the CSV small.
MAX_DESCRIPTION_WORDS = 500
SAMPLE_ROWS = 15_000
SEED = 42
MIN_CATEGORY_ROWS = 20

AMOUNT = "[amount]"
PERIOD = r"(?:hr|hour|hourly|yr|year|yearly|annum|annually|wk|week|weekly|mo|month|monthly)"
# Words that name a pay period on their own, so "35 hours per week" (working time, not pay) stays.
PAY_PERIOD = r"(?:hourly|yearly|annually|annual|monthly|weekly|biweekly)"
NUMBER = r"\d[\d,]*(?:\.\d+)?"
RANGE = NUMBER + r"(?:\s*(?:-|–|to)\s*" + NUMBER + r")?"
MONEY_PATTERNS = [
    # $85,000  $45.50  $120K  $30KCommission  $1B  $27bn  (no word boundary: amounts are often glued to text)
    re.compile(r"(?:US\s?)?\$\s*" + NUMBER + r"(?:\s*(?:mm|bn|[kKmMbB]))?"),
    re.compile(r"\bUSD\s*" + NUMBER + r"(?:\s*[kKmM])?", re.I),                         # USD 90000
    re.compile(RANGE + r"\s*(?:USD|dollars)", re.I),                                   # 19.00 - 22.00 USD
    re.compile(RANGE + r"\s*(?:/|per|an|a)\s*" + PERIOD + r"\b", re.I),               # 22/hr  20 per hour
    re.compile(RANGE + r"\s*" + PAY_PERIOD, re.I),                                    # 28-30 Hourly(Shift)
    re.compile(r"\b\d+\.\d{2}\s*(?:-|–|to)\s*\d+\.\d{2}\b"),                          # 15.50 - 17.50
    re.compile(r"\b" + NUMBER + r"\s*[kK]\b"),                                         # 120k
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?"),                                     # 120,000
    re.compile(r"\b\d{5,7}(?:\.\d+)?\b"),                                              # 95000
]
EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b")


def mask_money(text):
    if not isinstance(text, str):
        return np.nan
    text = PHONE.sub("[phone]", EMAIL.sub("[email]", text))
    for pattern in MONEY_PATTERNS:
        text = pattern.sub(AMOUNT, text)
    text = " ".join(text.split())
    return text or np.nan


def truncate(text, words: int):
    if not isinstance(text, str):
        return text
    tokens = text.split()
    return " ".join(tokens[:words])


def merge_rare(values: pd.Series) -> pd.Series:
    counts = values.value_counts()
    return values.where(values.isin(counts[counts >= MIN_CATEGORY_ROWS].index) | values.isna(), "Other")


def main(raw_dir=RAW, output=OUT):
    raw_dir, output = Path(raw_dir).resolve(), Path(output).resolve()
    raw = pd.read_csv(raw_dir / "postings.csv", low_memory=False)
    print(f"Raw: {raw.shape[0]} postings x {raw.shape[1]} columns")
    n = len(raw)
    raw = raw[raw["normalized_salary"].notna() & raw["currency"].eq("USD")]
    print(f"Kept {len(raw)} of {n} postings with a USD salary")
    n = len(raw)
    raw = raw[raw["normalized_salary"].between(MIN_SALARY, MAX_SALARY)]
    print(f"Dropped {n - len(raw)} salaries outside ${MIN_SALARY:,}-${MAX_SALARY:,} per year")

    companies = pd.read_csv(raw_dir / "companies" / "companies.csv").set_index("company_id")
    counts = (pd.read_csv(raw_dir / "companies" / "employee_counts.csv").sort_values("time_recorded")
              .groupby("company_id").last())
    company_industry = pd.read_csv(raw_dir / "companies" / "company_industries.csv").groupby("company_id")["industry"].first()
    industry_names = pd.read_csv(raw_dir / "mappings" / "industries.csv").set_index("industry_id")["industry_name"]
    job_industry = (pd.read_csv(raw_dir / "jobs" / "job_industries.csv").groupby("job_id")["industry_id"].first()
                    .map(industry_names))
    n_skills = pd.read_csv(raw_dir / "jobs" / "job_skills.csv").groupby("job_id").size()
    n_benefits = pd.read_csv(raw_dir / "jobs" / "benefits.csv").groupby("job_id").size()

    df = pd.DataFrame(index=raw.index)
    df["title"] = raw["title"].map(mask_money)
    df["description"] = raw["description"].map(mask_money).map(lambda t: truncate(t, MAX_DESCRIPTION_WORDS))
    df["experience_level"] = raw["formatted_experience_level"]
    df["work_type"] = raw["formatted_work_type"]
    df["remote_allowed"] = raw["remote_allowed"].fillna(0).astype(int)
    df["state"] = raw["location"].str.extract(r",\s*([A-Z]{2})$")[0]
    df["application_type"] = raw["application_type"]
    company = raw["company_id"]
    df["company_size"] = company.map(companies["company_size"])
    df["company_employees"] = company.map(counts["employee_count"])
    df["company_followers"] = company.map(counts["follower_count"])
    df["company_industry"] = company.map(company_industry)
    df["job_industry"] = raw["job_id"].map(job_industry)
    df["n_skills"] = raw["job_id"].map(n_skills).fillna(0).astype(int)
    df["n_benefits"] = raw["job_id"].map(n_benefits).fillna(0).astype(int)
    df["log_salary"] = np.log(raw["normalized_salary"])

    n = len(df)
    df = df[df["title"].notna() & df["description"].notna()]
    # Reposts of one ad (often one company hiring for several sites) would sit in train and test at once.
    df = df.drop_duplicates(subset=["description"])
    print(f"Dropped {n - len(df)} postings without text or repeating another posting's description")

    df = df.sample(n=min(SAMPLE_ROWS, len(df)), random_state=SEED).reset_index(drop=True)
    for col in ["state", "company_industry", "job_industry"]:
        df[col] = merge_rare(df[col])

    for name, pattern in [("$<digit>", r"\$\s*\d"), ("<number> USD/dollars", r"\d\s*(?:USD|dollars)"),
                          ("<number> hourly/yearly", r"\d\s*" + PAY_PERIOD),
                          ("<x.xx> - <x.xx>", r"\d+\.\d{2}\s*(?:-|–|to)\s*\d+\.\d{2}")]:
        left = df["description"].str.contains(pattern, case=False, regex=True).sum()
        print(f"Descriptions still containing {name} after masking: {left}")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote a random sample of {len(df)} postings x {df.shape[1]} columns to {output} "
          f"({output.stat().st_size / 1e6:.1f} MB)")
    salary = np.exp(df["log_salary"])
    print(f"salary: median ${salary.median():,.0f}, 5%-95% ${salary.quantile(.05):,.0f}-${salary.quantile(.95):,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Source directory containing postings.csv, companies/, jobs/ and mappings/")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.raw_dir, args.output)

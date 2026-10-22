"""Clean the Glassdoor UK job reviews into a text-tabular CSV for predicting the reviewer's business outlook.

    python prep/glassdoor.py       # data/raw/glassdoor/glassdoor_reviews.csv -> data/processed/glassdoor_outlook.csv

Source: https://www.kaggle.com/datasets/davidgauthier/glassdoor-job-reviews (CC BY-SA 4.0), 838,566 reviews of 428
UK firms, 2008-2021. Target: outlook, the reviewer's answer to "will this company's business get better or worse in
the next 6 months?" (positive / neutral / negative); reviews without an opinion are dropped.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "glassdoor" / "glassdoor_reviews.csv"
OUT = ROOT / "data" / "processed" / "glassdoor_outlook.csv"

TEXT = ["headline", "pros", "cons"]
OPINION = {"v": "positive", "r": "mild", "x": "negative", "o": "no_opinion"}
OUTLOOK = {"v": "positive", "r": "neutral", "x": "negative"}

# MulTaBench uses at most 10K train + 5K test rows per fold, so a sample of exactly that size wastes nothing.
SAMPLE_ROWS = 15_000
SEED = 42
# Rare firms and cities (fewer rows than this in the sample) are merged into "Other".
MIN_CATEGORY_ROWS = 20
EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def clean_text(values: pd.Series) -> pd.Series:
    """Collapse whitespace, mask emails and phone numbers, and treat empty strings as missing."""
    def clean(text):
        if not isinstance(text, str):
            return np.nan
        text = " ".join(PHONE.sub("[phone]", EMAIL.sub("[email]", text)).split())
        return text or np.nan
    return values.map(clean)


def tenure(current: str) -> str:
    """'Former Employee, more than 3 years' -> 'more than 3 years'; no tenure given -> 'not stated'."""
    parts = str(current).split(",", 1)
    return parts[1].strip() if len(parts) == 2 else "not stated"


def merge_rare(values: pd.Series) -> pd.Series:
    counts = values.value_counts()
    return values.where(values.isin(counts[counts >= MIN_CATEGORY_ROWS].index) | values.isna(), "Other")


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    print(f"Raw: {raw.shape[0]} reviews x {raw.shape[1]} columns")
    n = len(raw)
    raw = raw.drop_duplicates()
    print(f"Dropped {n - len(raw)} exact duplicate reviews")
    n = len(raw)
    raw = raw[raw["outlook"].isin(OUTLOOK)]
    print(f"Dropped {n - len(raw)} reviews with no outlook opinion")

    df = pd.DataFrame(index=raw.index)
    for col in TEXT:
        df[col] = clean_text(raw[col])
    df["firm"] = raw["firm"]
    df["year"] = pd.to_datetime(raw["date_review"], errors="coerce").dt.year
    df["status"] = raw["current"].str.split(",").str[0].str.replace(" Employee", "").str.strip().str.lower()
    df["tenure"] = raw["current"].map(tenure)
    df["city"] = raw["location"].str.split(",").str[0].str.strip()
    df["overall_rating"] = raw["overall_rating"]
    df["recommend"] = raw["recommend"].map(OPINION)
    df["ceo_approval"] = raw["ceo_approv"].map(OPINION)
    df["outlook"] = raw["outlook"].map(OUTLOOK)

    n = len(df)
    df = df[df["pros"].notna() & df["cons"].notna()]
    # The same review text posted twice (e.g. under two locations) would sit in train and test at once.
    df = df.drop_duplicates(subset=TEXT)
    print(f"Dropped {n - len(df)} reviews with empty pros/cons or repeating another review's text")

    df = df.groupby("outlook", group_keys=False).sample(frac=SAMPLE_ROWS / len(df), random_state=SEED)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["firm"] = merge_rare(df["firm"])
    df["city"] = merge_rare(df["city"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote a stratified sample of {len(df)} reviews x {df.shape[1]} columns to {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print("outlook:", df["outlook"].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()

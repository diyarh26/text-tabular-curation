"""Clean the Los Angeles Airbnb listings (Inside Airbnb scrape of 2017-05-03) into a text-tabular CSV.

    python prep/la_airbnb.py            # data/raw/Detail_listings.csv -> data/processed/la_airbnb.csv

One row per listing. Free text is kept as text; everything else becomes a numerical or categorical column. The
candidate targets (price, review_scores_rating, host_is_superhost) all stay in the CSV; a config picks one and drops
whatever would leak it.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Detail_listings.csv"
OUT = ROOT / "data" / "processed" / "la_airbnb.csv"
SCRAPE_DATE = pd.Timestamp("2017-05-03")

# Prose written by the host. summary and space are left out: description is summary + space (97% / 99.7% of rows).
TEXT = ["name", "description", "neighborhood_overview", "transit", "house_rules", "host_about"]

CATEGORICAL = ["host_response_time", "neighbourhood_cleansed", "property_type", "room_type", "bed_type",
               "cancellation_policy"]
BOOLEAN = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "is_location_exact",
           "instant_bookable", "require_guest_profile_picture", "require_guest_phone_verification",
           "requires_license"]
MONEY = ["price", "weekly_price", "monthly_price", "security_deposit", "cleaning_fee", "extra_people"]
NUMERICAL = ["host_listings_count", "latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds",
             "guests_included", "minimum_nights", "maximum_nights", "availability_30", "availability_60",
             "availability_90", "availability_365", "number_of_reviews", "review_scores_rating",
             "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
             "review_scores_communication", "review_scores_location", "review_scores_value",
             "calculated_host_listings_count", "reviews_per_month"]
DATES = {"host_since": "host_days", "first_review": "days_since_first_review",
         "last_review": "days_since_last_review"}

# Rare categories (fewer rows than this) are merged into "Other", so no category exists in one fold only.
MIN_CATEGORY_ROWS = 20
EMAIL = re.compile(r"\S+@\S+\.\S+")
PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def clean_text(values: pd.Series) -> pd.Series:
    """Collapse whitespace, mask emails and phone numbers, and treat empty strings as missing."""
    def clean(text):
        if not isinstance(text, str):
            return np.nan
        text = PHONE.sub("[phone]", EMAIL.sub("[email]", text))
        text = " ".join(text.split())
        return text or np.nan
    return values.map(clean)


def money(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")


def main():
    raw = pd.read_csv(RAW, low_memory=False)
    print(f"Raw: {raw.shape[0]} listings x {raw.shape[1]} columns")
    df = pd.DataFrame(index=raw.index)

    for col in TEXT:
        df[col] = clean_text(raw[col])
    for col in CATEGORICAL:
        df[col] = raw[col]
    for col in BOOLEAN:
        df[col] = raw[col].map({"t": 1, "f": 0})
    for col in MONEY:
        df[col] = money(raw[col])
    for col in NUMERICAL:
        df[col] = pd.to_numeric(raw[col], errors="coerce")
    for col, new in DATES.items():
        df[new] = (SCRAPE_DATE - pd.to_datetime(raw[col], errors="coerce")).dt.days
    df["host_response_rate"] = pd.to_numeric(raw["host_response_rate"].str.rstrip("%"), errors="coerce")
    df["amenities_count"] = raw["amenities"].fillna("").str.strip("{}").map(
        lambda s: len([a for a in s.split(",") if a.strip('" ') and "translation missing" not in a]))
    df["host_verifications_count"] = raw["host_verifications"].fillna("").str.count("'") // 2
    df["log_price"] = np.log(df["price"])

    for col in CATEGORICAL:
        counts = df[col].value_counts()
        df[col] = df[col].where(df[col].isin(counts[counts >= MIN_CATEGORY_ROWS].index) | df[col].isna(), "Other")

    n = len(df)
    df = df[df["description"].notna() & df["price"].notna()]
    print(f"Dropped {n - len(df)} listings without a description or price")
    n = len(df)
    # A repeated description is mostly one host advertising near-identical units (89% share a host_id), whose copies
    # would sit in train and test at once; keep the first.
    df = df.drop_duplicates(subset=["description"])
    print(f"Dropped {n - len(df)} listings repeating another listing's description")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} listings x {df.shape[1]} columns to {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Build a text-tabular CSV of English Steam reviews for predicting whether the reviewer recommends the game.

    python prep/steam.py       # cached random blocks -> data/processed/steam_recommended.csv

Source: Steam Reviews Dataset 2021 (https://www.kaggle.com/datasets/najzeko/steam-reviews-2021, GPL 2), 21.7M
reviews of ~300 games, read from its Hugging Face parquet copy (GianLucaSpagnolo/Steam_Reviews_Dataset_2021). The
source is read through Hugging Face's row API to avoid the stalled remote Parquet download. Uniformly sampled
100-row blocks across the full source form a candidate pool; a simple random sample of its English, nonempty
reviews supplies at most 15,000 rows. There are no per-game quotas or minimum review length. This is two-stage
block sampling, not a simple random sample directly from all 21.7M source rows.

Votes, comments, the review's later updates and the reviewer's later playtime are dropped. Games owned and
reviews written are retained by project decision; they are not verified historical snapshots at review time.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "steam" / "row_groups_sample.parquet"
OUT = ROOT / "data" / "processed" / "steam_recommended.csv"
HF_DATASET = "GianLucaSpagnolo/Steam_Reviews_Dataset_2021"
CACHE = RAW.parent / "row_api_blocks"
BLOCK_ROWS = 100
DOWNLOAD_WORKERS = 2
POOL_MULTIPLIER = 1
# Space public viewer API requests; cached blocks need no request, and HTTP 429 gets a longer retry delay.
REQUEST_INTERVAL_SECONDS = 1.6
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
SAMPLE_ROWS = 15_000
SEED = 42
MIN_CATEGORY_ROWS = 20

COLUMNS = ["app_name", "language", "review", "timestamp_created", "recommended", "steam_purchase",
           "received_for_free", "written_during_early_access", "author.num_games_owned", "author.num_reviews",
           "author.playtime_at_review", "review_id"]


def download_random_blocks() -> pd.DataFrame:
    """Resumable random-block acquisition; each source row has equal marginal inclusion probability per batch."""
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata_path = CACHE / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        response = fetch_rows(0, 1)
        metadata = {"dataset": HF_DATASET, "source_rows": response["num_rows_total"], "block_rows": BLOCK_ROWS,
                    "seed": SEED, "sampling": "uniform random source blocks; random eligible-row subsample"}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    total = metadata["source_rows"]
    n_blocks = (total + BLOCK_ROWS - 1) // BLOCK_ROWS
    blocks = np.random.default_rng(SEED).permutation(n_blocks)

    def read_block(block):
        offset = int(block) * BLOCK_ROWS
        path = CACHE / f"{offset:09d}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        response = fetch_rows(offset, min(BLOCK_ROWS, total - offset))
        rows = []
        for item in response["rows"]:
            if "review" in item.get("truncated_cells", []):
                raise ValueError(f"Row API truncated review at source row {item['row_idx']}; refuse altered text")
            rows.append({col: item["row"][col] for col in COLUMNS})
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return rows

    rows, downloaded = [], 0
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        while downloaded < min(n_blocks, 2_000):
            batch_size = 400 if downloaded == 0 else 100
            batch = blocks[downloaded:min(downloaded + batch_size, n_blocks, 2_000)]
            for i, block_rows in enumerate(pool.map(read_block, batch), start=1):
                rows.extend(block_rows)
                if i % 100 == 0:
                    print(f"  fetched/cached {downloaded + i} source blocks", flush=True)
            downloaded += len(batch)
            frame = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates(subset=["review_id"])
            eligible = (frame["language"].eq("english") & frame["review"].map(
                lambda text: isinstance(text, str) and bool(text.strip())) & frame["recommended"].notna())
            print(f"Candidate pool: {len(frame):,} rows, {eligible.sum():,} nonempty English reviews, "
                  f"{frame.loc[eligible, 'app_name'].nunique()} games", flush=True)
            if eligible.sum() >= POOL_MULTIPLIER * SAMPLE_ROWS:
                break
    if eligible.sum() < SAMPLE_ROWS:
        raise ValueError(f"Only {eligible.sum()} eligible reviews; need {SAMPLE_ROWS}. Cached blocks are retained.")
    metadata.update({"downloaded_blocks": downloaded, "candidate_rows": len(frame),
                     "eligible_rows": int(eligible.sum()), "eligible_games": int(frame.loc[eligible, "app_name"].nunique())})
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def fetch_rows(offset, length):
    global _LAST_REQUEST
    query = urlencode({"dataset": HF_DATASET, "config": "default", "split": "train", "offset": offset,
                       "length": length})
    for attempt in range(16):
        try:
            with _REQUEST_LOCK:
                time.sleep(max(0, REQUEST_INTERVAL_SECONDS - (time.monotonic() - _LAST_REQUEST)))
                _LAST_REQUEST = time.monotonic()
            with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            if attempt == 15 or error.code not in (429, 500, 502, 503, 504):
                raise
            delay = 30 if error.code == 429 else min(2 ** attempt, 8)
            print(f"  row API HTTP {error.code}; retrying after {delay}s (cached progress retained)", flush=True)
            time.sleep(delay)
        except Exception:
            if attempt == 15:
                raise
            time.sleep(min(2 ** attempt, 8))


def merge_rare(values: pd.Series) -> pd.Series:
    counts = values.value_counts()
    return values.where(values.isin(counts[counts >= MIN_CATEGORY_ROWS].index) | values.isna(), "Other")


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        download_random_blocks().to_parquet(RAW)
    raw = pd.read_parquet(RAW)
    source_candidate_rows = len(raw)
    print(f"Raw sample: {len(raw)} reviews of {raw['app_name'].nunique()} games")
    raw = raw.drop_duplicates(subset=["review_id"])
    n = len(raw)
    raw = raw[raw["language"].eq("english") & raw["recommended"].notna()]
    print(f"Kept {len(raw)} of {n} reviews in English")

    df = pd.DataFrame(index=raw.index)
    df["review"] = raw["review"].map(lambda t: " ".join(t.split()) if isinstance(t, str) else np.nan).replace("", np.nan)
    df["game"] = raw["app_name"]
    df["year"] = pd.to_datetime(raw["timestamp_created"], unit="s").dt.year
    df["playtime_at_review_hours"] = (raw["author.playtime_at_review"] / 60).round(1)
    df["games_owned"] = raw["author.num_games_owned"]
    df["reviews_written"] = raw["author.num_reviews"]
    for col in ["steam_purchase", "received_for_free", "written_during_early_access"]:
        df[col] = raw[col].astype(int)
    df["recommended"] = raw["recommended"].map({True: "yes", False: "no"})

    n = len(df)
    df = df[df["review"].notna()]
    print(f"Dropped {n - len(df)} empty reviews")

    df = df.sample(n=min(SAMPLE_ROWS, len(df)), random_state=SEED)
    selected_ids = raw.loc[df.index, "review_id"].astype(int).tolist()
    df = df.reset_index(drop=True)
    game_counts = df["game"].value_counts()
    print(f"Random sample covers {len(game_counts)} games; top 5 counts: {game_counts.head().to_dict()}")
    df["game"] = merge_rare(df["game"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    (RAW.parent / "selected_review_ids.json").write_text(json.dumps(selected_ids), encoding="utf-8")
    summary = {"rows": len(df), "seed": SEED, "source": HF_DATASET, "language": "english",
               "minimum_words": None, "per_game_quotas": False, "target_stratified_sampling": False,
               "source_candidate_rows": source_candidate_rows, "english_rows_before_empty_filter": len(raw),
               "eligible_rows": int(raw["review"].map(
                   lambda text: isinstance(text, str) and bool(text.strip())).sum()),
               "games_before_rare_merging": len(game_counts), "game_counts_before_rare_merging": game_counts.to_dict(),
               "game_categories_after_rare_merging": int(df["game"].nunique()),
               "rare_game_threshold": MIN_CATEGORY_ROWS, "target_counts": df["recommended"].value_counts().to_dict(),
               "word_count_quantiles": df["review"].str.split().str.len().quantile([0, .25, .5, .75, 1]).to_dict(),
               "duplicate_review_text_rows": int(df["review"].duplicated().sum()),
               "csv_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
               "csv_sha256_lf": hashlib.sha256(OUT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote a random sample of {len(df)} reviews x {df.shape[1]} columns to {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print("recommended:", df["recommended"].value_counts(normalize=True).round(3).to_dict(),
          f"| games: {df['game'].nunique()} | review words median {df['review'].str.split().str.len().median():.0f}")


if __name__ == "__main__":
    main()

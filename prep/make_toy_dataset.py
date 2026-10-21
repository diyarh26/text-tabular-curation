"""A small synthetic text-tabular dataset for smoke-testing the pipeline: the target depends on a numeric feature, a
category and a word in the review, so every condition has some signal. Not a deliverable.

    python prep/make_toy_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "toy" / "toy.csv"
POSITIVE = ["excellent", "loved it", "great value", "would buy again", "superb quality"]
NEGATIVE = ["terrible", "broke quickly", "waste of money", "very disappointed", "poor quality"]
FILLER = ["arrived on time", "the box was blue", "used it for a week", "bought it as a gift", "standard packaging"]


def main(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    price = rng.lognormal(3, 0.6, n).round(2)
    category = rng.choice(["books", "toys", "tools", "kitchen"], n)
    sentiment = rng.normal(0, 1, n)
    reviews = [f"{rng.choice(POSITIVE if s > 0 else NEGATIVE)}, {rng.choice(FILLER)}." for s in sentiment]
    logit = 1.5 * sentiment - 0.03 * (price - 20) + (category == "tools") * 1.0
    target = (logit + rng.normal(0, 0.7, n) > 0).astype(int)
    df = pd.DataFrame({"id": range(n), "price": price, "category": category, "review": reviews,
                       "recommended": target})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()

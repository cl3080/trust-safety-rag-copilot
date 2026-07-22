"""
01_data_prep.py
Loads the Deceptive Opinion Spam Corpus (Ott et al., 2011/2013) and produces
a clean train/test split used by every downstream script.

Dataset: 1,600 hotel reviews for the 20 most-reviewed Chicago hotels.
  - 800 truthful reviews scraped from TripAdvisor / other travel sites
  - 800 deceptive ("fake") reviews written by paid Mechanical Turk workers
  - Balanced on sentiment (800 positive / 800 negative)

Citation:
  M. Ott, Y. Choi, C. Cardie, J.T. Hancock. "Finding Deceptive Opinion Spam
  by Any Stretch of the Imagination." ACL 2011.
  M. Ott, C. Cardie, J.T. Hancock. "Negative Deceptive Opinion Spam." NAACL 2013.

Public mirror used here: data/deceptive-opinion.csv (source: Kaggle, rtatman/
deceptive-opinion-spam-corpus). No header row; columns are positional.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

COLS = ["label", "hotel", "polarity", "source", "text"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "deceptive-opinion.csv", header=None, names=COLS)
    # label: 1 = deceptive (MTurk-written), 0 = truthful (real traveler review)
    df["is_deceptive"] = df["label"].astype(int)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df["word_count"] = df["text"].str.split().apply(len)
    return df[["text", "is_deceptive", "polarity", "source", "hotel", "word_count"]]


def main():
    df = load_raw()
    print(f"Loaded {len(df)} reviews")
    print(df["is_deceptive"].value_counts(normalize=True).rename("class_balance"))
    print(df.groupby("is_deceptive")["word_count"].describe()[["mean", "std", "min", "max"]])

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["is_deceptive"], random_state=42
    )
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    print(f"\nSaved {len(train)} train / {len(test)} test rows to outputs/")


if __name__ == "__main__":
    main()

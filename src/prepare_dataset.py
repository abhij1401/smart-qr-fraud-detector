"""
prepare_dataset.py
------------------
Combines phishing URLs (bad) and legitimate URLs (good) into one
labelled dataset that the model will learn from.

Label meaning:
  1 = phishing / fraudulent URL
  0 = legitimate / safe URL
"""

import pandas as pd
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PHISH_CSV     = os.path.join(RAW_DIR, "phishtank.csv")
LEGIT_CSV     = os.path.join(RAW_DIR, "majestic_million.csv")
OUT_CSV       = os.path.join(PROCESSED_DIR, "dataset.csv")


def load_phishing_urls(path: str) -> pd.DataFrame:
    """
    Read the PhishTank CSV and return a DataFrame with two columns:
      - url   : the phishing URL
      - label : 1  (phishing)

    PhishTank exports vary; we try common column names and fall back
    to the first column if none match.
    """
    df = pd.read_csv(path, low_memory=False)

    # Common column names used by PhishTank exports
    url_col_candidates = ["url", "phish_url", "URL"]
    url_col = next((c for c in url_col_candidates if c in df.columns), df.columns[0])

    print(f"[phishing] using column '{url_col}' from {len(df):,} rows")

    return pd.DataFrame({"url": df[url_col].astype(str), "label": 1})


def load_legitimate_urls(path: str) -> pd.DataFrame:
    """
    Read the Majestic Million CSV and return a DataFrame with two columns:
      - url   : full URL built by prepending 'https://' to each domain
      - label : 0  (legitimate)

    Majestic Million always has a 'Domain' column.
    """
    df = pd.read_csv(path, low_memory=False)

    # Majestic uses 'Domain'; fall back to first column just in case
    domain_col = "Domain" if "Domain" in df.columns else df.columns[0]

    print(f"[legitimate] using column '{domain_col}' from {len(df):,} rows")

    # Turn bare domain into a proper URL, e.g. google.com → https://google.com
    urls = "https://" + df[domain_col].astype(str)

    return pd.DataFrame({"url": urls, "label": 0})


def build_dataset(phish_path: str, legit_path: str, out_path: str) -> None:
    """
    Merge phishing and legitimate DataFrames, shuffle rows, and save.
    """
    phish_df = load_phishing_urls(phish_path)
    legit_df  = load_legitimate_urls(legit_path)

    # Stack the two DataFrames on top of each other
    combined = pd.concat([phish_df, legit_df], ignore_index=True)

    # Shuffle so the model doesn't just memorise order
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop rows where the URL is empty or NaN
    combined = combined.dropna(subset=["url"])
    combined = combined[combined["url"].str.strip() != ""]

    # Make sure the output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    combined.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved → {out_path}")
    print(f"   Total rows : {len(combined):,}")
    print(f"   Phishing   : {(combined['label'] == 1).sum():,}")
    print(f"   Legitimate : {(combined['label'] == 0).sum():,}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_dataset(PHISH_CSV, LEGIT_CSV, OUT_CSV)
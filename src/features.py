"""
features.py
-----------
Turns raw URLs into numeric features that a machine-learning model
can understand.

We keep the logic in a single function (extract_features_from_url) so
the exact same code is used during training AND during live prediction —
this prevents "training/serving skew".
"""

import pandas as pd
import re
from urllib.parse import urlparse
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
DATASET_CSV   = os.path.join(PROCESSED_DIR, "dataset.csv")
FEATURES_CSV  = os.path.join(PROCESSED_DIR, "features.csv")


# ── Core feature extractor ─────────────────────────────────────────────────────

def extract_features_from_url(url: str) -> dict:
    """
    Given a single URL string, return a dictionary of numeric features.

    These features were chosen because fraudulent URLs often differ from
    legitimate ones in predictable ways (longer paths, many digits, keywords
    like 'login' or 'verify', missing HTTPS, many subdomains, etc.).
    """
    # Parse the URL into its components (scheme, netloc, path, …)
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    full_url = url.lower()                     # lowercase for keyword checks
    domain   = parsed.netloc.lower()           # e.g. "sub.example.com"
    path     = parsed.path.lower()             # e.g. "/login/verify"

    # Remove port from domain if present (e.g. "example.com:8080" → "example.com")
    domain_no_port = domain.split(":")[0]

    # Count subdomains: split by dot, subtract TLD + second-level domain
    # e.g. "a.b.example.com" → ["a","b","example","com"] → 2 subdomains
    domain_parts = domain_no_port.split(".")
    num_subdomains = max(0, len(domain_parts) - 2)

    features = {
        # ── Length-based features ──────────────────────────────────────────
        # Phishing URLs tend to be long to hide the real destination
        "url_length":    len(url),
        "domain_length": len(domain_no_port),
        "path_length":   len(path),

        # ── Character-count features ───────────────────────────────────────
        # Many dots can indicate subdomain abuse; hyphens are common in spoofs
        "num_dots":    full_url.count("."),
        "num_hyphens": full_url.count("-"),

        # Lots of digits in a URL is suspicious (e.g. IP-based or obfuscated)
        "num_digits": sum(c.isdigit() for c in full_url),

        # ── Protocol feature ───────────────────────────────────────────────
        # Legitimate sites almost always use HTTPS; many phishing sites don't
        "https": 1 if parsed.scheme == "https" else 0,

        # ── Structural feature ─────────────────────────────────────────────
        "num_subdomains": num_subdomains,

        # ── Keyword features ───────────────────────────────────────────────
        # These words appear often in phishing URLs trying to look trustworthy
        "contains_login":    1 if "login"    in full_url else 0,
        "contains_verify":   1 if "verify"   in full_url else 0,
        "contains_secure":   1 if "secure"   in full_url else 0,
        "contains_account":  1 if "account"  in full_url else 0,
        "contains_password": 1 if "password" in full_url else 0,
    }

    return features


# ── Batch processing ───────────────────────────────────────────────────────────

def build_features(dataset_path: str, out_path: str) -> None:
    """
    Read the labelled dataset, extract features for every URL, and save.
    """
    print(f"Loading dataset from {dataset_path} …")
    df = pd.read_csv(dataset_path)

    print(f"Extracting features for {len(df):,} URLs …")

    # Apply our feature extractor to every URL → list of dicts → DataFrame
    feature_rows = df["url"].apply(extract_features_from_url)
    features_df  = pd.DataFrame(feature_rows.tolist())

    # Attach the label column so train.py can split X / y easily
    features_df["label"] = df["label"].values

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features_df.to_csv(out_path, index=False)

    print(f"\n✅ Features saved → {out_path}")
    print(f"   Shape : {features_df.shape}  (rows × columns)")
    print(f"   Columns: {list(features_df.columns)}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_features(DATASET_CSV, FEATURES_CSV)
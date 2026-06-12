"""
predict.py
----------
Loads the saved model and predicts whether a URL is phishing or
legitimate.  Uses the identical feature extraction logic from features.py.

Usage (interactive):
  python src/predict.py
  → prompts you to type a website URL, shows the result, repeats until
    you type 'quit' or press Ctrl-C.

Usage (as a module):
  from src.predict import predict_url
  result = predict_url("https://example.com")
"""

import joblib
import pandas as pd
import sys
import os

# Re-use the exact same feature extractor as training — critical!
sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features_from_url

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

# ── Load model once at import time ─────────────────────────────────────────────
_model = None

def _get_model():
    """Lazy-load the model so it's only read from disk once."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}.\n"
                "Run `python src/train.py` first to train and save the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


# ── Public API ─────────────────────────────────────────────────────────────────

def predict_url(url: str) -> dict:
    """
    Predict whether a URL is phishing or legitimate.

    Returns a dict with:
      - url        : the input URL
      - prediction : "phishing" or "legitimate"
      - confidence : probability of the predicted class (0.0 – 1.0)
      - features   : the numeric features that were used
    """
    model = _get_model()

    # Extract features (same function used during training)
    feature_dict = extract_features_from_url(url)
    X = pd.DataFrame([feature_dict])

    # Predict
    label     = model.predict(X)[0]           # 0 or 1
    proba     = model.predict_proba(X)[0]     # [prob_legit, prob_phish]
    confidence = float(proba[label])

    return {
        "url":        url,
        "prediction": "phishing" if label == 1 else "legitimate",
        "confidence": round(confidence, 4),
        "features":   feature_dict,
    }


def predict_bulk(urls: list) -> pd.DataFrame:
    """
    Predict a list of URLs and return a DataFrame with one row per URL.
    Useful for batch QR code scanning.
    """
    model   = _get_model()
    records = [extract_features_from_url(u) for u in urls]
    X       = pd.DataFrame(records)

    labels  = model.predict(X)
    probas  = model.predict_proba(X)

    results = pd.DataFrame({
        "url":        urls,
        "prediction": ["phishing" if l == 1 else "legitimate" for l in labels],
        "confidence": [round(float(p[l]), 4) for l, p in zip(labels, probas)],
    })
    return results


# ── Interactive CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   🔍 Smart QR Code Fraud Detector")
    print("   Type a website URL to check, or 'quit' to exit.")
    print("=" * 55)

    while True:
        # Ask the user to enter a URL
        try:
            raw = input("\n  Enter URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Allow the user to exit cleanly
        if raw.lower() in ("quit", "exit", "q", ""):
            print("Goodbye!")
            break

        # Auto-add https:// if the user forgot the scheme
        url = raw if "://" in raw else "https://" + raw

        # Run prediction
        try:
            result = predict_url(url)
        except FileNotFoundError as e:
            print(f"\n  ⚠️  {e}")
            continue
        except Exception as e:
            print(f"\n  ⚠️  Error during prediction: {e}")
            continue

        # Display result
        is_phish = result["prediction"] == "phishing"
        emoji    = "🚨" if is_phish else "✅"
        verdict  = result["prediction"].upper()
        conf     = result["confidence"]

        print(f"\n  {emoji}  {verdict}   (confidence: {conf:.1%})")
        print(f"      URL checked : {result['url']}")

        # Show the feature breakdown so users can understand why
        print("\n  Feature breakdown:")
        for feature, value in result["features"].items():
            # Add a small flag next to suspicious values
            flag = ""
            if feature in ("contains_login", "contains_verify",
                           "contains_secure", "contains_account",
                           "contains_password") and value == 1:
                flag = "  ⚠️"
            if feature == "https" and value == 0:
                flag = "  ⚠️  (no HTTPS)"
            if feature == "num_subdomains" and value >= 3:
                flag = "  ⚠️  (many subdomains)"
            print(f"    {feature:<22} = {value}{flag}")

        print()  # blank line before next prompt
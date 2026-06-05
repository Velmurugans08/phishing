"""
predictor.py
------------
This module provides real-time prediction functionality.
Given a raw email text, it:
    1. Preprocesses the text (clean, lowercase, remove stopwords)
    2. Extracts features (TF-IDF + URL count + suspicious keywords)
    3. Runs the trained model to classify the email
    4. Returns a human-readable result: "Phishing Email" or "Safe Email"

This is the module called by the web UI (app.py).

Author: Phishing Detection Project
"""

import os
import joblib

from preprocessor import preprocess_text
from feature_extractor import FeatureExtractor, count_urls, count_suspicious_keywords, SUSPICIOUS_KEYWORDS


# ---------------------------------------------------------------
# Default paths for saved models
# ---------------------------------------------------------------
MODELS_DIR      = "models"
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
EXTRACTOR_PATH  = os.path.join(MODELS_DIR, "feature_extractor.pkl")


class PhishingPredictor:
    """
    A ready-to-use predictor class that loads the saved model
    and feature extractor, then classifies new emails on demand.

    Usage:
        predictor = PhishingPredictor()
        result = predictor.predict("Your account has been suspended...")
        print(result["label"])  # "Phishing Email" or "Safe Email"
    """

    def __init__(self, model_path=BEST_MODEL_PATH, extractor_path=EXTRACTOR_PATH):
        """
        Loads the saved model and feature extractor from disk.
        If models don't exist yet, raises a helpful error message.

        Args:
            model_path (str): Path to the saved best model (.pkl file).
            extractor_path (str): Path to the saved feature extractor (.pkl file).
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Please run 'python train_and_evaluate.py' first to train the model."
            )

        if not os.path.exists(extractor_path):
            raise FileNotFoundError(
                f"Feature extractor not found at '{extractor_path}'.\n"
                "Please run 'python train_and_evaluate.py' first to train the model."
            )

        # Load the trained model
        self.model = joblib.load(model_path)
        print(f"[INFO] Model loaded from: {model_path}")

        # Load the fitted feature extractor (TF-IDF vectorizer inside)
        self.extractor = FeatureExtractor.load(extractor_path)
        print("[INFO] PhishingPredictor is ready.")

    def predict(self, email_text):
        """
        Classifies a single email as Phishing or Safe.

        Steps:
            1. Preprocess the raw email text
            2. Extract combined features (TF-IDF + URL + keywords)
            3. Run model prediction
            4. Get confidence probability (if available)
            5. Return structured result dictionary

        Args:
            email_text (str): Raw email content pasted by the user.

        Returns:
            dict: {
                "label"       : "Phishing Email" or "Safe Email",
                "prediction"  : 1 (phishing) or 0 (safe),
                "confidence"  : float (0.0 to 100.0),
                "url_count"   : int,
                "keyword_count" : int,
                "found_keywords": list of suspicious words found
            }
        """
        if not email_text or not email_text.strip():
            return {
                "label"          : "No Input",
                "prediction"     : -1,
                "confidence"     : 0.0,
                "url_count"      : 0,
                "keyword_count"  : 0,
                "found_keywords" : []
            }

        # Step 1: Preprocess text
        cleaned_text = preprocess_text(email_text)

        # Step 2: Extract combined features
        features = self.extractor.transform_single(email_text, cleaned_text)

        # Step 3: Model prediction
        prediction = self.model.predict(features)[0]

        # Step 4: Confidence score (probability)
        confidence = 50.0  # Default if model doesn't support probabilities
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            # proba[1] = probability of being phishing
            confidence = round(proba[prediction] * 100, 1)

        # Step 5: Extra info — which suspicious keywords were found
        found_keywords = self._find_suspicious_keywords(email_text)
        url_count      = count_urls(email_text)
        keyword_count  = count_suspicious_keywords(email_text)

        # Map prediction number to human-readable label
        label = "🚨 Phishing Email" if prediction == 1 else "✅ Safe Email"

        return {
            "label"          : label,
            "prediction"     : int(prediction),
            "confidence"     : confidence,
            "url_count"      : url_count,
            "keyword_count"  : keyword_count,
            "found_keywords" : found_keywords
        }

    def _find_suspicious_keywords(self, text):
        """
        Returns a list of suspicious keywords actually found in the email.
        Used to show the user WHY an email was flagged as phishing.

        Args:
            text (str): Raw email text.

        Returns:
            list: Suspicious keywords found in the text.
        """
        import re
        text_lower = text.lower()
        found = []
        for keyword in SUSPICIOUS_KEYWORDS:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text_lower):
                found.append(keyword)
        return found


def predict_email(email_text):
    """
    Convenience function for quick one-off predictions.
    Creates a predictor, classifies the email, and returns the result.

    Args:
        email_text (str): Raw email text to classify.

    Returns:
        dict: Prediction result dictionary.
    """
    predictor = PhishingPredictor()
    return predictor.predict(email_text)


# ---------------------------------------------------------------
# Quick test: run this file directly with sample emails
# ---------------------------------------------------------------
if __name__ == "__main__":
    # Test with a clear phishing email
    phishing_sample = """
    URGENT: Your bank account has been suspended!
    Click http://fake-bank-login.xyz to verify your password and account details now.
    Failure to verify within 24 hours will result in permanent suspension.
    """

    # Test with a safe email
    safe_sample = """
    Hi Sarah, just following up on our meeting from yesterday.
    Could you please send over the project timeline? Thanks a lot!
    """

    predictor = PhishingPredictor()

    print("\n--- Test 1: Phishing Email Sample ---")
    result1 = predictor.predict(phishing_sample)
    print(f"  Result     : {result1['label']}")
    print(f"  Confidence : {result1['confidence']}%")
    print(f"  URLs found : {result1['url_count']}")
    print(f"  Suspicious : {result1['found_keywords']}")

    print("\n--- Test 2: Safe Email Sample ---")
    result2 = predictor.predict(safe_sample)
    print(f"  Result     : {result2['label']}")
    print(f"  Confidence : {result2['confidence']}%")
    print(f"  URLs found : {result2['url_count']}")
    print(f"  Suspicious : {result2['found_keywords']}")

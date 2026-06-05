"""
feature_extractor.py
--------------------
This module handles feature extraction from preprocessed email text.
It extracts THREE types of features:

    1. TF-IDF Features  : Numerical representation of word importance
    2. URL Count        : How many URLs are in the email
    3. Suspicious Words : Count of words commonly used in phishing emails

These features are combined into a single feature matrix used
for model training and prediction.

Author: Phishing Detection Project
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import os


# ---------------------------------------------------------------
# List of suspicious keywords commonly found in phishing emails
# ---------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "account", "password", "bank", "click",
    "urgent", "immediately", "suspended", "security", "update",
    "confirm", "credit", "card", "social", "security", "number",
    "free", "winner", "prize", "claim", "limited", "offer",
    "expires", "alert", "warning", "unauthorized", "compromised",
    "reset", "access", "unlock", "validate", "restricted",
    "refund", "transfer", "invoice", "payment", "wire"
]


def count_urls(text):
    """
    Counts the number of URLs present in the email text.
    Phishing emails often contain suspicious or fake URLs.

    Args:
        text (str): Raw or preprocessed email text.

    Returns:
        int: Number of URLs found in the text.
    """
    # Match common URL patterns: http://, https://, or www.
    url_pattern = re.compile(
        r"(http[s]?://|www\.)[^\s]+",
        re.IGNORECASE
    )
    urls = url_pattern.findall(text)
    return len(urls)


def count_suspicious_keywords(text):
    """
    Counts how many suspicious phishing-related keywords appear
    in the email text.

    Args:
        text (str): Preprocessed email text (lowercase).

    Returns:
        int: Number of suspicious keyword matches.
    """
    text_lower = text.lower()
    count = 0
    for keyword in SUSPICIOUS_KEYWORDS:
        # Use word boundary to avoid partial matches
        pattern = r"\b" + re.escape(keyword) + r"\b"
        matches = re.findall(pattern, text_lower)
        count += len(matches)
    return count


def extract_handcrafted_features(df, text_column="text"):
    """
    Extracts URL count and suspicious keyword count from each email.
    These are 'handcrafted' features that we design based on domain knowledge
    about what makes an email look like phishing.

    Args:
        df (pd.DataFrame): DataFrame containing email texts.
        text_column (str): Column name containing raw email text.

    Returns:
        np.ndarray: A 2D array where each row has [url_count, suspicious_keyword_count].
    """
    print("[INFO] Extracting handcrafted features (URL count + suspicious keywords)...")

    url_counts = df[text_column].apply(count_urls).values
    keyword_counts = df[text_column].apply(count_suspicious_keywords).values

    # Stack into a 2D array: shape = (n_samples, 2)
    handcrafted = np.column_stack([url_counts, keyword_counts])

    print(f"[INFO] Handcrafted features extracted: {handcrafted.shape}")
    return handcrafted


class FeatureExtractor:
    """
    A class that manages TF-IDF vectorization and combines it
    with handcrafted features (URL count and suspicious keywords).

    Usage:
        extractor = FeatureExtractor()
        X_train = extractor.fit_transform(train_df)
        X_test  = extractor.transform(test_df)
    """

    def __init__(self, max_features=3000):
        """
        Initializes the TF-IDF vectorizer with specified parameters.

        Args:
            max_features (int): Maximum number of words to use in TF-IDF.
                                Higher = more detail, but slower training.
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use both single words and two-word phrases
            min_df=1             # Include words that appear at least once
        )
        self.is_fitted = False

    def fit_transform(self, df, text_column="cleaned_text", raw_text_column="text"):
        """
        Fits the TF-IDF vectorizer on the training data and transforms it.
        Also extracts handcrafted features and combines everything.

        Call this ONLY on the training set.

        Args:
            df (pd.DataFrame): Training DataFrame.
            text_column (str): Column with preprocessed (cleaned) text.
            raw_text_column (str): Column with original text (for URL/keyword detection).

        Returns:
            scipy.sparse matrix: Combined feature matrix for training.
        """
        print("[INFO] Fitting TF-IDF vectorizer and transforming training data...")

        # TF-IDF features from cleaned text
        tfidf_features = self.vectorizer.fit_transform(df[text_column])

        # Handcrafted features from original text
        handcrafted = extract_handcrafted_features(df, raw_text_column)
        handcrafted_sparse = csr_matrix(handcrafted)

        # Combine TF-IDF and handcrafted features side by side
        combined = hstack([tfidf_features, handcrafted_sparse])

        self.is_fitted = True
        print(f"[INFO] Training features shape: {combined.shape}")
        return combined

    def transform(self, df, text_column="cleaned_text", raw_text_column="text"):
        """
        Transforms new data using the already-fitted TF-IDF vectorizer.
        
        Call this on test data AFTER calling fit_transform() on training data.

        Args:
            df (pd.DataFrame): Test or prediction DataFrame.
            text_column (str): Column with preprocessed (cleaned) text.
            raw_text_column (str): Column with original text.

        Returns:
            scipy.sparse matrix: Combined feature matrix for testing/prediction.
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before calling transform().")

        print("[INFO] Transforming data using fitted TF-IDF vectorizer...")

        # TF-IDF features (using already fitted vectorizer)
        tfidf_features = self.vectorizer.transform(df[text_column])

        # Handcrafted features
        handcrafted = extract_handcrafted_features(df, raw_text_column)
        handcrafted_sparse = csr_matrix(handcrafted)

        # Combine features
        combined = hstack([tfidf_features, handcrafted_sparse])

        print(f"[INFO] Test features shape: {combined.shape}")
        return combined

    def transform_single(self, raw_text, cleaned_text):
        """
        Transforms a single email text for real-time prediction.

        Args:
            raw_text (str): Original email text.
            cleaned_text (str): Preprocessed email text.

        Returns:
            scipy.sparse matrix: Feature matrix for a single email.
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before calling transform_single().")

        # TF-IDF for single email
        tfidf_feature = self.vectorizer.transform([cleaned_text])

        # Handcrafted features for single email
        url_count = count_urls(raw_text)
        keyword_count = count_suspicious_keywords(raw_text)
        handcrafted = csr_matrix([[url_count, keyword_count]])

        # Combine
        combined = hstack([tfidf_feature, handcrafted])
        return combined

    def save(self, filepath="models/feature_extractor.pkl"):
        """
        Saves the fitted FeatureExtractor to disk.

        Args:
            filepath (str): Path where the extractor will be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[INFO] FeatureExtractor saved to: {filepath}")

    @staticmethod
    def load(filepath="models/feature_extractor.pkl"):
        """
        Loads a previously saved FeatureExtractor from disk.

        Args:
            filepath (str): Path to the saved extractor file.

        Returns:
            FeatureExtractor: The loaded extractor object.
        """
        extractor = joblib.load(filepath)
        print(f"[INFO] FeatureExtractor loaded from: {filepath}")
        return extractor


# ---------------------------------------------------------------
# Quick test: run this file directly to see feature extraction
# ---------------------------------------------------------------
if __name__ == "__main__":
    sample_text = "Click here to verify your account at http://phishing-site.com urgent password reset"
    print(f"URL count       : {count_urls(sample_text)}")
    print(f"Keyword count   : {count_suspicious_keywords(sample_text)}")

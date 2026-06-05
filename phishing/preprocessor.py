"""
preprocessor.py
---------------
This module handles all text preprocessing steps before feature extraction.
It cleans the raw email text by:
  - Converting to lowercase
  - Removing special characters and punctuation
  - Removing stop words (common English words like 'the', 'is', 'at')
  - Tokenizing the text (splitting into individual words)

Author: Phishing Detection Project
"""

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# ---------------------------------------------------------------
# Download required NLTK data files (only runs once)
# ---------------------------------------------------------------
def download_nltk_resources():
    """
    Downloads the necessary NLTK datasets if they are not already present.
    This function should be called before using any NLTK features.
    """
    resources = ["stopwords", "punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[WARNING] Could not download NLTK resource '{resource}': {e}")

# Run the download when this module is imported
download_nltk_resources()


# ---------------------------------------------------------------
# Core preprocessing functions
# ---------------------------------------------------------------

def to_lowercase(text):
    """
    Converts all characters in the text to lowercase.
    
    Example:
        "Hello WORLD!" -> "hello world!"

    Args:
        text (str): Raw email text.

    Returns:
        str: Lowercased text.
    """
    return text.lower()


def remove_special_characters(text):
    """
    Removes punctuation and special characters, keeping only
    letters, digits, and spaces.

    Example:
        "hello! visit: http://xyz.com" -> "hello visit httpxyzcom"

    Args:
        text (str): Text to clean.

    Returns:
        str: Cleaned text with special characters removed.
    """
    # Keep only alphanumeric characters and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return cleaned


def remove_stopwords(text):
    """
    Removes common English stop words from the text.
    Stop words are words like 'the', 'is', 'at', 'which' that
    carry little meaningful information for classification.

    Example:
        "the account is at risk" -> "account risk"

    Args:
        text (str): Text after lowercasing and cleaning.

    Returns:
        str: Text with stop words removed.
    """
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)


def tokenize(text):
    """
    Splits the text into individual tokens (words).
    
    Example:
        "click here now" -> ["click", "here", "now"]

    Args:
        text (str): Cleaned text string.

    Returns:
        list: A list of word tokens.
    """
    tokens = word_tokenize(text)
    return tokens


def preprocess_text(text):
    """
    Runs the complete preprocessing pipeline on a single email text.
    
    Steps:
        1. Convert to lowercase
        2. Remove special characters
        3. Remove stop words
        4. Return cleaned text as a single string

    Args:
        text (str): Raw email text.

    Returns:
        str: Fully preprocessed and cleaned text.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = to_lowercase(text)

    # Step 2: Remove special characters
    text = remove_special_characters(text)

    # Step 3: Remove stop words
    text = remove_stopwords(text)

    # Step 4: Strip extra whitespace
    text = text.strip()

    return text


def preprocess_dataframe(df, text_column="text"):
    """
    Applies the preprocessing pipeline to all emails in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing raw email texts.
        text_column (str): Name of the column with email text.

    Returns:
        pd.DataFrame: DataFrame with an added 'cleaned_text' column.
    """
    print("[INFO] Preprocessing email texts...")
    df = df.copy()
    df["cleaned_text"] = df[text_column].apply(preprocess_text)
    print(f"[INFO] Preprocessing complete. {len(df)} emails processed.")
    return df


# ---------------------------------------------------------------
# Quick test: run this file directly to see preprocessing in action
# ---------------------------------------------------------------
if __name__ == "__main__":
    sample_email = """
    URGENT: Your bank account has been COMPROMISED!
    Click here -> http://secure-bank-login.xyz to verify 
    your password & account details NOW!!! 
    """

    print("Original Text:")
    print(sample_email)
    print("\nAfter Preprocessing:")
    print(preprocess_text(sample_email))

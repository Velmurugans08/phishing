"""
model_trainer.py
----------------
This module trains and compares two machine learning models:
    1. Logistic Regression
    2. Naive Bayes (Multinomial)

Both models are trained on the same feature set, their performances
are compared, and the best model is automatically selected and saved.

Author: Phishing Detection Project
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from dataset_loader import load_dataset
from preprocessor import preprocess_dataframe
from feature_extractor import FeatureExtractor


# ---------------------------------------------------------------
# Directory for saving trained models
# ---------------------------------------------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.
    80% of data is used for training, 20% for testing.

    Args:
        df (pd.DataFrame): The full preprocessed dataset.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: (train_df, test_df) - two DataFrames.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]  # Ensure equal phishing/safe split in both sets
    )

    print(f"[INFO] Data split complete:")
    print(f"       Training samples : {len(train_df)}")
    print(f"       Testing samples  : {len(test_df)}")

    return train_df, test_df


def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression classifier.

    Logistic Regression is a simple but powerful linear model that
    works well for text classification tasks. It finds a decision
    boundary between phishing and legitimate emails.

    Args:
        X_train: Feature matrix for training data.
        y_train: Labels for training data (1=Phishing, 0=Safe).

    Returns:
        LogisticRegression: Trained model object.
    """
    print("\n[MODEL] Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,       # Allow enough iterations to converge
        random_state=42,     # For reproducibility
        solver="lbfgs",      # Efficient solver for small datasets
        C=1.0                # Regularization strength (default)
    )

    model.fit(X_train, y_train)
    print("[MODEL] Logistic Regression training complete!")
    return model


def train_naive_bayes(X_train, y_train):
    """
    Trains a Multinomial Naive Bayes classifier.

    Naive Bayes is a probabilistic classifier based on Bayes theorem.
    It is especially popular for spam/phishing detection because it
    works well with word frequency features like TF-IDF.

    Note: MultinomialNB requires non-negative feature values.
          TF-IDF values are always >= 0, so this works perfectly.

    Args:
        X_train: Feature matrix for training data.
        y_train: Labels for training data (1=Phishing, 0=Safe).

    Returns:
        MultinomialNB: Trained model object.
    """
    print("\n[MODEL] Training Naive Bayes (Multinomial)...")

    model = MultinomialNB(
        alpha=1.0   # Laplace smoothing to handle unseen words
    )

    model.fit(X_train, y_train)
    print("[MODEL] Naive Bayes training complete!")
    return model


def evaluate_model_quick(model, X_test, y_test, model_name):
    """
    Quickly evaluates a model on test data and returns accuracy.
    (Full evaluation with plots is done in evaluator.py)

    Args:
        model: Trained sklearn model.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name (str): Name of the model for display.

    Returns:
        float: Accuracy score (0 to 1).
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[EVAL] {model_name} Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def select_best_model(lr_model, nb_model, lr_accuracy, nb_accuracy):
    """
    Compares both models and returns the best one based on accuracy.

    Args:
        lr_model: Trained Logistic Regression model.
        nb_model: Trained Naive Bayes model.
        lr_accuracy (float): Accuracy of Logistic Regression.
        nb_accuracy (float): Accuracy of Naive Bayes.

    Returns:
        tuple: (best_model, best_name) - the winning model and its name.
    """
    print("\n[COMPARISON] Comparing model performances...")
    print(f"  Logistic Regression Accuracy : {lr_accuracy * 100:.2f}%")
    print(f"  Naive Bayes Accuracy         : {nb_accuracy * 100:.2f}%")

    if lr_accuracy >= nb_accuracy:
        print("[RESULT] Best Model: Logistic Regression")
        return lr_model, "Logistic Regression"
    else:
        print("[RESULT] Best Model: Naive Bayes")
        return nb_model, "Naive Bayes"


def save_model(model, model_name, filepath=None):
    """
    Saves a trained model to disk using joblib.

    Args:
        model: Trained sklearn model to save.
        model_name (str): Name of the model (used for filename if no path given).
        filepath (str, optional): Custom save path.
    """
    if filepath is None:
        safe_name = model_name.lower().replace(" ", "_")
        filepath = os.path.join(MODELS_DIR, f"{safe_name}.pkl")

    joblib.dump(model, filepath)
    print(f"[SAVE] Model saved to: {filepath}")


def load_model(filepath):
    """
    Loads a saved model from disk.

    Args:
        filepath (str): Path to the saved model file.

    Returns:
        Trained sklearn model.
    """
    model = joblib.load(filepath)
    print(f"[LOAD] Model loaded from: {filepath}")
    return model


def train_and_save_all():
    """
    Main training function that runs the complete training pipeline:
        1. Load dataset
        2. Preprocess text
        3. Extract features
        4. Train both models
        5. Compare and select best model
        6. Save all models and the feature extractor

    Returns:
        dict: A dictionary containing both models, the best model,
              the feature extractor, and test data for evaluation.
    """
    print("=" * 60)
    print("  PHISHING EMAIL DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Preprocess the text
    df = preprocess_dataframe(df, text_column="text")

    # Step 3: Split into train and test sets
    train_df, test_df = split_data(df)

    # Step 4: Extract features
    extractor = FeatureExtractor(max_features=3000)
    X_train = extractor.fit_transform(train_df, "cleaned_text", "text")
    X_test = extractor.transform(test_df, "cleaned_text", "text")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Step 5: Train both models
    lr_model = train_logistic_regression(X_train, y_train)
    nb_model = train_naive_bayes(X_train, y_train)

    # Step 6: Quick evaluation
    print("\n[EVAL] Evaluating models on test set...")
    lr_acc = evaluate_model_quick(lr_model, X_test, y_test, "Logistic Regression")
    nb_acc = evaluate_model_quick(nb_model, X_test, y_test, "Naive Bayes")

    # Step 7: Select best model
    best_model, best_name = select_best_model(lr_model, nb_model, lr_acc, nb_acc)

    # Step 8: Save all artifacts
    save_model(lr_model, "logistic_regression")
    save_model(nb_model, "naive_bayes")
    save_model(best_model, "best_model")
    extractor.save(os.path.join(MODELS_DIR, "feature_extractor.pkl"))

    print("\n[DONE] Training complete! All models saved in the 'models/' folder.")
    print(f"[DONE] Best Model: {best_name}")

    # Return everything needed for evaluation
    return {
        "lr_model": lr_model,
        "nb_model": nb_model,
        "best_model": best_model,
        "best_name": best_name,
        "extractor": extractor,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
        "lr_accuracy": lr_acc,
        "nb_accuracy": nb_acc
    }


# ---------------------------------------------------------------
# Run training when this file is executed directly
# ---------------------------------------------------------------
if __name__ == "__main__":
    results = train_and_save_all()

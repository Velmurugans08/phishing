"""
train_and_evaluate.py
---------------------
Master script that ties everything together.
Run this file to:
    1. Train both models (Logistic Regression + Naive Bayes)
    2. Evaluate both models (metrics + charts)
    3. Save the best model and feature extractor for the web UI

Usage:
    python train_and_evaluate.py

Author: Phishing Detection Project
"""

from model_trainer import train_and_save_all
from evaluator import run_full_evaluation


def main():
    """
    Runs the full training and evaluation pipeline.
    """
    print("\n" + "#" * 65)
    print("#   PHISHING EMAIL DETECTION — FULL TRAINING & EVALUATION   #")
    print("#" * 65 + "\n")

    # Step 1: Train all models and save them
    training_results = train_and_save_all()

    # Step 2: Run full evaluation with metrics and charts
    eval_results = run_full_evaluation(training_results)

    # Step 3: Final summary
    lr = eval_results["lr_metrics"]
    nb = eval_results["nb_metrics"]

    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<20} {'Logistic Regression':>20} {'Naive Bayes':>15}")
    print(f"  {'-' * 57}")
    print(f"  {'Accuracy':<20} {lr['accuracy']*100:>19.2f}% {nb['accuracy']*100:>14.2f}%")
    print(f"  {'Precision':<20} {lr['precision']*100:>19.2f}% {nb['precision']*100:>14.2f}%")
    print(f"  {'Recall':<20} {lr['recall']*100:>19.2f}% {nb['recall']*100:>14.2f}%")
    print(f"  {'F1-Score':<20} {lr['f1_score']*100:>19.2f}% {nb['f1_score']*100:>14.2f}%")
    print("=" * 65)

    best = training_results["best_name"]
    print(f"\n  [OK] Best Model Selected: {best}")
    print(f"  [OK] Model saved in     : models/")
    print(f"  [OK] Charts saved in    : reports/")
    print(f"\n  >> Now run: python app.py")
    print(f"     Then open: http://127.0.0.1:5000 in your browser")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

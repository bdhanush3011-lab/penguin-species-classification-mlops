"""Lab 3 - evaluate the baseline model, save metrics, figures and predictions."""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

from common import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, PROCESSED_DIR, ensure_dirs


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def run_evaluation(model_path="models/svm_baseline.pkl", tag="baseline"):
    print("Starting Model Evaluation...")
    ensure_dirs()
    X_test = np.load(f"{PROCESSED_DIR}/X_test_final.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    model = joblib.load(model_path)
    le = joblib.load("models/label_encoder.pkl")

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n--- Model Evaluation Report ---")
    for k, v in metrics.items():
        print(f"{k:<20}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    with open(f"{METRICS_DIR}/{tag}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        le.inverse_transform(y_test), le.inverse_transform(y_pred), ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {tag}")
    fig.savefig(f"{FIGURES_DIR}/{tag}_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    raw = pd.read_csv(f"{PROCESSED_DIR}/test_raw.csv")
    raw["predicted_species"] = le.inverse_transform(y_pred)
    raw.to_csv(f"{PREDICTIONS_DIR}/{tag}_predictions.csv", index=False)
    raw[raw["species"] != raw["predicted_species"]].to_csv(
        f"{PREDICTIONS_DIR}/{tag}_misclassified.csv", index=False)

    print("Evaluation complete. Outputs saved under outputs/.")


if __name__ == "__main__":
    run_evaluation()

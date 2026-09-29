"""Lab 4 - MLflow experiment tracking (several runs, one experiment)."""
import warnings

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated"
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import SVC

from common import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE, setup_mlflow


def load_data():
    return (np.load(f"{PROCESSED_DIR}/X_train_final.npy"), np.load(f"{PROCESSED_DIR}/X_test_final.npy"),
            np.load(f"{PROCESSED_DIR}/y_train.npy"), np.load(f"{PROCESSED_DIR}/y_test.npy"))


def train_and_track(model, run_name, params, family):
    import mlflow
    import mlflow.sklearn
    X_train, X_test, y_train, y_test = load_data()
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_param("model_family", family)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
        ax.set_title(f"Confusion Matrix - {run_name}")
        cm_path = f"{ARTIFACTS_DIR}/confusion_matrix_{run_name}.png"
        fig.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(f"{PROCESSED_DIR}/dataset_metadata.json", artifact_path="metadata")
        mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)
        joblib.dump(model, f"models/{run_name}.pkl")
        print(f"Run '{run_name}' successfully tracked!")


if __name__ == "__main__":
    setup_mlflow()
    train_and_track(SVC(C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
                    "SVM_Baseline", {"C": 1.0, "kernel": "rbf", "gamma": "scale"}, "SVM")
    train_and_track(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
                    "RandomForest_Baseline", {"n_estimators": 100, "random_state": RANDOM_STATE}, "RandomForest")
    train_and_track(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                    "LogisticRegression_Baseline", {"max_iter": 1000}, "LogisticRegression")

"""Lab 6 - train a model and register it in the MLflow Model Registry."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from common import PROCESSED_DIR, RANDOM_STATE, REGISTERED_MODEL_NAME, setup_mlflow


# Model version name
RUN_NAME = "SVM_Registry_V1"
PARAMS = {"C": 1, "kernel": "rbf", "gamma": "scale"}


def train_and_register_model():
    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    mlflow = setup_mlflow()
    import mlflow.sklearn

    try:
        X_train = np.load(f"{PROCESSED_DIR}/X_train_final.npy")
        X_test = np.load(f"{PROCESSED_DIR}/X_test_final.npy")
        y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
        y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run Lab 5 first.")
        raise SystemExit(1)

    with mlflow.start_run(run_name=RUN_NAME):

        # Log parameters
        mlflow.log_params(PARAMS)
        mlflow.log_param("model_family", "SVM")

        # ✅ FIX 1: Replace deprecated probability=True
        base_model = SVC(random_state=RANDOM_STATE, **PARAMS)
        model = CalibratedClassifierCV(base_model, ensemble=False)

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
        }

        mlflow.log_metrics(metrics)

        # Log preprocessing artifacts
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")
        mlflow.log_artifact("models/label_encoder.pkl", artifact_path="preprocessing_pipeline")

        # ✅ FIX 2: Allow MLflow to trust calibrated model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="svm_model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration"
            ]
        )

        print(f"[SUCCESS] F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")
        print(f"[SUCCESS] Registered under '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    train_and_register_model()
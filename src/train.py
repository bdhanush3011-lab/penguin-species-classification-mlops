"""Lab 3 - baseline model training (SVM)."""

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from common import PROCESSED_DIR, RANDOM_STATE, ensure_dirs


def run_training():
    print("Starting Model Training...")
    ensure_dirs()

    # Load data
    X_train = np.load(f"{PROCESSED_DIR}/X_train_final.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")

    # Base SVM model
    base_model = SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        random_state=RANDOM_STATE
    )

    # Calibrated model (replaces probability=True)
    model = CalibratedClassifierCV(base_model, ensemble=False)

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "models/svm_baseline.pkl")

    print("Model training complete and saved to models/svm_baseline.pkl")


if __name__ == "__main__":
    run_training()
"""Lab 5 - validate the arrays produced by preprocessing."""
import json
import numpy as np

from common import ARTIFACTS_DIR, PROCESSED_DIR, ensure_dirs


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")
    ensure_dirs()
    try:
        X_train = np.load(f"{PROCESSED_DIR}/X_train_final.npy")
        X_test = np.load(f"{PROCESSED_DIR}/X_test_final.npy")
        y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
        y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []
    if np.isnan(X_train).any():
        errors.append("NaNs detected in X_train after preprocessing.")
    if np.isnan(X_test).any():
        errors.append("NaNs detected in X_test after preprocessing.")
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: {X_train.shape[1]} vs {X_test.shape[1]} columns.")
    if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between features and labels.")
    if len(np.unique(y_train)) != 3 or len(np.unique(y_test)) != 3:
        errors.append("Expected 3 species classes in both train and test sets.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {"X_train_shape": list(X_train.shape), "X_test_shape": list(X_test.shape),
                              "y_train_shape": list(y_train.shape), "y_test_shape": list(y_test.shape)},
        "data_quality": {"missing_values_X_train": int(np.isnan(X_train).sum()),
                         "missing_values_X_test": int(np.isnan(X_test).sum())},
        "errors": errors,
    }
    path = f"{ARTIFACTS_DIR}/preprocessing_summary_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False
    print(f"[SUCCESS] Output Validation PASSED. Report saved to {path}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_preprocessing_outputs() else 1)

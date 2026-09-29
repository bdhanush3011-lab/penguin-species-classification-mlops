"""Lab 4 - train twice with fixed seeds and confirm identical results."""
import warnings

warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated"
)
import json
import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from common import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE, ensure_dirs


def run_deterministic_test():
    print("Running Reproducibility Validation...")
    ensure_dirs()
    X_train = np.load(f"{PROCESSED_DIR}/X_train_final.npy")
    X_test = np.load(f"{PROCESSED_DIR}/X_test_final.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{PROCESSED_DIR}/y_test.npy")

    params = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": RANDOM_STATE}
    scores = []
    for _ in range(2):
        m = SVC(**params).fit(X_train, y_train)
        scores.append(f1_score(y_test, m.predict(X_test), average="weighted"))

    ok = scores[0] == scores[1]
    report = {"test_name": "Pipeline Reproducibility Validation", "parameters": params,
              "execution_1_f1": scores[0], "execution_2_f1": scores[1],
              "is_strictly_reproducible": bool(ok), "status": "PASSED" if ok else "FAILED"}
    with open(f"{ARTIFACTS_DIR}/reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 F1: {scores[0]:.6f}\nExecution 2 F1: {scores[1]:.6f}")
    print("SUCCESS: Pipeline is reproducible." if ok else "FAILED: Pipeline is non-deterministic.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run_deterministic_test()

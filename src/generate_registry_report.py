"""Lab 6 - write a deployment-ready JSON report for the current champion."""
import json
from mlflow.tracking import MlflowClient

from common import ARTIFACTS_DIR, REGISTERED_MODEL_NAME, setup_mlflow


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")
    setup_mlflow()
    client = MlflowClient()
    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except Exception:
        print("[ERROR] No model has the 'champion' alias!")
        raise SystemExit(1)

    run = client.get_run(champion.run_id)
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    report = {
        "registry_status": "READY_FOR_DEPLOYMENT",
        "model_lineage": {"registered_name": REGISTERED_MODEL_NAME, "version": int(champion.version),
                          "alias": "champion", "run_id": champion.run_id,
                          "artifact_uri": champion.source},
        "performance_metrics": {k: run.data.metrics.get(k) for k in keys},
        "hyperparameters": run.data.params,
        "preprocessing_dependency": ["models/preprocessor.pkl", "models/label_encoder.pkl"],
    }
    path = f"{ARTIFACTS_DIR}/production_model_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[SUCCESS] Report generated for v{champion.version} -> {path}")


if __name__ == "__main__":
    generate_registry_report()

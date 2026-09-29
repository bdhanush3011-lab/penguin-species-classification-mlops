"""Shared constants and helpers used by every script in src/."""
import os

RAW_DATA_PATH = "data/raw/penguins_size.csv"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
FIGURES_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"
PREDICTIONS_DIR = "outputs/predictions"
ARTIFACTS_DIR = "artifacts"

TARGET = "species"
NUMERIC_FEATURES = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]

RANDOM_STATE = 42
TEST_SIZE = 0.20

EXPERIMENT_NAME = "Penguin_Species_Classification"
REGISTERED_MODEL_NAME = "Penguin_Species_Production_Model"


def ensure_dirs():
    for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR,
              PREDICTIONS_DIR, ARTIFACTS_DIR, "logs"):
        os.makedirs(d, exist_ok=True)


def setup_mlflow():
    """Point MLflow at a local SQLite DB (needed for the model registry);
    run artifacts are stored under artifacts/mlruns."""
    import mlflow
    ensure_dirs()
    mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=os.path.abspath(os.path.join(ARTIFACTS_DIR, "mlruns")).replace("\\", "/"),
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow

"""Lab 3 - baseline preprocessing: clean, split, impute, scale, encode, save."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from common import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROCESSED_DIR, RANDOM_STATE,
                    RAW_DATA_PATH, TARGET, TEST_SIZE, ensure_dirs)


def load_and_clean(path=RAW_DATA_PATH):
    df = pd.read_csv(path)
    # The raw file has one invalid sex value "." -> treat as missing
    df["sex"] = df["sex"].replace(".", np.nan)
    return df


def build_preprocessor():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, NUMERIC_FEATURES),
                              ("cat", cat_pipe, CATEGORICAL_FEATURES)])


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")
    ensure_dirs()
    df = load_and_clean()

    X = df.drop(columns=[TARGET])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])

    # Split BEFORE fitting any transformer to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    preprocessor = build_preprocessor()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    np.save(f"{PROCESSED_DIR}/X_train_final.npy", X_train_final)
    np.save(f"{PROCESSED_DIR}/X_test_final.npy", X_test_final)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train.astype(np.int64))
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test.astype(np.int64))
    # Keep the raw test rows so evaluate.py can export readable error analysis
    X_test.assign(species=label_encoder.inverse_transform(y_test)).to_csv(
        f"{PROCESSED_DIR}/test_raw.csv", index=False)

    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    metadata = {
        "dataset_name": "Palmer Penguins (penguins_size)",
        "target": TARGET,
        "classes": list(label_encoder.classes_),
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open(f"{PROCESSED_DIR}/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Train shape: {X_train_final.shape} | Test shape: {X_test_final.shape}")
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()

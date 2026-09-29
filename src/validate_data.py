"""Lab 5 - schema validation of the raw penguin data with pandera."""
import os
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from common import ARTIFACTS_DIR, RAW_DATA_PATH


def get_penguin_schema():
    positive = Check.gt(0)
    return pa.DataFrameSchema({
        "species": Column(str, Check.isin(["Adelie", "Chinstrap", "Gentoo"])),
        "island": Column(str, Check.isin(["Torgersen", "Biscoe", "Dream"])),
        "culmen_length_mm": Column(float, positive, nullable=True),
        "culmen_depth_mm": Column(float, positive, nullable=True),
        "flipper_length_mm": Column(float, positive, nullable=True),
        "body_mass_g": Column(float, positive, nullable=True),
        # "." is a known invalid value that preprocess.py converts to NaN, so tolerate it here
        "sex": Column(str, Check.isin(["MALE", "FEMALE", "."]), nullable=True),
    }, strict=True)


def validate_schema(df, output_report_name="schema_validation_errors.csv"):
    print(f"[INFO] Validating schema (Records: {len(df)})...")
    try:
        get_penguin_schema().validate(df, lazy=True)
        print("[SUCCESS] Schema Validation PASSED.")
        return True
    except pa.errors.SchemaErrors as err:
        print("[ERROR] Schema Validation FAILED.")
        failures = err.failure_cases[["schema_context", "column", "check", "failure_case", "index"]]
        print(failures.to_string())
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, output_report_name)
        failures.to_csv(path, index=False)
        print(f"[INFO] Failure report saved to '{path}'.")
        return False


if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] File not found: {RAW_DATA_PATH}")
        raise SystemExit(1)
    ok = validate_schema(pd.read_csv(RAW_DATA_PATH), "baseline_validation.csv")
    raise SystemExit(0 if ok else 1)

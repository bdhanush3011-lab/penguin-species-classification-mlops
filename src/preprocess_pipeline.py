"""Lab 5 - production preprocessing (same logic as preprocess.py, with logging)."""
from preprocess import run_preprocessing

if __name__ == "__main__":
    print("[INFO] Starting sklearn Pipeline preprocessing...")
    run_preprocessing()
    print("[SUCCESS] Preprocessing pipeline saved as models/preprocessor.pkl")

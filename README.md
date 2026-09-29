# Penguin Species Classification (MLOps Project)

Multiclass classification of Palmer penguins (Adelie, Chinstrap, Gentoo) from
physical measurements, organised as a small MLOps project: reproducible
preprocessing, baseline training, MLflow tracking, data validation and a model registry.

## Project structure

```
data/raw/penguins_size.csv          raw dataset (344 rows)
data/processed/                     arrays + dataset_metadata.json (generated)
notebooks/project_implementation.ipynb   EDA, model comparison, tuning
src/                                pipeline scripts (preprocess, train, evaluate, ...)
pipelines/                          one runner per lab (lab 3 to lab 6)
models/                             trained models + preprocessor (generated)
outputs/{figures,metrics,predictions}    evaluation results (generated)
logs/                               execution logs
artifacts/                          MLflow DB/runs and validation reports (generated)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run every command below from the **project root**.

## Running the labs

| Lab | Command | What it does |
|-----|---------|--------------|
| 3 | `python pipelines/run_lab3_baseline.py` | preprocess, train baseline SVM, evaluate |
| 4 | `python pipelines/run_lab4_tracking.py` | MLflow tracking (SVM, RF, LogReg) + reproducibility check |
| 5 | `python pipelines/run_lab5_pipeline.py` | pandera schema validation, preprocessing pipeline, output validation |
| 6 | `python pipelines/run_lab6_registry.py` | register model, champion/challenger promotion, deployment report |

Run Lab 3 (or 5) before Lab 4 and 6, because they need the processed data and `models/preprocessor.pkl`.

To view MLflow results: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`

To create a second registry version for Lab 6, edit `RUN_NAME` and `PARAMS` in
`src/train_registry.py` and run the lab again; the better model (by F1) becomes the `champion`.

## Data notes

- Target: `species`. Features: `culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g`, `island`, `sex`.
- The `sex` column contains one invalid value `.`, treated as missing.
- Missing values: median imputation (numeric), most-frequent imputation (categorical), fitted on the training split only.
- 80/20 stratified split, `random_state=42`.

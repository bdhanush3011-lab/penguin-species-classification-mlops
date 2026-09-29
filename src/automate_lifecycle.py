"""Lab 6 - champion/challenger promotion using registry ALIASES.

MLflow's old stage API (Staging/Production) is deprecated, so this uses the
"champion" alias instead. The decision metric is f1_score.
"""
from mlflow.tracking import MlflowClient

from common import REGISTERED_MODEL_NAME, setup_mlflow

METRIC = "f1_score"


def score_of(client, mv):
    return client.get_run(mv.run_id).data.metrics.get(METRIC, 0.0)


def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    setup_mlflow()
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        print("[INFO] No registered versions found.")
        return

    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    except Exception:
        champion = None

    candidates = [mv for mv in versions if champion is None or mv.version != champion.version]
    if not candidates:
        print("[INFO] No challengers to evaluate.")
        return

    best = max(candidates, key=lambda mv: score_of(client, mv))
    best_score = score_of(client, best)
    print(f"[INFO] Best challenger: v{best.version} ({METRIC}: {best_score:.4f})")

    if champion is None:
        print("[INFO] No champion yet - challenger wins by default.")
        promote = True
    else:
        champ_score = score_of(client, champion)
        print(f"[INFO] Current champion: v{champion.version} ({METRIC}: {champ_score:.4f})")
        promote = best_score > champ_score
        print("[SUCCESS] Challenger beat the champion!" if promote
              else "[INFO] Champion retains its title.")

    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", best.version)
        print(f"[INFO] v{best.version} is now the champion.")
    print("[SUCCESS] Automated lifecycle execution complete!")


if __name__ == "__main__":
    automate_champion_challenger()

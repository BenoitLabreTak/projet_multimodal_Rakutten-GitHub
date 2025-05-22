import os
import mlflow
import dagshub
import logging

def init_mlflow_if_enabled(experiment_name: str) -> bool:
    if os.getenv("ENABLE_MLFLOW_LOGGING", "false").lower() != "true":
        logging.info("🔕 MLflow tracking désactivé (ENABLE_MLFLOW_LOGGING != true).")
        return False

    try:
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        if not all([repo_owner, repo_name, tracking_uri]):
            raise ValueError("🛑 Variables DagsHub manquantes dans l'environnement (.env)")

        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        logging.info(f"📡 MLflow connecté via DagsHub à {tracking_uri}, exp: {experiment_name}")
        return True

    except Exception as e:
        logging.warning(f"⚠️ MLflow non initialisé : {str(e)}")
        return False  # <== Ajoute bien ce `return False`
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml import Model, log_metadata
from zenml.enums import ArtifactType

from typing_extensions import Annotated
from typing import Dict

import requests
import config


@step
def evaluate_image_model_step() -> Annotated[
    Dict[
        Annotated[float, "average_confidence_score"],
        Annotated[float, "weighted_f1_score"]
    ], "scores"
]:
    """
    Step d'évaluation automatique du modèle
    À implémenter: charger le modèle, évaluer sur un sous-ensemble, retourner les métriques.
    """
    with open('data/retraining_text/test_1pct.csv', 'rb') as f:
        files = {"file": ("test_1pct.csv", f, "text/csv")}
        response = requests.post('http://localhost:8000/evaluate/image/file?sample_size=50', files=files)
    if response.status_code == 200:
        metrics = response.json()
    else:
        metrics = {} #TODO: déclencher une erreur
    return metrics


@step
def conditional_image_retrain_step(metrics: dict) -> Annotated[bool, "trigger_retrain"]:
    """
    Step de réentraînement conditionnel (
    si le score F1 < seuil, réentraîner et retourner True, sinon False.
    """
    f1_score = metrics.get("weighted_f1_score", 0)
    threshold = 0.8
    retrain = f1_score < threshold
    return retrain

@step(model=Model(name="image_model", artifact_type=ArtifactType.MODEL))
def image_retrain_step(retrain_triggered: bool) -> Annotated[str, "model_path"]:
    """
    Step de réentrainement
    """
    response = requests.post('http://localhost:8000/train/image?epochs=1&batch_size=8')
    if response.status_code == 200:
        output = response.json()
        log_metadata(metadata={"new_f1": output.get("new_f1", 0)}, infer_model=True)
        if output["model_saved"]:
            return output["model_path"]
        else:
            return ""

@step
def image_notify_slack_on_success_step(retrain_triggered: bool, model_str : str) -> None:
    """
    Step de notification Slack
    """
    if retrain_triggered:
        msg = f"Nouveau modèle image sauvegardé : {model_str}"
        print(msg)
        requests.post(config.SLACK_WEBHOOK_URL, 
                      json={"text": msg},
                      headers={"Content-Type": "application/json"})
    else:
        msg = "Pas de nouveau modèle image sauvegardé : les performances n'ont pas été améliorées."
        print(msg)
        requests.post(config.SLACK_WEBHOOK_URL, 
                      json={"text": msg},
                      headers={"Content-Type": "application/json"})


@pipeline(enable_cache=False)
def image_auto_eval_and_retrain_pipeline():
    metrics = evaluate_image_model_step()
    retrain_triggered = conditional_image_retrain_step(metrics)
    if retrain_triggered:
        model_str = image_retrain_step(retrain_triggered)
        image_notify_slack_on_success_step(retrain_triggered, model_str)


if __name__ == "__main__":
    image_auto_eval_and_retrain_pipeline()

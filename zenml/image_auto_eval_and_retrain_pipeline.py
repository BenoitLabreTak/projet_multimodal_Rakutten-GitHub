from zenml.pipelines import pipeline
from zenml.steps import step

@step
def evaluate_model_step() -> dict:
    """
    Step d'évaluation automatique du modèle (dummy).
    À implémenter: charger le modèle, évaluer sur un sous-ensemble, retourner les métriques.
    """
    metrics = {"f1_score": 0.85}  # Valeur fictive
    3+2
    return metrics


@step
def conditional_retrain_step(metrics: dict) -> bool:
    """
    Step de réentraînement conditionnel (dummy).
    À implémenter: si le score F1 < seuil, réentraîner et retourner True, sinon False.
    """
    f1_score = metrics.get("f1_score", 0)
    threshold = 0.9
    retrain = f1_score < threshold
    return retrain

@step
def notify_slack_on_success_step(retrain_triggered: bool) -> None:
    """
    Step de notification Slack (dummy).
    À implémenter: envoyer une notification si retrain_triggered est True.
    """
    if retrain_triggered:
        print("Slack notification: Nouveau modèle sauvegardé.")
    else:
        print("Slack notification: Pas de nouveau modèle.")


@pipeline
def text_auto_eval_and_retrain_pipeline():
    metrics = evaluate_model_step()
    retrain_triggered = conditional_retrain_step(metrics)
    notify_slack_on_success_step(retrain_triggered)

if __name__ == "__main__":
    text_auto_eval_and_retrain_pipeline()
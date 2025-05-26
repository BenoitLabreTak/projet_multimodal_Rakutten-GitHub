from zenml.pipelines import pipeline
from text_auto_eval_and_retrain_pipeline import text_auto_eval_and_retrain_pipeline
from image_auto_eval_and_retrain_pipeline import image_auto_eval_and_retrain_pipeline

@pipeline
def allmodels_auto_eval_and_retrain_pipeline():
    """
    Pipeline combinant les pipelines d'évaluation et de réentraînement automatique
    pour les modèles texte et image.
    """
    text_auto_eval_and_retrain_pipeline()
    image_auto_eval_and_retrain_pipeline()

if __name__ == "__main__":
    allmodels_auto_eval_and_retrain_pipeline()

import os
import pandas as pd
import numpy as np

from zenml import pipeline, step
from zenml import Model, log_metadata
from zenml import ArtifactConfig
from zenml.enums import ArtifactType
from zenml.client import Client

from typing_extensions import Annotated
from typing import Tuple

from transformers import CamembertForSequenceClassification
import torch.nn as nn

from sklearn.ensemble import StackingClassifier
from joblib import load

# A ajouter: les proba prédictions de chaque modèle pour train et test
@step(model=Model(name="stacking_model", artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}"))
def multimodal_model_train(model_text: CamembertForSequenceClassification, model_image: nn.Module) -> Annotated[
    StackingClassifier,
    ArtifactConfig(name="stacking_model",
                   artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}")
]:
    model = StackingClassifier()
    return model


@step(model=Model(name="stacking_model", artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}"))
def multimodal_model_load(stacking_model_path: str) -> Annotated[
    StackingClassifier,
    ArtifactConfig(name="stacking_model", artifact_type=ArtifactType.MODEL)
]:
    return load(os.path.join("models/stacking", "stacking_xgb_model.joblib"))


@step
def multimodel_model_evaluate(predict_proba_text: pd.DataFrame, predict_proba_image: pd.DataFrame, model: StackingClassifier) -> Tuple[
    Annotated[float, "f1_score"],
    Annotated[pd.DataFrame, "confusion_matrix"],
    Annotated[pd.DataFrame, "predictions"]
]:
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.98})

@pipeline
def pipeline_multimodel_train(
    model_text: CamembertForSequenceClassification, 
    model_image: nn.Module,
    textmodel_pred: pd.DataFrame,
    imagemodel_pred: pd.DataFrame,
    external_stackingmodel: str = None,
    stackmodel_version: str = None,
    
) -> Tuple[
    Annotated[float, "f1_score"],
    Annotated[pd.DataFrame, "confusion_matrix"],
    Annotated[pd.DataFrame, "predictions"]
]:
    if stackmodel_version is None:
        stackmodel_version = str(Client().get_pipeline("global_pipeline").id)

    if external_stackingmodel is not None:
        model_stack = multimodal_model_load.with_options(
            substitutions={"STACKMODEL_VERSION": stackmodel_version}
        )(external_stackingmodel)
    else:
        model_stack = multimodal_model_train.with_options(
            substitutions={"STACKMODEL_VERSION": stackmodel_version}
        )(model_text, model_image)
    f1_score, confusion_matrix, predictions = multimodel_model_evaluate(textmodel_pred, imagemodel_pred, model_stack)
    
    return f1_score, confusion_matrix, predictions

if __name__ == "__main__":
    # Récupération du paramètre MODEL_VERSION passé en argument, sinon "0.0"
    version_model = sys.argv[1] if len(sys.argv) > 1 else "0.0"
    # pipeline_textual(external_textmodel="models/camembert_modele_1")
    pipeline_multimodel_train(
        textmodel_pred=pd.DataFrame(),
        imagemodel_pred=pd.DataFrame(),
        external_stackingmodel="models/stacking/stacking_xgb_model.joblib",
        textmodel_version=version_model,
        imagemodel_version=version_model,
        stackmodel_version=version_model
    ) 
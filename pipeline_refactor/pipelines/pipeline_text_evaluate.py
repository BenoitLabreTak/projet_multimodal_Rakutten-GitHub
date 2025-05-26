import os
import sys
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

from pipelines import pipeline_text_preprocess
from utils import generate_version
from utils import TextDataSchema

# bug zenml 0.82: l'interface graphique ne reconnait pas le nom du modèls si on annote pas le résultat de la fonction
# bug zenml 0.82: l'interface graphique ne reconnait pas les métadonnées du modèles si on annote le résultat de la fonction
@step(model=Model(name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}"))
def textdata_train(dataset_train: pd.DataFrame,
                   dataset_val: pd.DataFrame) -> Annotated[
                       CamembertForSequenceClassification,
                       ArtifactConfig(
                           name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}")
]:
    log_metadata(metadata={"train_loss": 0.95,
                 "val_loss": 0.92}, infer_model=True)
    return CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=27)





@step(model=Model(name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}"))
def textdat_model_load(model_path: str) -> Annotated[CamembertForSequenceClassification, ArtifactConfig(name="text_model", artifact_type=ArtifactType.MODEL)]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle Camembert n'existe pas à l'emplacement : {model_path}")
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    return model


@step
def textdata_evaluate(dataset_test: pd.DataFrame, model: CamembertForSequenceClassification) -> Tuple[
    Annotated[float, "textmodel_f1_score"],
    Annotated[pd.DataFrame, "textmodel_confusion_matrix"],
    Annotated[pd.DataFrame, "textmodel_predictions"]
]:
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.75})

@pipeline
def pipeline_text_evaluate(
    dataset_train: pd.DataFrame = None, # dataframe de préprocessing à utiliser en priorité si fourni
    dataset_test: pd.DataFrame = None,
    dataset_val: pd.DataFrame = None,
    textdata_version: str = None, # version de l'artefact de préprocessing text à utiliser (si ni dataframe, ni version fournie, un pipeline de prétraitement sera lancé)
    textmodel_version: str = None, # version de l'artefact de modèle à utilisé si fourni (si textmodel_version est None)
    external_textmodel: str = None # chemin vers le modèle à utiliser Si rien fourni, un step d'entrainement sera lancé
) -> Tuple[
    Annotated[float, "textmodel_f1_score"],
    Annotated[pd.DataFrame, "textmodel_confusion_matrix"],
    Annotated[pd.DataFrame, "textmodel_predictions"],
    Annotated[CamembertForSequenceClassification, "model_text"]
]:
    client = Client()
    # récupération de  dataset_train, dataset_test, dataset_val 
    if textdata_version is None:
        # récupération via le passage en paramètre
        # si pas le cas, on lance un pipeline de prétraitement
        if dataset_train is None or dataset_test is None or dataset_val is None:
            dataset_train, dataset_test, dataset_val = pipeline_text_preprocess(enable_translation=False)
    else:
        # récupération via version d'artefact
        dataset_train = client.get_artifact_version("preprocessed_dataset_train", textdata_version)
        dataset_test = client.get_artifact_version("preprocessed_dataset_test", textdata_version)
        dataset_val = client.get_artifact_version("preprocessed_dataset_val", textdata_version)
    
    # si modele externe fourni, on l'utilise
    if external_textmodel is not None:
        model_text = textdat_model_load.with_options(
            substitutions={"TEXTMODEL_VERSION": textmodel_version}
        )(external_textmodel)
    else: # si pas de modèle externe fourni
            if textmodel_version is None:
                # si ni modèle externe, ni version de mdoèle fourni: on entraine
                textmodel_version = generate_version()
                model_text = textdata_train(dataset_train, dataset_val)
            else: 
                # si version de modèle fourni, on cherche l'artefact
                model_text = client.get_model_version("text_model", textmodel_version)
        
    textmodel_f1_score, textmodel_confusion_matrix, textmodel_pred = textdata_evaluate.with_options(
        substitutions={"TEXTMODEL_VERSION": textmodel_version}
    )(dataset_test, model_text)
    return textmodel_f1_score, textmodel_confusion_matrix, textmodel_pred, model_text

"""
Impossible à faire fonctionner
if __name__ == "__main__":
    # premier paramètre: textdata_version
    # second paramètre: textdata_version
    textdata_version = sys.argv[1] if len(sys.argv) > 1 else None
    textmodel_version = sys.argv[2] if len(sys.argv) > 2 else None
    pipeline_text_evaluate(
        dataset_train=pd.DataFrame(),
        dataset_test=pd.DataFrame(),
        dataset_val=pd.DataFrame(),
        textdata_version=textdata_version,
        textmodel_version=textmodel_version,
        external_textmodel="models/camembert_modele_1")
"""

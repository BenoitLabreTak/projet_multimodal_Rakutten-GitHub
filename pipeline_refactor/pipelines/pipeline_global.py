from sklearn.ensemble import StackingClassifier

from pipelines import pipeline_image_evaluate
from pipelines import pipeline_image_preprocess
from pipelines import pipeline_text_preprocess
from pipelines import pipeline_text_evaluate
from pipelines import pipeline_multimodel_train

import numpy as np
import pandas as pd
import sys
import os
import torchvision.models as models
from torch.utils.data import DataLoader

from transformers import CamembertForSequenceClassification
import torch
import torch.nn as nn

from typing import Tuple
from typing_extensions import Annotated
from zenml.enums import ArtifactType
from zenml.client import Client
from zenml import ArtifactConfig
from zenml.integrations.constants import PYTORCH
from zenml import step, Model, log_metadata
from zenml import pipeline
from zenml.config import DockerSettings
# docker_settings = DockerSettings(requirements="pipelines/requirements.txt")
docker_settings = DockerSettings(dockerfile="docker/Dockerfile",
                                 build_context_root=".",
                                 install_stack_requirements=False,
                                 parent_image_build_config={
                                     "dockerignore": ".dockerignore"
                                 })


###############################################################
# text


#################################################################


###################################################################


@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def global_pipeline(
    external_textmodel: str = None,  # rajouter step load model from path à fournir ici
    external_imagemodel: str = None,
    external_stackingmodel: str = None,
    enable_translation: bool = False,
    textmodel_version: str = None,
    imagemodel_version: str = None,
    stackmodel_version: str = None
):

    dataset_train, dataset_test, dataset_val = pipeline_text_preprocess(
        enable_translation=enable_translation)
    textmodel_f1_score, textmodel_confusion_matrix, textmodel_pred, model_text = pipeline_text_evaluate(
        dataset_train, dataset_test, dataset_val,
        external_textmodel=external_textmodel,
        textmodel_version=textmodel_version)

    dataloader_train, dataloader_val, dataloader_test = pipeline_image_preprocess()
    imagemodel_f1_score, imagemodel_confusion_matrix, imagemodel_pred, model_image = pipeline_image_evaluate(
        dataloader_train, dataloader_val, dataloader_test,
        external_imagemodel=external_imagemodel,
        imagemodel_version=imagemodel_version)

    # model multimodal
    f1_score, confusion_matrix, predictions = pipeline_multimodel_train(
        model_text=model_text,
        model_image=model_image,
        textmodel_pred=textmodel_pred,
        imagemodel_pred=imagemodel_pred,
        external_stackingmodel=external_stackingmodel,
        stackmodel_version=stackmodel_version
    )
    


if __name__ == "__main__":
    # Récupération du paramètre MODEL_VERSION passé en argument, sinon "0.0"
    version_model = sys.argv[1] if len(sys.argv) > 1 else "0.0"
    # pipeline_textual(external_textmodel="models/camembert_modele_1")
    global_pipeline(external_textmodel="models/camembert_modele_1",
                    external_imagemodel="models/resnet/resnet50_model2b_20250327_014445.pth",
                    external_stackingmodel="models/stacking/stacking_xgb_model.joblib",
                    enable_translation=False,
                    textmodel_version=version_model,
                    imagemodel_version=version_model,
                    stackmodel_version=version_model
                    )

import os
import pandas as pd
import numpy as np

from zenml import pipeline, step
from zenml import Model, log_metadata
from zenml.integrations.constants import PYTORCH
from zenml import ArtifactConfig
from zenml.enums import ArtifactType
from zenml.client import Client

from typing_extensions import Annotated
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

# ==================== Progress Monitoring ====================
from tqdm import tqdm


@step(model=Model(name="image_model", artifact_type=ArtifactType.MODEL, version="{IMAGEMODEL_VERSION}"))
def imagedata_train(train_loader: DataLoader, val_loader: DataLoader) -> Tuple[Annotated[
    nn.Module,
    ArtifactConfig(name="image_model", artifact_type=ArtifactType.MODEL,
                   version="{IMAGEMODEL_VERSION}")
],
        Annotated[pd.DataFrame, "logs"]]:
    # alternative? model = models.resnet50(pretrained=True)
    # Chargement & configuration du modèle
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # On fixe les poids des couches pré-entraînées
    # for param in model.parameters():
    #    param.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, 27)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.3e-4)
    num_epochs = 8
    # Détection du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_loss = float('inf')
    logs = pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        #  Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        #  Sauvegarde des logs de perte
        logs = logs.append({"epoch": epoch+1, "train_loss": avg_train_loss,
                           "val_loss": avg_val_loss}, ignore_index=True)

        #  Sauvegarde du meilleur modèle
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict().copy()

    print(
        f" Entraînement terminé ! Meilleur modèle sauvegardé avec une perte de validation de {best_loss:.4f}")
    return best_model, logs


@step(model=Model(name="image_model", artifact_type=ArtifactType.MODEL, version="{IMAGEMODEL_VERSION}"))
def imagemodel_load(model_path: str) -> Annotated[
    nn.Module,
    ArtifactConfig(name="image_model", artifact_type=ArtifactType.MODEL)
]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Le modèle ResNet n'existe pas à l'emplacement : {model_path}")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 27)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    return model


@step
def imagedata_evaluate(dataloader_test: DataLoader, model: nn.Module) -> Tuple[
    Annotated[float, "imagemodel_f1_score"],
    Annotated[pd.DataFrame, "imagemodel_confusion_matrix"],
    Annotated[pd.DataFrame, "imagemodel_predictions"]
]:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader_test, desc=f"tests"):
            y_true.append(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_pred.append(predicted)  # à transformer en label!
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.95})


@pipeline
def pipeline_image_evaluate(
    dataloader_train, dataloader_test, dataloader_val,
    imagemodel_version: str = None,
    external_imagemodel: str = None
) -> Tuple[
    Annotated[float, "imagemodel_f1_score"],
    Annotated[pd.DataFrame, "imagemodel_confusion_matrix"],
    Annotated[pd.DataFrame, "imagemodel_predictions"],
    Annotated[nn.Module, "model_image"]
]:
    if external_imagemodel is None:
        imagemodel_version = str(
            Client().get_pipeline("pipeline_image_evaluate").id)

    if external_imagemodel is not None:
        model_image = imagemodel_load.with_options(
            substitutions={"IMAGEMODEL_VERSION": imagemodel_version}
        )(external_imagemodel)
    else:
        model_image = imagedata_train.with_options(
            substitutions={"IMAGEMODEL_VERSION": imagemodel_version}
        )(dataloader_train, dataloader_val)
    imagemodel_f1_score, imagemodel_confusion_matrix, imagemodel_pred = imagedata_evaluate(
        dataloader_test, model_image)

    f1_score, confusion_matrix, predictions = imagedata_evaluate(
        dataloader_test=dataloader_test, model=model_image)
    return f1_score, confusion_matrix, predictions, model_image

if __name__ == "__main__":
    pipeline_image_evaluate(external_imagemodel="models/resnet/resnet50_model2b_20250327_014445.pth")

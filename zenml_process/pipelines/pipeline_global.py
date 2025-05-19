from zenml import pipeline
from zenml.config import DockerSettings
#docker_settings = DockerSettings(requirements="pipelines/requirements.txt")
docker_settings = DockerSettings(dockerfile="docker/Dockerfile", 
                                 build_context_root=".", 
                                 install_stack_requirements=False,
                                 parent_image_build_config={
                                     "dockerignore": ".dockerignore"
                                 })


from zenml import step, Model, log_metadata
from zenml.integrations.constants import PYTORCH
from zenml import ArtifactConfig
from zenml.client import Client
from zenml.enums import ArtifactType
from typing_extensions import Annotated
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
# ==================== Progress Monitoring ====================
from tqdm import tqdm
# ==================== Image Processing ====================
from PIL import Image, ImageEnhance


import sys
import pandas as pd
import numpy as np
import os
from transformers import CamembertForSequenceClassification



###############################################################
# image
# Prétraitement des images
def preprocess_image(image):
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)  # Netteté
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)  # Luminosité
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)  # Saturation

    img_array = np.array(image).astype(np.float32) # Conversion en float pour opérations mathématiques
    img_array[:, :, 0] *= 0.9   # Réduction du rouge
    img_array[:, :, 1] *= 1.05  # Augmentation du vert
    img_array[:, :, 2] *= 1.05  # Augmentation du bleu
    img_array = np.clip(img_array, 0, 255)

# Transformation des images pour ResNet50 
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformation des images avec Data Augmentation ----
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#  Définition du Dataset 
class ProductImageDataset(Dataset):
    def __init__(self, labels_dict, image_folder, transform=None, use_custom_preprocess=False):
        self.labels_dict = labels_dict
        self.image_folder = image_folder
        self.image_files = list(labels_dict.keys())
        self.transform = transform
        self.use_custom_preprocess = use_custom_preprocess 
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f" Image non trouvée : {img_path}")
            return torch.zeros((3, 224, 224)), -1

        if self.use_custom_preprocess:
            image = preprocess_image(image)
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels_dict[img_name], dtype=torch.long)

        return image, label


@step(enable_cache=True)
def imagedata_load(frac = 0.01, random_state = 42) -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_train"],
    Annotated[pd.DataFrame, "raw_dataset_test"],
    Annotated[pd.DataFrame, "raw_dataset_val"]
]:
    # Définition des chemins
    train_image_folder = r"images/train"
    val_image_folder = r"images/val"
    test_image_folder = r"images/test"

    train_file = "./datasets/train.csv"
    val_file = "./datasets/val.csv"
    test_file = "./datasets/test.csv"

    # Chargement des fichiers CSV 
    df_train = pd.read_csv(train_file).sample(frac= frac, random_state=random_state)
    df_val = pd.read_csv(val_file).sample(frac= frac, random_state=random_state)
    df_test = pd.read_csv(test_file).sample(frac= frac, random_state=random_state)

    # Création des chemins d’images pour chaque ligne
    df_train["image_path"] = df_train.apply(lambda row: os.path.join(train_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)
    df_val["image_path"] = df_val.apply(lambda row: os.path.join(val_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)
    df_test["image_path"] = df_test.apply(lambda row: os.path.join(test_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)

    # Extraction du nom de fichier image
    df_train["image_name"] = df_train["image_path"].apply(lambda x: os.path.basename(x))
    df_val["image_name"] = df_val["image_path"].apply(lambda x: os.path.basename(x))
    df_test["image_name"] = df_test["image_path"].apply(lambda x: os.path.basename(x))

    # Création d’un mapping des labels
    unique_labels = sorted(set(df_train["prdtypecode"].unique()).union(set(df_val["prdtypecode"].unique())).union(set(df_test["prdtypecode"].unique())))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Remapping des labels pour chaque image
    train_labels_dict = {img: label_map[label] for img, label in zip(df_train["image_name"], df_train["prdtypecode"])}
    val_labels_dict = {img: label_map[label] for img, label in zip(df_val["image_name"], df_val["prdtypecode"])}
    test_labels_dict = {img: label_map[label] for img, label in zip(df_test["image_name"], df_test["prdtypecode"])}

    # on retire toutes les images qui n'existent pas
    train_labels_dict = {img: label for img, label in train_labels_dict.items() if os.path.exists(os.path.join(train_image_folder, img))}
    val_labels_dict = {img: label for img, label in val_labels_dict.items() if os.path.exists(os.path.join(val_image_folder, img))}
    test_labels_dict = {img: label for img, label in test_labels_dict.items() if os.path.exists(os.path.join(test_image_folder, img))}

    return pd.DataFrame(train_labels_dict.items(), columns = ['image', 'target']), pd.DataFrame(val_labels_dict.items(), columns = ['image', 'target']), pd.DataFrame(test_labels_dict.items(), columns = ['image', 'target'])

@step
def imagedata_preprocessing_train(raw_dataset_train: pd.DataFrame, image_folder: str) -> Annotated[DataLoader, "train_dataloader"]:
    labels_dict = {img: label for img, label in zip(raw_dataset_train["image"], raw_dataset_train["target"])}
    #dataset = ProductImageDataset(labels_dict=labels_dict, image_folder=image_folder, transform=train_transform, use_custom_preprocess=True)
    dataset = ProductImageDataset(labels_dict=labels_dict, image_folder=image_folder, transform=train_transform, use_custom_preprocess=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    assert len(labels_dict) > 0, "Le dictionnaire des labels est vide. Vérifiez les données d'entrée."
    assert len(dataset) > 0, "Le dataset est vide. Vérifiez les images et les labels."
    print(f"Nombre d'éléments dans le DataLoader : {len(loader)}")
    return loader

@step
def imagedata_preprocessing(raw_dataset: pd.DataFrame, image_folder: str, pin_memory: bool) -> Annotated[DataLoader, "dataloader_{dataset_type}"]:
    labels_dict = {img: label for img, label in zip(raw_dataset["image"], raw_dataset["target"])}
    dataset = ProductImageDataset(labels_dict=labels_dict, image_folder=image_folder, transform=transform, use_custom_preprocess=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return loader

@step(model=Model(name="image_model", artifact_type=ArtifactType.MODEL, version="{IMAGEMODEL_VERSION}"))
def imagedata_train(train_loader: DataLoader, val_loader: DataLoader) -> Tuple[ Annotated[
    nn.Module, 
    ArtifactConfig(name="image_model", artifact_type=ArtifactType.MODEL, version="{IMAGEMODEL_VERSION}")
    ], 
                                                                               Annotated[pd.DataFrame, "logs"]]:
    # alternative? model = models.resnet50(pretrained=True)
    # Chargement & configuration du modèle
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # On fixe les poids des couches pré-entraînées
    #for param in model.parameters():
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

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        #  Sauvegarde des logs de perte
        logs = logs.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}, ignore_index=True)

        #  Sauvegarde du meilleur modèle
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model.state_dict().copy()

    print(f" Entraînement terminé ! Meilleur modèle sauvegardé avec une perte de validation de {best_loss:.4f}")
    return best_model, logs

@step(model=Model(name="image_model", artifact_type=ArtifactType.MODEL, version="{IMAGEMODEL_VERSION}"))
def imagemodel_load(model_path: str) -> Annotated[
    nn.Module, 
    ArtifactConfig(name="image_model", artifact_type=ArtifactType.MODEL)
    ]:
    if not os.path.exists(model_path):  
        raise FileNotFoundError(f"Le modèle ResNet n'existe pas à l'emplacement : {model_path}")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 27)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
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
            y_pred.append(predicted) # à transformer en label!
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.95})



###############################################################
# text

@step
def textdata_loader() -> Tuple[
    Annotated[pd.DataFrame, "raw_dataset_train"],
    Annotated[pd.DataFrame, "raw_dataset_test"],
    Annotated[pd.DataFrame, "raw_dataset_val"]
]:
    train = pd.DataFrame()
    test = pd.DataFrame()
    val = pd.DataFrame()
    return train, test, val

@step
def textdata_translation(dataset: pd.DataFrame) -> Annotated[pd.DataFrame, "translated_dataset"]:
    return dataset

@step
def textdata_preprocessing(raw_dataset_train: pd.DataFrame,
                          raw_dataset_test: pd.DataFrame,
                          raw_dataset_val: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "preprocessed_dataset_train"],
    Annotated[pd.DataFrame, "preprocessed_dataset_test"],
    Annotated[pd.DataFrame, "preprocessed_dataset_val"]
]:
    train = raw_dataset_train
    test = raw_dataset_test
    val = raw_dataset_val
    return train, test, val

# bug zenml 0.82: l'interface graphique ne reconnait pas le nom du modèls si on annote pas le résultat de la fonction
# bug zenml 0.82: l'interface graphique ne reconnait pas les métadonnées du modèles si on annote le résultat de la fonction
@step(model=Model(name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}"))
def textdata_train(dataset_train: pd.DataFrame, 
                   dataset_val: pd.DataFrame) -> Annotated[
                       CamembertForSequenceClassification, 
                       ArtifactConfig(name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}")
                       ]:
    log_metadata(metadata={"train_loss": 0.95, "val_loss": 0.92}, infer_model=True)
    return CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=27)

@step(model=Model(name="text_model", artifact_type=ArtifactType.MODEL, version="{TEXTMODEL_VERSION}"))
def textdat_model_load(model_path: str) -> Annotated[CamembertForSequenceClassification, ArtifactConfig(name="text_model", artifact_type=ArtifactType.MODEL)]:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle Camembert n'existe pas à l'emplacement : {model_path}")
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    return model


@step
def textdata_evaluate(dataset_test: pd.DataFrame, model: CamembertForSequenceClassification) -> Tuple[
    Annotated[float, "textmodel_f1_score"],  
    Annotated[pd.DataFrame, "textmodel_confusion_matrix"],
    Annotated[pd.DataFrame, "textmodel_predictions"]
]:
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.75})

#################################################################
from joblib import load
from sklearn.ensemble import StackingClassifier

### A ajouter: les proba prédictions de chaque modèle pour train et test
@step(model=Model(name="stacking_model", artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}"))
def multimodal_model_train(model_text: CamembertForSequenceClassification, model_image: nn.Module) -> Annotated[
    StackingClassifier,
    ArtifactConfig(name="stacking_model", artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}")
]:
    model = StackingClassifier()
    return model

@step(model=Model(name="stacking_model", artifact_type=ArtifactType.MODEL, version="{STACKMODEL_VERSION}"))
def multimodal_model_load(stacking_model_path : str) -> Annotated[
    StackingClassifier,
    ArtifactConfig(name="stacking_model", artifact_type=ArtifactType.MODEL)
]:
    return load(os.path.join("models/stacking","stacking_xgb_model.joblib"))

@step
def multimodel_model_evaluate(predict_proba_text: pd.DataFrame, predict_proba_image: pd.DataFrame, model: StackingClassifier) -> Tuple[
    Annotated[float, "f1_score"],  
    Annotated[pd.DataFrame, "confusion_matrix"],
    Annotated[pd.DataFrame, "predictions"]
]:
    return 0.88, pd.DataFrame(), pd.DataFrame({"filename": ['monfichier.png'], "y_true": 5, "y_pred": 5, "y_pred_proba": 0.98})

###################################################################
@pipeline(settings={"docker": docker_settings}, enable_cache=False)
def global_pipeline(
    external_textmodel: str = None, # rajouter step load model from path à fournir ici
    external_imagemodel: str = None,
    external_stackingmodel: str = None,
    enable_translation: bool = False,
    textmodel_version: str = None,
    imagemodel_version: str = None,
    stackmodel_version: str = None
):
    # Gestion versions: si pas de version fournie, on utilise le pipeline_id (on pourrait utiliser le commit git)
    if textmodel_version is None:
        textmodel_version = str(Client().get_pipeline("global_pipeline").id)
    if imagemodel_version is None:
        imagemodel_version = str(Client().get_pipeline("global_pipeline").id)
    if stackmodel_version is None:
        stackmodel_version = str(Client().get_pipeline("global_pipeline").id)

    raw_dataset_train, raw_dataset_test, raw_dataset_val = textdata_loader()
    if enable_translation:
        raw_dataset_train = textdata_translation(raw_dataset_train)
        raw_dataset_test = textdata_translation(raw_dataset_test)
        raw_dataset_val = textdata_translation(raw_dataset_val)
    dataset_train, dataset_test, dataset_val = textdata_preprocessing(raw_dataset_train, raw_dataset_test, raw_dataset_val)
    if external_textmodel is not None:
        model_text = textdat_model_load.with_options(
            substitutions={"TEXTMODEL_VERSION": textmodel_version}
        )(external_textmodel)
    else:
        model_text = textdata_train(dataset_train, dataset_val)
    textmodel_f1_score, textmodel_confusion_matrix, textmodel_pred = textdata_evaluate.with_options(
            substitutions={"TEXTMODEL_VERSION": textmodel_version}
        )(dataset_test, model_text)

    raw_dataset_train, raw_dataset_test, raw_dataset_val = imagedata_load()
    dataloader_train = imagedata_preprocessing_train(raw_dataset_train, r"images/train")
    dataloader_test = imagedata_preprocessing.with_options(
        substitutions={"dataset_type": "test"}
    )(raw_dataset_test, r"images/test", pin_memory=True)
    dataloader_val = imagedata_preprocessing.with_options(
        substitutions={"dataset_type": "val"}
    )(raw_dataset_val, r"images/val", pin_memory=False)
    if external_imagemodel is not None:
        model_image = imagemodel_load.with_options(
            substitutions={"IMAGEMODEL_VERSION": imagemodel_version}
        )(external_imagemodel)
    else:
        model_image = imagedata_train.with_options(
            substitutions={"IMAGEMODEL_VERSION": imagemodel_version}
        )(dataloader_train, dataloader_val)
    imagemodel_f1_score, imagemodel_confusion_matrix, imagemodel_pred = imagedata_evaluate(dataloader_test, model_image)

    ## model multimodal
    if external_stackingmodel is not None:
        model_stack = multimodal_model_load.with_options(
            substitutions={"STACKMODEL_VERSION": stackmodel_version}
        )(external_imagemodel)
    else:
        model_stack = multimodal_model_train.with_options(
            substitutions={"STACKMODEL_VERSION": stackmodel_version}
        )(model_text, model_image)
    multimodel_model_evaluate(textmodel_pred, imagemodel_pred, model_stack)

if __name__ == "__main__":
    # Récupération du paramètre MODEL_VERSION passé en argument, sinon "0.0"
    version_model = sys.argv[1] if len(sys.argv) > 1 else "0.0"
    #pipeline_textual(external_textmodel="models/camembert_modele_1")
    global_pipeline(external_textmodel="models/camembert_modele_1",
                    external_imagemodel="models/resnet/resnet50_model2b_20250327_014445.pth",
                    external_stackingmodel="models/stacking/stacking_xgb_model.joblib",
                    enable_translation=False,
                    textmodel_version=version_model,
                    imagemodel_version=version_model,
                    stackmodel_version=version_model
                    )

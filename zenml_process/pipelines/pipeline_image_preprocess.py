import os
import pandas as pd
import numpy as np

from zenml import pipeline
from zenml import step
from zenml.client import Client
from zenml.integrations.constants import PYTORCH
from typing_extensions import Annotated
from typing import Tuple


import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==================== Image Processing ====================
from PIL import Image, ImageEnhance
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

    # Conversion en float pour opérations mathématiques
    img_array = np.array(image).astype(np.float32)
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
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.02),
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
def imagedata_load(frac=0.01, random_state=42) -> Tuple[
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
    df_train = pd.read_csv(train_file).sample(
        frac=frac, random_state=random_state)
    df_val = pd.read_csv(val_file).sample(frac=frac, random_state=random_state)
    df_test = pd.read_csv(test_file).sample(
        frac=frac, random_state=random_state)

    # Création des chemins d’images pour chaque ligne
    df_train["image_path"] = df_train.apply(lambda row: os.path.join(
        train_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)
    df_val["image_path"] = df_val.apply(lambda row: os.path.join(
        val_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)
    df_test["image_path"] = df_test.apply(lambda row: os.path.join(
        test_image_folder, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1)

    # Extraction du nom de fichier image
    df_train["image_name"] = df_train["image_path"].apply(
        lambda x: os.path.basename(x))
    df_val["image_name"] = df_val["image_path"].apply(
        lambda x: os.path.basename(x))
    df_test["image_name"] = df_test["image_path"].apply(
        lambda x: os.path.basename(x))

    # Création d’un mapping des labels
    unique_labels = sorted(set(df_train["prdtypecode"].unique()).union(
        set(df_val["prdtypecode"].unique())).union(set(df_test["prdtypecode"].unique())))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Remapping des labels pour chaque image
    train_labels_dict = {img: label_map[label] for img, label in zip(
        df_train["image_name"], df_train["prdtypecode"])}
    val_labels_dict = {img: label_map[label] for img, label in zip(
        df_val["image_name"], df_val["prdtypecode"])}
    test_labels_dict = {img: label_map[label] for img, label in zip(
        df_test["image_name"], df_test["prdtypecode"])}

    # on retire toutes les images qui n'existent pas
    train_labels_dict = {img: label for img, label in train_labels_dict.items(
    ) if os.path.exists(os.path.join(train_image_folder, img))}
    val_labels_dict = {img: label for img, label in val_labels_dict.items(
    ) if os.path.exists(os.path.join(val_image_folder, img))}
    test_labels_dict = {img: label for img, label in test_labels_dict.items(
    ) if os.path.exists(os.path.join(test_image_folder, img))}

    return pd.DataFrame(train_labels_dict.items(), columns=['image', 'target']), pd.DataFrame(val_labels_dict.items(), columns=['image', 'target']), pd.DataFrame(test_labels_dict.items(), columns=['image', 'target'])


@step
def imagedata_preprocessing_train(raw_dataset_train: pd.DataFrame, image_folder: str) -> Annotated[DataLoader, "train_dataloader"]:
    labels_dict = {img: label for img, label in zip(
        raw_dataset_train["image"], raw_dataset_train["target"])}
    # dataset = ProductImageDataset(labels_dict=labels_dict, image_folder=image_folder, transform=train_transform, use_custom_preprocess=True)
    dataset = ProductImageDataset(labels_dict=labels_dict, image_folder=image_folder,
                                  transform=train_transform, use_custom_preprocess=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=0, pin_memory=True)
    assert len(
        labels_dict) > 0, "Le dictionnaire des labels est vide. Vérifiez les données d'entrée."
    assert len(
        dataset) > 0, "Le dataset est vide. Vérifiez les images et les labels."
    print(f"Nombre d'éléments dans le DataLoader : {len(loader)}")
    return loader


@step
def imagedata_preprocessing(raw_dataset: pd.DataFrame, image_folder: str, pin_memory: bool) -> Annotated[DataLoader, "dataloader_{dataset_type}"]:
    labels_dict = {img: label for img, label in zip(
        raw_dataset["image"], raw_dataset["target"])}
    dataset = ProductImageDataset(
        labels_dict=labels_dict, image_folder=image_folder, transform=transform, use_custom_preprocess=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,
                        num_workers=0, pin_memory=pin_memory)
    return loader


@pipeline
def pipeline_image_preprocess(frac = 0.01, random_state = 42) -> Tuple[
    Annotated[DataLoader, "train_dataloader"],
    Annotated[DataLoader, "val_dataloader"],
    Annotated[DataLoader, "test_dataloader"]
]:
    raw_dataset_train, raw_dataset_test, raw_dataset_val= imagedata_load(frac=frac, random_state=random_state)
    dataloader_train = imagedata_preprocessing_train(raw_dataset_train, r"images/train")
    dataloader_test = imagedata_preprocessing.with_options(
        substitutions={"dataset_type": "test"}
    )(raw_dataset_test, r"images/test", pin_memory=True)
    dataloader_val = imagedata_preprocessing.with_options(
        substitutions={"dataset_type": "val"}
    )(raw_dataset_val, r"images/val", pin_memory=False)
    return dataloader_train, dataloader_val, dataloader_test

if __name__ == "__main__":
    pipeline_image_preprocess()

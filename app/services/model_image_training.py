import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from datetime import datetime
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from dagshub import dagshub_logger
import mlflow
import dagshub

from app.services.image_preprocessing import preprocess_image_from_pil
from app.utils.mlflow_utils import init_mlflow_if_enabled
import app.core.config as config


class ProductImageDataset(Dataset):
    def __init__(self, labels_dict, image_folder, transform=None, preprocess_fn=None):
        self.labels_dict = labels_dict
        self.image_folder = image_folder
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.image_files = list(labels_dict.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label = self.labels_dict[image_name]
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.preprocess_fn:
            image = self.preprocess_fn(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(predictions)
    return f1_score(y_true, y_pred, average="weighted")


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=3, logger=None):
    best_f1 = 0.0
    for epoch in range(num_epochs):
        print(f"\n🟢 Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_f1 = evaluate_model(model, val_loader, device)

        print(f"📉 Training Loss: {avg_loss:.4f}")
        print(f"🧪 Validation F1: {val_f1:.4f}")

        if logger:
            logger.log_metrics({"train_loss": avg_loss, "val_f1": val_f1}, step=epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1

    return model, best_f1


def run_retrain_resnet_model(num_epochs=3, batch_size=32):
    print("📦 Loading data and labels")
    df_train = pd.read_csv(config.DATAFRAME_DIR_TRAIN_1PCT)
    df_val = pd.read_csv(config.DATAFRAME_DIR_VALID_1PCT)

    df_train["image_path"] = df_train.apply(
        lambda row: os.path.join(config.DATASET_IMAGE_DIR_TRAIN_1PCT, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1
    )
    df_val["image_path"] = df_val.apply(
        lambda row: os.path.join(config.DATASET_IMAGE_DIR_VALID_1PCT, f"image_{row['imageid']}_product_{row['productid']}.jpg"), axis=1
    )

    df_train["image_name"] = df_train["image_path"].apply(os.path.basename)
    df_val["image_name"] = df_val["image_path"].apply(os.path.basename)

    labels = sorted(set(df_train["prdtypecode"]).union(df_val["prdtypecode"]))
    label_map = {label: idx for idx, label in enumerate(labels)}

    train_labels = {img: label_map[label] for img, label in zip(df_train["image_name"], df_train["prdtypecode"])}
    val_labels = {img: label_map[label] for img, label in zip(df_val["image_name"], df_val["prdtypecode"])}

    print("📂 Creating datasets and loaders")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ProductImageDataset(train_labels, config.DATASET_IMAGE_DIR_TRAIN_1PCT, transform, preprocess_image_from_pil)
    val_dataset = ProductImageDataset(val_labels, config.DATASET_IMAGE_DIR_VALID_1PCT, transform, preprocess_image_from_pil)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print("📥 Loading pretrained model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load(config.MODEL_DIR_RESNET, map_location=device))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    print("🔍 Evaluating previous model")
    old_f1 = evaluate_model(model, val_loader, device)
    print(f"📊 Previous model F1 score: {old_f1:.4f}")

    mlflow_enabled = init_mlflow_if_enabled(os.getenv("MLFLOW_EXPERIMENT_RETRAIN_RESNET"))
    if not mlflow_enabled:
        print("⚠️ MLflow désactivé → entraînement local uniquement.")

    print("🚀 Starting retraining")
    if mlflow_enabled:
        with mlflow.start_run():
            with dagshub_logger() as logger:
                model, new_f1 = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs, logger=logger)
    else:
        model, new_f1 = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=num_epochs)

    if new_f1 > old_f1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"resnet50_retrained_{timestamp}.pth"
        model_path_target = os.path.join(config.MODEL_DIR, model_name)
        torch.save(model.state_dict(), model_path_target)

        if mlflow_enabled:
            mlflow.log_metric("old_f1", old_f1)
            mlflow.log_metric("new_f1", new_f1)
            mlflow.log_params({
                "epochs": num_epochs,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "lr": 3e-4
            })
            mlflow.log_artifact(model_path_target, artifact_path="resnet_model")

        print(f"✅ Nouveau modèle sauvegardé (F1: {new_f1:.4f} > {old_f1:.4f})")
        return {"model_saved": True, "model_path": model_path_target, "old_f1": old_f1, "new_f1": new_f1}
    else:
        print(f"❌ Nouveau modèle ignoré (F1: {new_f1:.4f} <= {old_f1:.4f})")
        return {"model_saved": False, "model_path": None, "old_f1": old_f1, "new_f1": new_f1}
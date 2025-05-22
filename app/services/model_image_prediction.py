# === IMPORTS DES LIBRAIRIES ===
import torch
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import models

from sklearn.metrics import f1_score
import numpy as np
import os
from tqdm import tqdm

import app.core.config as config 
from app.services.image_preprocessing import preprocess_image_from_pil

# === DÉTECTION DE L'APPAREIL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

# === DICTIONNAIRE DES CATÉGORIES RAKUTEN ===
labelcat = {
    10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console",
    60: "Console de jeu", 1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau",
    1280: "Jouet enfant, déguisement", 1281: "Jeu de société", 1300: "Jouet tech",
    1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtements", 1320: "Autour du bébé",
    1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine", 2060: "Décoration intérieure",
    2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
    2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur",
    2583: "Autour de la piscine", 2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC"
}
all_labels = list(labelcat.keys())
all_label_names = [labelcat[k] for k in all_labels]

# === UTILITAIRE : ID → Nom de catégorie ===
def label_to_string(label_id):
    return labelcat.get(label_id, f"Catégorie inconnue ({label_id})")

# === PRÉTRAITEMENT D’IMAGE ===
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")

    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.5)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.2)

    img_array = np.array(image).astype(np.float32)
    img_array[:, :, 0] *= 0.9
    img_array[:, :, 1] *= 1.05
    img_array[:, :, 2] *= 1.05
    image = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# === CHARGEMENT DU MODÈLE RESNET ===
def load_image_model(num_classes):
    model_path = config.MODEL_DIR_RESNET
    if not os.path.exists(model_path):  
        raise FileNotFoundError(f"Le modèle ResNet n'existe pas à l'emplacement : {model_path}")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_resnet_model(path):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 27)  # 🔁 À adapter selon ton nombre de classes
    model.load_state_dict(torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    return model


# === PRÉDICTION UNIQUE SUR IMAGE ===
def predict(model, image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        logits = model(image_tensor)
    return torch.softmax(logits, dim=1).squeeze().cpu().numpy()

def predict_image(image_path):
    # Dictionnaire des labels → noms de catégorie
    categorie = {
        10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console", 60: "Console de jeu",
        1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau", 1280: "Jouet enfant, déguisement",
        1281: "Jeu de société", 1300: "Jouet tech", 1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtement",
        1320: "Autour du bébé", 1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine",
        2060: "Décoration intérieure", 2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
        2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur", 2583: "Autour de la piscine",
        2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC",
    }

    model = load_image_model(num_classes=27)
    image_tensor = preprocess_image(image_path)
    proba = predict(model, image_tensor)

    pred_index = int(np.argmax(proba))
    predicted_label = int(all_labels[pred_index])
    confidence_score = float(proba[pred_index])
    label_name = categorie.get(predicted_label, "Catégorie inconnue")

    return predicted_label, label_name, confidence_score

# === PRÉDICTION EN LOT (DATAFRAME) ===
def predict_dataframe(df, imageid_col="imageid", productid_col="productid",
                      true_label_col="prdtypecode", image_dir="data/images/test",
                      text_col=None, sample_size=50):

    model = load_image_model(num_classes=27)
    preds, y_true, confidences = [], [], []

    # === Limitation stricte du nombre d'images
    subset_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    for _, row in subset_df.iterrows():
        image_path = os.path.join(image_dir, f"image_{row[imageid_col]}_product_{row[productid_col]}.jpg")
        if not os.path.exists(image_path):
            preds.append(None)
            y_true.append(None)
            confidences.append(None)
            continue

        try:
            # Ouvrir l'image avec PIL
            image = Image.open(image_path).convert("RGB")

            # Prétraitement avec ta fonction
            processed_pil_image = preprocess_image_from_pil(image)

            # Conversion en tenseur PyTorch prêt pour le modèle
            transform_final = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform_final(processed_pil_image).unsqueeze(0)

            # Prédiction
            proba = predict(model, image_tensor)
            pred_index = np.argmax(proba)
            pred_label = all_labels[pred_index]
            confidence_score = float(proba[pred_index])

            preds.append(pred_label)
            confidences.append(confidence_score)
            y_true.append(row.get(true_label_col, None))
        except Exception:
            preds.append(None)
            confidences.append(None)
            y_true.append(None)

    subset_df["predicted_label"] = preds
    subset_df["predicted_category"] = subset_df["predicted_label"].apply(label_to_string)
    subset_df["confidence_score"] = confidences
    subset_df = subset_df.fillna("N/A")

    return subset_df

def evaluate_image_model_on_dataset(df, image_dir, sample_size=100):
    model = load_image_model(num_classes=27)
    
    df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    preds, y_true, confidences = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Évaluation Images"):
        image_path = os.path.join(image_dir, f"image_{row['imageid']}_product_{row['productid']}.jpg")
        if not os.path.exists(image_path):
            continue
        try:
            image_tensor = preprocess_image(image_path)
            proba = predict(model, image_tensor)
            pred_label = all_labels[np.argmax(proba)]
            conf = float(np.max(proba))

            preds.append(pred_label)
            y_true.append(row["prdtypecode"])
            confidences.append(conf)
        except Exception as e:
            preds.append(None)
            y_true.append(None)
            confidences.append(0.0)

    df["predicted_label"] = preds
    df["true_label"] = y_true
    df["confidence_score"] = confidences

    # Calculs globaux
    valid_rows = df.dropna(subset=["predicted_label", "true_label"])
    f1 = f1_score(valid_rows["true_label"], valid_rows["predicted_label"], average="weighted")
    avg_conf = valid_rows["confidence_score"].mean()

    return {
        "average_confidence_score": round(avg_conf, 2),
        "weighted_f1_score": round(f1, 2),
        "evaluated_samples": len(valid_rows)
    }
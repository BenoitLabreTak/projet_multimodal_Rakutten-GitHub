import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import re
import os
import html
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
#from googletrans import Translator

import app.core.config as config 
from app.services.text_preprocessing import preprocess_txt

from sklearn.metrics import log_loss
from sklearn.metrics import f1_score


# === DÉTECTION DE L'APPAREIL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === DICTIONNAIRE DES CATÉGORIES RAKUTEN ===

labelcat = {
    10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console",
    60: "Console de jeu", 1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau",
    1280: "Jouet enfant, déguisement", 1281: "Jeu de société", 1300: "Jouet tech",
    1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtement", 1320: "Autour du bébé",
    1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine", 2060: "Décoration intérieure",
    2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
    2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur",
    2583: "Autour de la piscine", 2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC"
}
sorted_labels = sorted(labelcat.keys())
index_to_label = {i: sorted_labels[i] for i in range(len(sorted_labels))}

# === UTILS PRÉTRAITEMENT ===
def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = html.unescape(text)
    text = re.sub(r"[^a-zA-Z0-9\sàâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

#google_translator = Translator()

#def maybe_translate_to_french(text):
#    try:
#        if not isinstance(text, str) or text.strip() == "":
#            return ""
#        detected = google_translator.detect(text).lang
#        if detected != "fr":
#            translated = google_translator.translate(text, src=detected, dest="fr")
#            return translated.text
#        return text
#    except Exception as e:
#        return text

# === CHARGEMENT DU MODÈLE CAMEMBERT ===
def load_text_model():
    model_path = config.MODEL_DIR_CAMEMBERT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle Camembert n'existe pas à l'emplacement : {model_path}")
    model = CamembertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = CamembertTokenizer.from_pretrained(model_path)
    model.eval()
    return model, tokenizer



# === PRÉDICTION MANUELLE ===
def predict_text_model(designation, description):
    model, tokenizer = load_text_model()

    categorie = {
        10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console", 60: "Console de jeu",
        1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau", 1280: "Jouet enfant, déguisement",
        1281: "Jeu de société", 1300: "Jouet tech", 1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtement",
        1320: "Autour du bébé", 1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine",
        2060: "Décoration intérieure", 2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
        2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur", 2583: "Autour de la piscine",
        2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC",
    }

    #raw_text = f"{designation} {description}"
    #text = preprocess_txt(raw_text)
    df_tmp = pd.DataFrame([{"designation": designation, "description": description}])
    text = preprocess_txt(df_tmp).iloc[0]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0)

        pred_index = int(np.argmax(probs))
        predicted_label = int(index_to_label[pred_index])
        confidence_score = float(probs[pred_index])
        label_name = categorie.get(predicted_label, "Catégorie inconnue")

    return predicted_label, label_name, confidence_score


def safe_probs(p):
    return [float(v) if np.isfinite(v) else 0.0 for v in p]


def predict_text_model_on_dataset(df=None, sample_size=50):
    df = df.sample(n=min(sample_size, len(df)), random_state=42)

    df["designation"] = df["designation"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # ⬇️ Updated: no longer expects a 3-tuple with probs, but (label, label_name, confidence_score)
    df["prediction_raw"] = df.apply(
        lambda row: predict_text_model(row["designation"], row["description"]), axis=1
    )

    df["predicted_label"] = df["prediction_raw"].apply(lambda x: x[0])
    df["label_name"] = df["prediction_raw"].apply(lambda x: x[1])
    df["confidence_score"] = df["prediction_raw"].apply(lambda x: x[2])

    return df


def evaluate_text_model(designation, description, true_label):
    model, tokenizer = load_text_model()

    categorie = {
        10: "Livre occasion", 40: "Jeu vidéo, accessoire tech.", 50: "Accessoire Console", 60: "Console de jeu",
        1140: "Figurine", 1160: "Carte Collection", 1180: "Jeu Plateau", 1280: "Jouet enfant, déguisement",
        1281: "Jeu de société", 1300: "Jouet tech", 1301: "Paire de chaussettes", 1302: "Jeu extérieur, vêtement",
        1320: "Autour du bébé", 1560: "Mobilier intérieur", 1920: "Chambre", 1940: "Cuisine",
        2060: "Décoration intérieure", 2220: "Animal", 2280: "Revues et journaux", 2403: "Magazines, livres et BDs",
        2462: "Jeu occasion", 2522: "Bureautique et papeterie", 2582: "Mobilier extérieur", 2583: "Autour de la piscine",
        2585: "Bricolage", 2705: "Livre neuf", 2905: "Jeu PC",
    }

    #raw_text = f"{designation} {description}"
    df_tmp = pd.DataFrame([{"designation": designation, "description": description}])
    text = preprocess_txt(df_tmp).iloc[0]
    #text = preprocess_txt(raw_text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0)

        pred_index = int(np.argmax(probs))
        predicted_label = int(index_to_label[pred_index])
        confidence_score = float(probs[pred_index])
        label_name = categorie.get(predicted_label, "Catégorie inconnue")

        # F1-score computation: true and predicted need to be iterable
        y_true = [true_label]
        y_pred = [predicted_label]
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return predicted_label, label_name, confidence_score, f1

def evaluate_text_model_on_dataset(df, sample_size=50):
    df = df.sample(n=min(sample_size, len(df)), random_state=42)

    df["designation"] = df["designation"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["prdtypecode"] = df["prdtypecode"].fillna(-1).astype(int)  # true labels

    df["evaluation"] = df.apply(
        lambda row: evaluate_text_model(row["designation"], row["description"], row["prdtypecode"]), axis=1
    )

    df["predicted_label"] = df["evaluation"].apply(lambda x: x[0])
    df["label_name"] = df["evaluation"].apply(lambda x: x[1])
    df["confidence_score"] = df["evaluation"].apply(lambda x: float(x[2]))
    df["f1_score"] = df["evaluation"].apply(lambda x: float(x[3]))

    return df
import sys
from pathlib import Path
import pandas as pd


# ==================== Standard Libraries ====================
import os
import json
import re
import pickle
from itertools import product


# ==================== Data Manipulation ====================
import numpy as np
import pandas as pd

# ==================== Visualization ====================
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== Scikit-learn ====================
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import evaluate

# ==================== Hugging Face Transformers ====================
from datasets import (Dataset, DatasetDict)
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

import torch
import torch.nn as nn

from transformers import CamembertTokenizer, CamembertForSequenceClassification
# ==================== TensorFlow / Keras ====================

#from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Embedding, Reshape, Conv2D, MaxPooling2D, Flatten,
    Dropout, Dense, Concatenate
)

# ================
from app.services.text_preprocessing import preprocess_txt
import app.core.config as config 

ROOT_DIR = config.BASE_DIR 
# ================
# ================
# Charger les fichiers CSV en ne gardant que les colonnes nécessaires

def run_retraining_pipeline():
    # Charger les fichiers CSV en ne gardant que les colonnes nécessaires
    X_train = pd.read_csv(os.path.join(ROOT_DIR,"data/retraining_text/train_1pct.csv"))
    X_val = pd.read_csv(os.path.join(ROOT_DIR,"data/retraining_text/val_1pct.csv"))
    X_test = pd.read_csv(os.path.join(ROOT_DIR,"data/retraining_text/test_1pct.csv"))

    # Prétraitement du texte
    X_train['txt_fr'] = preprocess_txt(X_train)
    X_val['txt_fr'] = preprocess_txt(X_val)
    X_test['txt_fr'] = preprocess_txt(X_test)

    # Suppression des colonnes inutiles
    X_train = X_train[["txt_fr", "prdtypecode","productid"]]
    X_val = X_val[["txt_fr", "prdtypecode","productid"]]
    X_test = X_test[["txt_fr", "prdtypecode","productid"]]

    
    labelcat = {
        10 : "Livre occasion",
        40 : "Jeu vidéo, accessoire tech.",
        50 : "Accessoire Console",
        60 : "Console de jeu",
        1140 : "Figurine",
        1160 : "Carte Collection",
        1180 : "Jeu Plateau",
        1280 : "Jouet enfant, déguisement",
        1281 : "Jeu de société",
        1300 : "Jouet tech",
        1301 : "Paire de chaussettes",
        1302 : "Jeu extérieur, vêtement",
        1320 : "Autour du bébé",
        1560 : "Mobilier intérieur",
        1920 : "Chambre",
        1940 : "Cuisine",
        2060 : "Décoration intérieure",
        2220 : "Animal",
        2280 : "Revues et journaux",
        2403 : "Magazines, livres et BDs",
        2462 : "Jeu occasion",
        2522 : "Bureautique et papeterie",
        2582 : "Mobilier extérieur",
        2583 : "Autour de la piscine",
        2585 : "Bricolage",
        2705 : "Livre neuf",
        2905 : "Jeu PC",
    }
    # Création d'une nouvelle colonne pour le nom de la catégorie
    X_train["category_name"] = X_train["prdtypecode"].map(labelcat)
    X_val["category_name"] = X_val["prdtypecode"].map(labelcat)
    X_test["category_name"] = X_test["prdtypecode"].map(labelcat)


    # Création d'un mapping label => ID
    label_mapping = {int(label): i for i, label in enumerate(sorted(X_train["prdtypecode"].unique()))}
    inverse_label_mapping = {i: int(label) for label, i in label_mapping.items()}  # Convertir les labels en int

    # Transformation des labels
    X_train["label"] = X_train["prdtypecode"].map(label_mapping)
    X_val["label"] = X_val["prdtypecode"].map(label_mapping)
    X_test["label"] = X_test["prdtypecode"].map(label_mapping)


    # Conversion en Dataset Hugging Face
    train_dataset = Dataset.from_pandas(X_train[["label", "txt_fr", "productid", "prdtypecode"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(X_val[["label", "txt_fr", "productid", "prdtypecode"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(X_test[["label", "txt_fr", "productid", "prdtypecode"]], preserve_index=False)



    def load_or_initialize_model(model_path, num_labels, device="cpu"):
        if os.path.exists(model_path):
            print(f"📦 Chargement du modèle depuis : {model_path}")
            model = CamembertForSequenceClassification.from_pretrained(model_path).to(device)
            tokenizer = CamembertTokenizer.from_pretrained(model_path)

            model_num_labels = model.classifier.out_proj.out_features
            if model_num_labels != num_labels:
                print(f"⚠️ Nombre de classes différent ({model_num_labels} ➝ {num_labels}) → Réinitialisation")
                model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels).to(device)
                tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        else:
            print(f"🆕 Chargement de camembert-base avec {num_labels} classes")
            model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels).to(device)
            tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

        model.eval()
        return model, tokenizer

    def tokenize_function(examples, tokenizer):
        return tokenizer(examples["txt_fr"], padding="max_length", truncation=True, max_length=256)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = config.MODEL_DIR_CAMEMBERT
    model, tokenizer = load_or_initialize_model(model_path, num_labels=27, device=device)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "productid", "prdtypecode"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "productid", "prdtypecode"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "productid", "prdtypecode"])

    print("✅ Tokenisation terminée !")


    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
            "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        }


    def get_model_f1(model, val_dataset):
        """
        Évalue un modèle donné sur le jeu de validation et retourne le F1 score.
        """
        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics
        )
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        return metrics.get("eval_f1", 0.0)



    def retrain_model(train_dataset, 
                      val_dataset,
                      num_labels=27, 
                      epochs=3, 
                      device="cpu"):
        
        model_path = config.MODEL_DIR_CAMEMBERT
        model, tokenizer = load_or_initialize_model(model_path, num_labels=num_labels, device=device)
        model_path_target= os.path.join(config.MODEL_DIR, "camembert_retrained")

        print("🔍 Évaluation du modèle actuel...")
        old_f1 = get_model_f1(model, val_dataset)
        print(f"F1 du modèle actuel : {old_f1:.4f}")

        training_args = TrainingArguments(
            output_dir=model_path_target,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir="./logs",
            logging_strategy="epoch",
            logging_steps=50,
            report_to="none",
            dataloader_pin_memory=False
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        new_f1 = get_model_f1(model, val_dataset)

        if new_f1 > old_f1:
            trainer.save_model(model_path_target)
            print(f"✅ Nouveau modèle sauvegardé (F1: {new_f1:.4f} > {old_f1:.4f})")
        else:
            print(f"❌ Nouveau modèle ignoré (F1: {new_f1:.4f} <= {old_f1:.4f})")



    retrain_model(
        train_dataset,
        val_dataset,
        num_labels=27,
        epochs=3, 
        device=device
    )
    return "Retraining terminé. Modèle sauvegardé si F1 amélioré."
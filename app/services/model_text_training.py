# Standard Libraries
import os

# Data Manipulation
import pandas as pd

# Scikit-learn
from sklearn.metrics import f1_score

# Hugging Face Transformers
from datasets import Dataset
from transformers import CamembertTokenizer, CamembertForSequenceClassification, TrainingArguments, Trainer
import evaluate

# PyTorch
import torch

# Local Modules
from app.services.text_preprocessing import preprocess_txt
import app.core.config as config

ROOT_DIR = config.BASE_DIR




def run_retrain_camembert_model(num_epochs=3, batch_size=4):

    # Load the datasets
    X_train = pd.read_csv(config.DATAFRAME_DIR_TRAIN_1PCT)
    X_val = pd.read_csv(config.DATAFRAME_DIR_VALID_1PCT)
    X_test = pd.read_csv(config.DATAFRAME_DIR_TEST_1PCT)

    # Preprocess the text data
    X_train['txt_fr'] = preprocess_txt(X_train)
    X_val['txt_fr'] = preprocess_txt(X_val)
    X_test['txt_fr'] = preprocess_txt(X_test)

    X_train = X_train[["txt_fr", "prdtypecode", "productid"]]
    X_val = X_val[["txt_fr", "prdtypecode", "productid"]]
    X_test = X_test[["txt_fr", "prdtypecode", "productid"]]

    label_mapping = {int(label): i for i, label in enumerate(sorted(X_train["prdtypecode"].unique()))}
    X_train["label"] = X_train["prdtypecode"].map(label_mapping)
    X_val["label"] = X_val["prdtypecode"].map(label_mapping)
    X_test["label"] = X_test["prdtypecode"].map(label_mapping)

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
        trainer = Trainer(model=model, compute_metrics=compute_metrics)
        metrics = trainer.evaluate(eval_dataset=val_dataset)
        return metrics.get("eval_f1", 0.0)

    def retrain_model(train_dataset, val_dataset, num_labels=27, epochs=num_epochs, batch_size=batch_size, device="cpu"):
        model_path = config.MODEL_DIR_CAMEMBERT
        model, tokenizer = load_or_initialize_model(model_path, num_labels=num_labels, device=device)
        model_path_target = os.path.join(config.MODEL_DIR, "camembert_retrained")

        print("🔍 Évaluation du modèle actuel...")
        old_f1 = get_model_f1(model, val_dataset)
        print(f"📊 F1 du modèle actuel : {old_f1:.4f}")

        training_args = TrainingArguments(
            output_dir=model_path_target,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
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
            return {"model_saved": True, "model_path": model_path_target, "old_f1": old_f1, "new_f1": new_f1}
        else:
            print(f"❌ Nouveau modèle ignoré (F1: {new_f1:.4f} <= {old_f1:.4f})")
            return {"model_saved": False, "model_path": None, "old_f1": old_f1, "new_f1": new_f1}

    result = retrain_model(
        train_dataset,
        val_dataset,
        num_labels=27,
        epochs=3,
        device=device
    )
    return result

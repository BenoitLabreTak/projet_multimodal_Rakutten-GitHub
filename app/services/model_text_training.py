import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Standard Libraries
import os
import pandas as pd
from datetime import datetime

# ML & Evaluation
from sklearn.metrics import f1_score
import evaluate

# Hugging Face Transformers
from datasets import Dataset
from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainingArguments
)

# PyTorch
import torch

# Experiment tracking
import mlflow
from dagshub import dagshub_logger
import dagshub

# Local Modules
from app.services.text_preprocessing import preprocess_txt
from app.utils.mlflow_utils import init_mlflow_if_enabled
import app.core.config as config

# Load metrics globally so they're available in compute_metrics
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def load_model(model_path, num_labels, device):
    if os.path.exists(model_path):
        print(f"📦 Loading existing model from: {model_path}")
        model = CamembertForSequenceClassification.from_pretrained(model_path).to(device)
        tokenizer = CamembertTokenizer.from_pretrained(model_path)
        if model.classifier.out_proj.out_features != num_labels:
            print("🔄 Label count mismatch → reinitializing")
            model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels).to(device)
            tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    else:
        print("🆕 Loading camembert-base")
        model = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=num_labels).to(device)
        tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["txt_fr"], padding="max_length", truncation=True, max_length=256)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }


def get_model_f1(model, val_dataset):
    args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=16,
        report_to="none",
        logging_strategy="no",
    )
    trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
    return trainer.evaluate(eval_dataset=val_dataset).get("eval_f1", 0.0)

def retrain(train_ds, val_ds, num_labels, epochs, batch_size, device):
    model, tokenizer = load_model(config.MODEL_DIR_CAMEMBERT, num_labels=num_labels, device=device)
    model_path_target = os.path.join(config.MODEL_DIR, "camembert_retrained")

    mlflow_enabled = init_mlflow_if_enabled(os.getenv("MLFLOW_EXPERIMENT_RETRAIN_CAMEMBERT"))
    if not mlflow_enabled:
        print("⚠️ MLflow désactivé → entraînement local uniquement.")
        
    print("🔍 Evaluating current model")
    old_f1 = get_model_f1(model, val_ds)
    print(f"📊 Old model F1: {old_f1:.4f}")

    training_args = TrainingArguments(
        output_dir=model_path_target,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        logging_strategy="epoch",
        dataloader_pin_memory=False
    )

    if mlflow_enabled:
        with mlflow.start_run():
            with dagshub_logger() as logger:
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_ds,
                    eval_dataset=val_ds,
                    compute_metrics=compute_metrics,
                )
                trainer.train()
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    new_f1 = get_model_f1(model, val_ds)

    if new_f1 > old_f1:
        trainer.save_model(model_path_target)

        if mlflow_enabled:
            mlflow.log_metric("old_f1", old_f1)
            mlflow.log_metric("new_f1", new_f1)
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size
            })
            mlflow.log_artifacts(model_path_target, artifact_path="camembert_model")

        print(f"✅ New model saved (F1: {new_f1:.4f} > {old_f1:.4f})")
        return {"model_saved": True, "model_path": model_path_target, "old_f1": old_f1, "new_f1": new_f1}
    else:
        print(f"❌ Model not saved (F1: {new_f1:.4f} <= {old_f1:.4f})")
        return {"model_saved": False, "model_path": None, "old_f1": old_f1, "new_f1": new_f1}


def run_retrain_camembert_model(num_epochs=3, batch_size=4):
    print("📦 Loading CSV files")
    X_train = pd.read_csv(config.DATAFRAME_DIR_TRAIN_1PCT)
    X_val = pd.read_csv(config.DATAFRAME_DIR_VALID_1PCT)
    X_test = pd.read_csv(config.DATAFRAME_DIR_TEST_1PCT)

    print("🧼 Preprocessing text")
    X_train['txt_fr'] = preprocess_txt(X_train)
    X_val['txt_fr'] = preprocess_txt(X_val)
    X_test['txt_fr'] = preprocess_txt(X_test)

    for df in [X_train, X_val, X_test]:
        df.drop(columns=[col for col in df.columns if col not in ["txt_fr", "prdtypecode", "productid"]], inplace=True)

    label_mapping = {int(label): i for i, label in enumerate(sorted(X_train["prdtypecode"].unique()))}
    for df in [X_train, X_val, X_test]:
        df["label"] = df["prdtypecode"].map(label_mapping)

    train_dataset = Dataset.from_pandas(X_train, preserve_index=False)
    val_dataset = Dataset.from_pandas(X_val, preserve_index=False)
    test_dataset = Dataset.from_pandas(X_test, preserve_index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(config.MODEL_DIR_CAMEMBERT, num_labels=27, device=device)

    # Proper tokenization + format
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("✅ Tokenization done")

    return retrain(train_dataset, val_dataset, 27, num_epochs, batch_size, device)
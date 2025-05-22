import os
import mlflow
import dagshub
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# ===================
# Dataset CSV paths
# ===================
DATAFRAME_DIR = os.getenv("DATAFRAME_DIR", os.path.join(BASE_DIR, "data/text"))
DATAFRAME_DIR_TRAIN = os.getenv("DATAFRAME_DIR_TRAIN", os.path.join(DATAFRAME_DIR, "train.csv"))
DATAFRAME_DIR_TEST = os.getenv("DATAFRAME_DIR_TEST", os.path.join(DATAFRAME_DIR, "test.csv"))
DATAFRAME_DIR_VALID = os.getenv("DATAFRAME_DIR_VALID", os.path.join(DATAFRAME_DIR, "val.csv"))

# ===================
# Image folders
# ===================
IMAGE_DIR = os.getenv("IMAGE_DIR", os.path.join(BASE_DIR, "data/images"))
DATASET_IMAGE_DIR_TRAIN = os.getenv("DATASET_IMAGE_DIR_TRAIN", os.path.join(IMAGE_DIR, "train"))
DATASET_IMAGE_DIR_VALID = os.getenv("DATASET_IMAGE_DIR_VALID", os.path.join(IMAGE_DIR, "val"))
DATASET_IMAGE_DIR_TEST = os.getenv("DATASET_IMAGE_DIR_TEST", os.path.join(IMAGE_DIR, "test"))

# Retraining paths
DATAFRAME_DIR_1PCT = os.getenv("DATAFRAME_DIR_1PCT", os.path.join(BASE_DIR, "data/retraining_text"))
DATAFRAME_DIR_TRAIN_1PCT = os.getenv("DATAFRAME_DIR_TRAIN_1PCT", os.path.join(DATAFRAME_DIR_1PCT, "train_1pct.csv"))
DATAFRAME_DIR_TEST_1PCT = os.getenv("DATAFRAME_DIR_TEST_1PCT", os.path.join(DATAFRAME_DIR_1PCT, "test_1pct.csv"))
DATAFRAME_DIR_VALID_1PCT = os.getenv("DATAFRAME_DIR_VALID_1PCT", os.path.join(DATAFRAME_DIR_1PCT, "val_1pct.csv"))

IMAGE_DIR_1PCT = os.getenv("IMAGE_DIR_1PCT", os.path.join(BASE_DIR, "data/retraining_images"))
DATASET_IMAGE_DIR_TRAIN_1PCT = os.getenv("DATASET_IMAGE_DIR_TRAIN_1PCT", os.path.join(IMAGE_DIR_1PCT, "train"))
DATASET_IMAGE_DIR_VALID_1PCT = os.getenv("DATASET_IMAGE_DIR_VALID_1PCT", os.path.join(IMAGE_DIR_1PCT, "val"))
DATASET_IMAGE_DIR_TEST_1PCT = os.getenv("DATASET_IMAGE_DIR_TEST_1PCT", os.path.join(IMAGE_DIR_1PCT, "test"))

# ===================
# Model paths
# ===================
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "app/models"))
MODEL_DIR_CAMEMBERT = os.getenv("MODEL_DIR_CAMEMBERT", os.path.join(MODEL_DIR, "camembert_model"))
MODEL_DIR_RESNET = os.getenv("MODEL_DIR_RESNET", os.path.join(MODEL_DIR, "resnet50_model.pth"))

# ===================
# MLflow / DagsHub (optionnel)
# ===================
DAGSHUB_URI = os.getenv("MLFLOW_TRACKING_URI")  # ex: https://dagshub.com/user/repo.mlflow
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")  # ex: user/projet_xx
MLFLOW_EXPERIMENT_RETRAIN_CAMEMBERT = os.getenv("MLFLOW_EXPERIMENT_RETRAIN_CAMEMBERT")
MLFLOW_EXPERIMENT_RETRAIN_RESNET = os.getenv("MLFLOW_EXPERIMENT_RETRAIN_RESNET")


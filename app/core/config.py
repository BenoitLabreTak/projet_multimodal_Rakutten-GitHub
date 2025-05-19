import os
BASE_DIR = r"/Users/mehdimalhas/DataScientest/Datascientest-Rakuten/Projet MLOps/rakuten_mlops"

# Licence for translastion Google API
# The path to the Google Cloud service account key file
#PATH_GOOGLE_APPLICATION_CREDENTIALS = os.path.join(BASE_DIR, "licence", "datascientest-453113-a3f88e32e4a7.json")

# Dataset paths
# The path to the dataset directory
DATAFRAME_DIR = os.path.join(BASE_DIR, "data/text")
DATAFRAME_DIR_TRAIN = os.path.join(DATAFRAME_DIR, "train.csv")
DATAFRAME_DIR_TEST = os.path.join(DATAFRAME_DIR, "test.csv")
DATAFRAME_DIR_VALID = os.path.join(DATAFRAME_DIR, "val.csv")


# Images paths
# The path to the images directory
MEDIA_DIR = os.path.join(BASE_DIR, "media")
IMAGE_DIR = os.path.join(BASE_DIR, "data/images")
DATASET_IMAGE_DIR_TRAIN = os.path.join(IMAGE_DIR, "train")
DATASET_IMAGE_DIR_VALID = os.path.join(IMAGE_DIR, "val")
DATASET_IMAGE_DIR_TEST = os.path.join(IMAGE_DIR, "test")
DEFAULT_FOLDER = os.path.join(IMAGE_DIR, "test")

#Retraining paths
DATAFRAME_DIR_1PCT = os.path.join(BASE_DIR, "data/retraining_text")
DATAFRAME_DIR_TRAIN_1PCT = os.path.join(DATAFRAME_DIR_1PCT, "train_1pct.csv")
DATAFRAME_DIR_TEST_1PCT = os.path.join(DATAFRAME_DIR_1PCT, "test_1pct.csv")
DATAFRAME_DIR_VALID_1PCT = os.path.join(DATAFRAME_DIR_1PCT, "val_1pct.csv")

IMAGE_DIR_1PCT = os.path.join(BASE_DIR, "data/retraining_images")
DATASET_IMAGE_DIR_TRAIN_1PCT = os.path.join(IMAGE_DIR_1PCT, "train")
DATASET_IMAGE_DIR_VALID_1PCT = os.path.join(IMAGE_DIR_1PCT, "val")
DATASET_IMAGE_DIR_TEST_1PCT = os.path.join(IMAGE_DIR_1PCT, "test")
DEFAULT_FOLDER_1PCT = os.path.join(IMAGE_DIR_1PCT, "test")




# Presentation image paths
SVMMATRIX = os.path.join(MEDIA_DIR, "confusion matrix TFIDF SVM.png")
SVMREPORT = os.path.join(MEDIA_DIR, "classification report TFIDF SVM.png")

BERTLOSS = os.path.join(MEDIA_DIR, "loss_camembert.png")
BERTMATRIX = os.path.join(MEDIA_DIR, "confusion matrix_camembert.png")
BERTREPORT = os.path.join(MEDIA_DIR, "classification report_camembert.png")

RESNETLOSS = os.path.join(MEDIA_DIR, "loss_resnet.png")
RESNETMATRIX = os.path.join(MEDIA_DIR, "confusion matrix_resnet.png")
RESNETREPORT = os.path.join(MEDIA_DIR, "classification report_resnet.png")

FUSIONMATRIX = os.path.join(MEDIA_DIR, "confusion matrix_multimodal.png")
FUSIONREPORT = os.path.join(MEDIA_DIR, "classification report_multimodal.png")

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "app/models")
MODEL_DIR_CAMEMBERT = os.path.join(MODEL_DIR, "camembert_modele_1")
MODEL_DIR_RESNET = os.path.join(MODEL_DIR, "resnet50_model2b_20250327_014445.pth")
MODEL_DIR_STACKING_XGB = os.path.join(MODEL_DIR, "stacking_xgb_model.joblib")


# intermediate dataset saves path
DATASET_SAVE_DIR = os.path.join(BASE_DIR, "saves")
DF_TEST_TRANSLATED = os.path.join(DATAFRAME_DIR, "X_test_translated_BERT.csv")

DF_PREDICT_FUSION = os.path.join(DATASET_SAVE_DIR, "X_test_fusion.csv")
DF_PREDICT_RESTNET = os.path.join(DATASET_SAVE_DIR, "resultats_predictions_resnet.csv")
DF_PREDICT_BERT = os.path.join(DATASET_SAVE_DIR, "resultats_predictions_bert.csv")

DF_CLASSIFICATION_REPORT_RESNET = os.path.join(DATASET_SAVE_DIR, "classification_report_resnet.csv")

IMAGE_DIR_TEST = os.path.join(IMAGE_DIR, "test")
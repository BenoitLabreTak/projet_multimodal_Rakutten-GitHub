import os
import pandas as pd
import numpy as np

from zenml import pipeline
from zenml import step, pipeline
from zenml.client import Client
from zenml import ArtifactConfig
from typing_extensions import Annotated
from typing import Tuple

from utils import generate_version


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
    Annotated[pd.DataFrame, "preprocessed_dataset_train", ArtifactConfig(version=generate_version())],
    Annotated[pd.DataFrame, "preprocessed_dataset_test", ArtifactConfig(version=generate_version())],
    Annotated[pd.DataFrame, "preprocessed_dataset_val", ArtifactConfig(version=generate_version())]
]:
    train = raw_dataset_train
    test = raw_dataset_test
    val = raw_dataset_val
    return train, test, val


@pipeline
def pipeline_text_preprocess(
    enable_translation: bool = False,
    textdataset_version: str = None
) -> Tuple[
    Annotated[pd.DataFrame, "preprocessed_dataset_train"],
    Annotated[pd.DataFrame, "preprocessed_dataset_test"],
    Annotated[pd.DataFrame, "preprocessed_dataset_val"]
]:
    if textdataset_version is None:
        textdataset_version = generate_version()
    
    raw_dataset_train, raw_dataset_test, raw_dataset_val = textdata_loader()
    if enable_translation:
        raw_dataset_train = textdata_translation(raw_dataset_train)
        raw_dataset_test = textdata_translation(raw_dataset_test)
        raw_dataset_val = textdata_translation(raw_dataset_val)
    #
    dataset_train, dataset_test, dataset_val = textdata_preprocessing(
        raw_dataset_train, raw_dataset_test, raw_dataset_val)
    return dataset_train, dataset_test, dataset_val


if __name__ == "__main__":
    pipeline_text_preprocess(enable_translation=False)

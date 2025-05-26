#from pipelines import pipeline_image_preprocess
from utils import TextDataSchema
import pandas as pd
import sys

from zenml import pipeline, step

from typing_extensions import Annotated
from typing import Tuple

@pipeline
def pipeline_text_evaluate(
    dataset_train: TextDataSchema = None,
) -> Annotated[float, "textmodel_f1_score"]:
    return 5.1

if __name__ == "__main__":
    # premier paramètre: textdata_version
    # second paramètre: textdata_version
    textdata_version = sys.argv[1] if len(sys.argv) > 1 else None
    textmodel_version = sys.argv[2] if len(sys.argv) > 2 else None
    pipeline_text_evaluate(
        pd.DataFrame({"designation": "toto", "description": "tata", "productid": 6594, "imageid": 6846, "prdtypecode": 1}, index = [0])
    )


from fastapi import FastAPI
from app.api.preprocess_text import router as preprocess_text_router
from app.api.preprocess_image import router as preprocess_image_router
from app.api.predict_text import router as predict_text_router
from app.api.predict_image import router as predict_image_router
from app.api.test_runner import router as test_router
from app.api.evaluate_text import router as evaluate_text_router
from app.api.evaluate_image import router as evaluate_image_router
from app.api.train_text import router as train_text_api_router
from app.api.train_image import router as train_image_api_router

app = FastAPI(
    title="Rakuten MLOps APIs",
    description="API for preprocessing, prediction and evaluation of product categories using CamemBERT and ResNet.",
    version="1.0.0",
    openapi_tags=[
        {"name": "Preprocessing / Text", "description": "Clean and normalize text data"},
        {"name": "Preprocessing / Image", "description": "Enhance and resize images"},
        {"name": "Prediction / Text", "description": "Predict product categories from text"},
        {"name": "Prediction / Image", "description": "Predict categories from product images"},
        {"name": "Evaluation / Text", "description": "Evaluate text model performance"},
        {"name": "Evaluation / Image", "description": "Evaluate image model performance"},
        {"name": "Training / Text", "description": "Train text classification models"},
        {"name": "Training / Image", "description": "Train text RESNET50 classification models"},
        {"name": "Tests", "description": "Run CI-style unit tests on endpoints"}
    ]
)
## Include routers for different functionalities
#Unit Tests
app.include_router(test_router, prefix="/test", tags=["Tests"])

# APIs Preprocessing
app.include_router(preprocess_text_router, prefix="/preprocess", tags=["Preprocessing / Text"])
app.include_router(preprocess_image_router, prefix="/preprocess", tags=["Preprocessing / Image"])

# APIs Prediction
app.include_router(predict_text_router, prefix="/predict", tags=["Prediction / Text"])
app.include_router(predict_image_router, prefix="/predict", tags=["Prediction / Image"])

# APIs Evaluations
app.include_router(evaluate_text_router, prefix="/evaluate", tags=["Evaluation / Text"])
app.include_router(evaluate_image_router, prefix="/evaluate", tags=["Evaluation / Image"])

# APIs Training
app.include_router(train_text_api_router, prefix="/train", tags=["Training / Text"])
app.include_router(train_image_api_router, prefix="/train", tags=["Training / Image"])
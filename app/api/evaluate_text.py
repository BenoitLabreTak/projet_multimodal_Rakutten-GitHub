from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi import Query
from typing import Optional
import pandas as pd
import io
import numpy as np

from app.services.model_text_prediction import evaluate_text_model , evaluate_text_model_on_dataset

router = APIRouter()

#@router.post("/text/manual", summary="Text evaluation (manual entry)")
#async def evaluate_text(
#    designation: str = Form(...),
#    description: str = Form(...),
#    true_label: int = Form(...)
#):
#    predicted_label, label_name, confidence, f1 = evaluate_text_model(designation, description, true_label)
#
#    return {
#        "designation": designation,
#        "description": description,
#        "true_label": true_label,
#        "predicted_label": predicted_label,
#        "label_name": label_name,
#        "confidence_score": confidence,
#        "f1_score": f1
#    }

@router.post("/text/file", summary="Text evaluation (cvs file entry)")
async def evaluate_text_api_file(
    file: UploadFile = File(...),
    sample_size: int = Query(50, description="Number of samples to process (max)")
):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    df["designation"] = df["designation"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # Safety check
    required_cols = {"designation", "description", "prdtypecode"}
    if not required_cols.issubset(df.columns):
        return JSONResponse(status_code=400, content={
            "error": f"Missing columns: {required_cols - set(df.columns)}"
        })

    df = evaluate_text_model_on_dataset(df, sample_size=sample_size)
    avg_confidence = df["confidence_score"].mean().round(2)
    f1 = df["f1_score"].mean().round(2)
    
    return {
        "average_confidence_score": avg_confidence,
        "weighted_f1_score": f1
    }
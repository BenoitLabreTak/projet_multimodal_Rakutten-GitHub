from fastapi import APIRouter, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from typing import Optional
import pandas as pd
import io
import numpy as np

from app.services.model_text_prediction import predict_text_model, predict_text_model_on_dataset

router = APIRouter()


@router.post("/text/manual", summary="Text prediction (manual entry)")
async def predict_text_api(
    designation: str = Form(""),
    description: str = Form("")
):
    predicted_label, label_name, confidence_score = predict_text_model(designation, description)
    return {
        "designation": designation,
        "description": description,
        "prediction": predicted_label,
        "label_name": label_name,
        "confidence": confidence_score
    }


@router.post("/text/file", summary="Text prediction (cvs file entry)")
async def predict_text_api_file(
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
    df = predict_text_model_on_dataset(df, sample_size=sample_size)

    return df[["designation", "description", "predicted_label", "label_name", "confidence_score"]].to_dict(orient="records")
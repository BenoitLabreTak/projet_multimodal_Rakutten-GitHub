from fastapi import APIRouter, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import pandas as pd
import io

from app.services.text_preprocessing import preprocess_txt

router = APIRouter()

@router.post("/text/manual", summary="Text preprocessing (manual entry)")
async def preprocess_text_manual(
    designation: str = Form(...),
    description: str = Form("")
):
    try:
        # Construction du DataFrame avec une seule ligne
        df = pd.DataFrame([{"designation": designation, "description": description}])

        # Application du pipeline de nettoyage
        txt_cleaned = preprocess_txt(df)

        return {
            "designation": designation,
            "description": description,
            "cleaned_text": txt_cleaned.iloc[0]
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/text/file", summary="Text preprocessing (cvs file entry)")
async def preprocess_text_file(
    file: UploadFile = File(...),
    sample_size: int = Query(5, description="Number of samples to process (max)")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        subset_df = df.sample(n=min(sample_size, len(df)), random_state=42)

        required_cols = {'designation', 'description'}
        if not required_cols.issubset(subset_df.columns):
            return JSONResponse(
                status_code=400,
                content={"error": f"Missing required columns: {required_cols - set(df.columns)}"}
            )

        txt_cleaned = preprocess_txt(subset_df)
        return {"cleaned_text": txt_cleaned.tolist()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
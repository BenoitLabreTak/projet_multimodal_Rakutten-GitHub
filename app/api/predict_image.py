from fastapi import APIRouter, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import pandas as pd
import io

from app.services.model_image_prediction import predict_image, predict_dataframe

router = APIRouter()

@router.post("/image/manual", summary="Image prediction (manual entry)")
async def predict_single_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        # Call predict_image with a BytesIO stream
        predicted_label, label_name, confidence_score = predict_image(io.BytesIO(image_bytes))
        
        return {
            "filename": file.filename,
            "predicted_label": predicted_label,
            "label_name": label_name,
            "confidence_score": confidence_score
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@router.post("/image/file", summary="Image prediction (cvs file entry)")
async def predict_from_dataframe(
    file: UploadFile = File(...),
    sample_size: int = Query(50, description="Number of samples to process (max)")
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        df["designation"] = df["designation"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)

        required_cols = {'imageid', 'productid'}
        if not required_cols.issubset(df.columns):
            return JSONResponse(status_code=400, content={
                "error": f"CSV must contain columns: {required_cols}"
            })

        result_df = predict_dataframe(df, sample_size=sample_size)

        # Formatage du résultat
        response = []
        for _, row in result_df.iterrows():
            response.append({
                "filename": f"image_{row['imageid']}_product_{row['productid']}.jpg",
                "predicted_label": int(row["predicted_label"]) if row["predicted_label"] != "N/A" else "N/A",
                "label_name": row["predicted_category"],
                "confidence_score": float(row["confidence_score"]) if "confidence_score" in row and row["confidence_score"] != "N/A" else None
            })

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
from fastapi import APIRouter, UploadFile, File
from fastapi import Query
from fastapi.responses import JSONResponse
import pandas as pd
import io

from app.services.model_image_prediction import evaluate_image_model_on_dataset

router = APIRouter()

@router.post("/image/file", summary="Image evaluation (cvs file entry)")
async def evaluate_image_file_api(
    file: UploadFile = File(...),
    sample_size: int = Query(50, description="Nombre d'images à évaluer"),
    image_dir: str = Query(..., description="Chemin vers le dossier contenant les images")
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df["imageid"] = df["imageid"].fillna("").astype(int)
        df["productid"] = df["productid"].fillna("").astype(int)
        df["prdtypecode"] = df["prdtypecode"].fillna("").astype(int)

        result = evaluate_image_model_on_dataset(df, image_dir=image_dir, sample_size=sample_size)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
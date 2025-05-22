from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import pandas as pd
import base64
import traceback
import app.core.config as config

from app.services.image_preprocessing import preprocess_image_from_pil

router = APIRouter()

@router.post("/image", summary="Image preprocessing (manual entry)")
async def preprocess_uploaded_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        processed_image = preprocess_image_from_pil(image)

        # Encodage en base64 pour retour client
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "filename": file.filename,
            "processed_image_base64": img_base64
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@router.post("/image/file", summary="Batch image preprocessing from CSV")
async def preprocess_images_from_csv(
    file: UploadFile = File(...),
    image_dir: str = Query(config.DATASET_IMAGE_DIR_TEST, description="Chemin du dossier contenant les images"),
    max_samples: int = Query(5, description="Nombre maximal d'échantillons aléatoires à prétraiter")
):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        required_cols = {"imageid", "productid"}
        if not required_cols.issubset(df.columns):
            return JSONResponse(status_code=400, content={"error": f"Colonnes requises : {required_cols}"})

        df_sampled = df.sample(n=min(len(df), max_samples), random_state=42)

        results = []
        for _, row in df_sampled.iterrows():
            image_path = os.path.join(image_dir, f"image_{row['imageid']}_product_{row['productid']}.jpg")
            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                processed_image = preprocess_image_from_pil(image)

                buffered = io.BytesIO()
                processed_image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                results.append({
                    "image_path": image_path,
                    "imageid": row["imageid"],
                    "productid": row["productid"],
                    "processed_image_base64": img_base64
                })
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })

        return results

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
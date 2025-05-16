from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64

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
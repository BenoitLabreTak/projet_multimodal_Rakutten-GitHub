from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from app.services.model_image_training import run_retrain_resnet_model
import traceback
import logging

router = APIRouter()

@router.post("/image", summary="Retrain ResNet image model", responses={
    500: {"description": "Internal Server Error"}
})
async def retrain_resnet_model(
    epochs: int = Query(3, description="Number of training epochs"),
    batch_size: int = Query(32, description="Batch size for training")
):
    try:
        result = run_retrain_resnet_model(
            num_epochs=epochs,
            batch_size=batch_size
        )

        return {
            "old_f1": round(result["old_f1"], 4),
            "new_f1": round(result["new_f1"], 4),
            "model_saved": result["model_saved"],
            "model_path": result["model_path"]
        }

    except Exception as e:
        logging.exception("❌ Erreur lors du retrain Resnet50")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

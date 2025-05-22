from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from app.services.model_text_training import run_retrain_camembert_model
import traceback
import logging

router = APIRouter()

@router.post("/text", summary="Retrain Camembert model", responses={
    500: {"description": "Internal Server Error"}
})

def retrain_camembert_model(
    epochs: int = Query(3, description="Number of training epochs"),
    batch_size: int = Query(4, description="Batch size for training")
):

    try:
        result = run_retrain_camembert_model(
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
        logging.exception("❌ Erreur lors du retrain Camembert")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
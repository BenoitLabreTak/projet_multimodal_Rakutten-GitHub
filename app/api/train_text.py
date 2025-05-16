from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.services.model_text_training import run_retraining_pipeline

router = APIRouter()

@router.post("/retrain/text", summary="Retrain Camembert model")
def retrain_camembert_model():
    try:
        result = run_retraining_pipeline()
        return {"message": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
from fastapi import APIRouter, Depends

from app.dependencies import get_model
from core.inference import DetectorService

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
def ready(model: DetectorService = Depends(get_model)) -> dict[str, object]:
    return {
        "status": "ready" if model.is_ready else "degraded",
        "model_loaded": model.checkpoint_loaded,
        "device": str(model.device),
        "threshold": model.threshold,
    }


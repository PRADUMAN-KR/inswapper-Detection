from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.dependencies import get_model, reload_model, require_admin
from core.inference import DetectorService

router = APIRouter(dependencies=[Depends(require_admin)])


class ThresholdUpdate(BaseModel):
    threshold: float = Field(ge=0.0, le=1.0)


@router.post("/reload-model")
def reload_loaded_model(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, object]:
    service = reload_model(settings)
    return {"model_loaded": service.checkpoint_loaded, "device": str(service.device)}


@router.post("/threshold")
def update_threshold(
    payload: ThresholdUpdate,
    service: Annotated[DetectorService, Depends(get_model)],
) -> dict[str, float]:
    service.threshold = payload.threshold
    return {"threshold": service.threshold}


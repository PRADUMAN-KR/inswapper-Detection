from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings
from core.inference import DetectorService

_service: DetectorService | None = None


def build_service(settings: Settings) -> DetectorService:
    return DetectorService.from_checkpoint(
        checkpoint_path=settings.model_path,
        device=settings.device,
        threshold=settings.threshold,
        allow_missing=True,
    )


def get_model(settings: Annotated[Settings, Depends(get_settings)]) -> DetectorService:
    global _service
    if _service is None:
        _service = build_service(settings)
    return _service


def reload_model(settings: Settings | None = None) -> DetectorService:
    global _service
    settings = settings or get_settings()
    _service = build_service(settings)
    return _service


def require_admin(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token is not configured.",
        )
    expected = f"Bearer {settings.admin_token}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
        )


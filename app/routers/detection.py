import base64
import binascii
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.config import Settings, get_settings
from app.dependencies import get_model
from app.schemas.detection import (
    BatchDetectionResponse,
    DetectionRequest,
    DetectionResponse,
    VideoDetectionResponse,
)
from core.inference import DetectorService

router = APIRouter()


async def _read_upload(file: UploadFile, settings: Settings) -> bytes:
    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )
    return data


@router.post("", response_model=DetectionResponse)
async def detect(
    file: Annotated[UploadFile, File(description="Image file to classify.")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DetectorService, Depends(get_model)],
) -> DetectionResponse:
    data = await _read_upload(file, settings)
    result = service.predict_bytes(data)
    return DetectionResponse(filename=file.filename, result=result)


@router.post("/base64", response_model=DetectionResponse)
async def detect_base64(
    payload: DetectionRequest,
    service: Annotated[DetectorService, Depends(get_model)],
) -> DetectionResponse:
    try:
        data = base64.b64decode(payload.image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid base64 image.") from exc
    result = service.predict_bytes(data)
    return DetectionResponse(filename=None, result=result)


@router.post("/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: Annotated[UploadFile, File(description="Video file to classify.")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DetectorService, Depends(get_model)],
    frames_per_scene: Annotated[int, Query(ge=1, le=12)] = 6,
    scene_threshold: Annotated[float, Query(ge=0.05, le=1.0)] = 0.55,
    max_scenes: Annotated[int, Query(ge=1, le=50)] = 12,
) -> VideoDetectionResponse:
    data = await _read_upload(file, settings)
    suffix = f".{file.filename.rsplit('.', 1)[-1]}" if file.filename and "." in file.filename else ".mp4"
    try:
        result = service.predict_video_bytes(
            data,
            suffix=suffix,
            frames_per_scene=frames_per_scene,
            scene_threshold=scene_threshold,
            max_scenes=max_scenes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return VideoDetectionResponse(filename=file.filename, **result.__dict__)


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: Annotated[list[UploadFile], File(description="Image files to classify.")],
    settings: Annotated[Settings, Depends(get_settings)],
    service: Annotated[DetectorService, Depends(get_model)],
) -> BatchDetectionResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    images = [await _read_upload(file, settings) for file in files]
    results = service.predict_batch_bytes(images)
    return BatchDetectionResponse(
        items=[
            DetectionResponse(filename=file.filename, result=result)
            for file, result in zip(files, results, strict=True)
        ]
    )

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded image.")


class DetectionResult(BaseModel):
    model_config = {"from_attributes": True}

    label: str
    is_fake: bool
    fake_probability: float = Field(ge=0.0, le=1.0)
    real_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class DetectionResponse(BaseModel):
    filename: str | None = None
    result: DetectionResult


class BatchDetectionResponse(BaseModel):
    items: list[DetectionResponse]


class VideoFrameDetection(BaseModel):
    model_config = {"from_attributes": True}

    scene_index: int
    frame_index: int
    timestamp_sec: float
    result: DetectionResult


class VideoDetectionResponse(BaseModel):
    model_config = {"from_attributes": True}

    filename: str | None = None
    result: DetectionResult
    scene_count: int
    sampled_frame_count: int
    frames: list[VideoFrameDetection]

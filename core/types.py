from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionResult:
    label: str
    is_fake: bool
    fake_probability: float
    real_probability: float
    confidence: float
    threshold: float


@dataclass(frozen=True)
class VideoFrameResult:
    scene_index: int
    frame_index: int
    timestamp_sec: float
    result: DetectionResult


@dataclass(frozen=True)
class VideoDetectionResult:
    result: DetectionResult
    scene_count: int
    sampled_frame_count: int
    frames: list[VideoFrameResult]

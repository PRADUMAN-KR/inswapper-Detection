from dataclasses import dataclass
from functools import lru_cache
import os

import cv2
import numpy as np
from PIL import Image


FaceBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class FaceDetection:
    box: FaceBox
    confidence: float
    backend: str


class FaceDetector:
    def detect(self, image: Image.Image) -> FaceDetection | None:
        raise NotImplementedError


class InsightFaceDetector(FaceDetector):
    def __init__(self, det_size: tuple[int, int] = (640, 640)) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError(
                "InsightFace is required for production face detection. "
                "Install serving/training dependencies with insightface and onnxruntime."
            ) from exc

        providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def detect(self, image: Image.Image) -> FaceDetection | None:
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.app.get(bgr)
        if not faces:
            return None
        face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
        x1, y1, x2, y2 = [int(round(value)) for value in face.bbox.tolist()]
        x1 = max(0, min(image.width, x1))
        y1 = max(0, min(image.height, y1))
        x2 = max(0, min(image.width, x2))
        y2 = max(0, min(image.height, y2))
        confidence = float(getattr(face, "det_score", 1.0))
        return FaceDetection(box=(x1, y1, x2, y2), confidence=confidence, backend="insightface")


class OpenCVHaarDetector(FaceDetector):
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: Image.Image) -> FaceDetection | None:
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return FaceDetection(box=(int(x), int(y), int(x + w), int(y + h)), confidence=0.5, backend="opencv_haar")


@lru_cache
def get_face_detector(backend: str | None = None) -> FaceDetector:
    backend = backend or os.getenv("INSWAPPER_FACE_DETECTOR", "insightface")
    if backend == "insightface":
        return InsightFaceDetector()
    if backend == "opencv_haar":
        return OpenCVHaarDetector()
    raise ValueError(f"Unknown face detector backend: {backend}")


def detect_face(
    image: Image.Image,
    backend: str | None = None,
    min_confidence: float | None = None,
) -> FaceDetection | None:
    detection = get_face_detector(backend).detect(image)
    if detection is None:
        return None
    threshold = float(os.getenv("INSWAPPER_FACE_MIN_CONFIDENCE", min_confidence or 0.5))
    if detection.confidence < threshold:
        return None
    return detection


def expand_box(box: FaceBox, scale: float, width: int, height: int) -> FaceBox:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * scale
    return (
        max(0, int(round(cx - size / 2))),
        max(0, int(round(cy - size / 2))),
        min(width, int(round(cx + size / 2))),
        min(height, int(round(cy + size / 2))),
    )

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.dependencies import get_model
from app.main import create_app
from core.types import DetectionResult


@pytest.fixture()
def client() -> TestClient:
    app = create_app()

    class StubDetectorService:
        checkpoint_loaded = True
        device = "cpu"
        threshold = 0.5
        image_size = 256
        frequency_mode = "fft"

        @property
        def is_ready(self):
            return True

        def predict_bytes(self, image_bytes):
            return DetectionResult("real", False, 0.1, 0.9, 0.9, 0.5)

        def predict_batch_bytes(self, images):
            return [self.predict_bytes(image) for image in images]

    app.dependency_overrides[get_model] = lambda: StubDetectorService()
    return TestClient(app)


@pytest.fixture()
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (64, 64), color=(130, 90, 40))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()

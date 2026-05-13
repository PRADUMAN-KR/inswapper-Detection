import torch

from core.inference import DetectorService
from core.model import ConvNeXtTinyDetector


def test_predict_tensor_returns_detection_result():
    service = DetectorService(
        model=ConvNeXtTinyDetector(pretrained=False),
        device=torch.device("cpu"),
        threshold=0.5,
        checkpoint_loaded=False,
        image_size=256,
        frequency_mode="fft",
    )
    result = service.predict_tensor(torch.zeros(3, 256, 256))
    assert result.label in {"real", "fake"}
    assert 0.0 <= result.fake_probability <= 1.0

from pathlib import Path

import torch

from core.model import ConvNeXtTinyDetector, load_from_checkpoint
from core.postprocessing import logits_to_result
from core.preprocessing import preprocess_image, preprocess_pil_image
from core.types import DetectionResult, VideoDetectionResult, VideoFrameResult
from core.video import sample_scene_aware_frames_from_bytes, sample_scene_aware_frames_from_path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def predict(model: torch.nn.Module, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(batch.to(device))


def predict_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> list[DetectionResult]:
    logits = predict(model, batch, device)
    return [logits_to_result(logit, threshold) for logit in logits]


def aggregate_frame_results(results: list[DetectionResult], threshold: float) -> DetectionResult:
    if not results:
        raise ValueError("Cannot aggregate empty frame results.")
    fake_probability = sum(result.fake_probability for result in results) / len(results)
    is_fake = fake_probability >= threshold
    confidence = fake_probability if is_fake else 1.0 - fake_probability
    return DetectionResult(
        label="fake" if is_fake else "real",
        is_fake=is_fake,
        fake_probability=round(fake_probability, 6),
        real_probability=round(1.0 - fake_probability, 6),
        confidence=round(confidence, 6),
        threshold=threshold,
    )


class DetectorService:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        threshold: float,
        checkpoint_loaded: bool,
        image_size: int = 224,
    ) -> None:
        self.model = model
        self.device = device
        self.threshold = threshold
        self.checkpoint_loaded = checkpoint_loaded
        self.image_size = image_size

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "auto",
        threshold: float = 0.5,
        allow_missing: bool = False,
    ) -> "DetectorService":
        resolved = resolve_device(device)
        path = Path(checkpoint_path)
        loaded = path.exists()
        if loaded:
            model = load_from_checkpoint(path, resolved)
            checkpoint = torch.load(path, map_location="cpu")
            image_size = int(checkpoint.get("config", {}).get("model", {}).get("image_size", 224))
        elif allow_missing:
            model = ConvNeXtTinyDetector(pretrained=False).to(resolved)
            image_size = 224
        else:
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        model.eval()
        return cls(model=model, device=resolved, threshold=threshold, checkpoint_loaded=loaded, image_size=image_size)

    def predict_tensor(self, image: torch.Tensor) -> DetectionResult:
        batch = image.unsqueeze(0) if image.ndim == 3 else image
        return predict_batch(self.model, batch, self.device, self.threshold)[0]

    def predict_bytes(self, image_bytes: bytes) -> DetectionResult:
        tensor = preprocess_image(image_bytes, image_size=self.image_size)
        return self.predict_tensor(tensor)

    def predict_batch_bytes(self, images: list[bytes]) -> list[DetectionResult]:
        tensors = [preprocess_image(image, image_size=self.image_size) for image in images]
        return predict_batch(self.model, torch.stack(tensors), self.device, self.threshold)

    def predict_video_path(
        self,
        video_path: str | Path,
        frames_per_scene: int = 6,
        scene_threshold: float = 0.55,
        max_scenes: int = 12,
    ) -> VideoDetectionResult:
        frames, scene_count = sample_scene_aware_frames_from_path(
            video_path,
            frames_per_scene=frames_per_scene,
            scene_threshold=scene_threshold,
            max_scenes=max_scenes,
        )
        tensors = [preprocess_pil_image(frame.image, image_size=self.image_size) for frame in frames]
        results = predict_batch(self.model, torch.stack(tensors), self.device, self.threshold)
        frame_results = [
            VideoFrameResult(
                scene_index=frame.scene_index,
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                result=result,
            )
            for frame, result in zip(frames, results, strict=True)
        ]
        return VideoDetectionResult(
            result=aggregate_frame_results(results, self.threshold),
            scene_count=scene_count,
            sampled_frame_count=len(frame_results),
            frames=frame_results,
        )

    def predict_video_bytes(
        self,
        video_bytes: bytes,
        suffix: str = ".mp4",
        frames_per_scene: int = 6,
        scene_threshold: float = 0.55,
        max_scenes: int = 12,
    ) -> VideoDetectionResult:
        frames, scene_count = sample_scene_aware_frames_from_bytes(
            video_bytes,
            suffix=suffix,
            frames_per_scene=frames_per_scene,
            scene_threshold=scene_threshold,
            max_scenes=max_scenes,
        )
        tensors = [preprocess_pil_image(frame.image, image_size=self.image_size) for frame in frames]
        results = predict_batch(self.model, torch.stack(tensors), self.device, self.threshold)
        frame_results = [
            VideoFrameResult(
                scene_index=frame.scene_index,
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                result=result,
            )
            for frame, result in zip(frames, results, strict=True)
        ]
        return VideoDetectionResult(
            result=aggregate_frame_results(results, self.threshold),
            scene_count=scene_count,
            sampled_frame_count=len(frame_results),
            frames=frame_results,
        )

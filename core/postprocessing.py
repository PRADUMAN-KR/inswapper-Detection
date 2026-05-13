import torch

from core.types import DetectionResult


def logits_to_result(logits: torch.Tensor, threshold: float) -> DetectionResult:
    prob_fake = torch.sigmoid(logits.detach().float()).flatten()[0].item()
    is_fake = prob_fake >= threshold
    confidence = prob_fake if is_fake else 1.0 - prob_fake
    return DetectionResult(
        label="fake" if is_fake else "real",
        is_fake=is_fake,
        fake_probability=round(prob_fake, 6),
        real_probability=round(1.0 - prob_fake, 6),
        confidence=round(confidence, 6),
        threshold=threshold,
    )

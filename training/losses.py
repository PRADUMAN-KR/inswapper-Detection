import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(targets == 1, prob, 1 - prob)
        loss = (1 - pt).pow(self.gamma) * bce
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * loss
        return loss.mean()


class MultiTaskDetectionLoss(nn.Module):
    def __init__(
        self,
        focal_gamma: float = 2.0,
        alpha: float | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.binary_loss = FocalLoss(gamma=focal_gamma, alpha=alpha)
        self.quality_loss = nn.CrossEntropyLoss()
        self.weights = weights or {
            "real_fake": 1.0,
            "inswapper": 0.7,
            "gan": 0.5,
            "boundary": 0.4,
            "quality": 0.2,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        losses = {
            "real_fake": self.binary_loss(outputs["real_fake"], targets["real_fake"].float().view(-1, 1)),
            "inswapper": self.binary_loss(outputs["inswapper"], targets["inswapper"].float().view(-1, 1)),
            "gan": self.binary_loss(outputs["gan"], targets["gan"].float().view(-1, 1)),
            "boundary": self.binary_loss(outputs["boundary"], targets["boundary"].float().view(-1, 1)),
            "quality": self.quality_loss(outputs["quality"], targets["quality"].long().view(-1)),
        }
        total = sum(self.weights[name] * loss for name, loss in losses.items())
        log_values = {name: float(loss.detach().cpu()) for name, loss in losses.items()}
        log_values["total"] = float(total.detach().cpu())
        return total, log_values

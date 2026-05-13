from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from training.metrics import BinaryMetrics, compute_binary_metrics, fuse_detection_scores


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
    amp: bool = True,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, grad_accum_steps)
    last_lr = optimizer.param_groups[0]["lr"]
    for step, batch in enumerate(loader, start=1):
        rgb = batch["rgb"].to(device, non_blocking=True)
        frequency = batch["frequency"].to(device, non_blocking=True)
        targets = {
            name: value.to(device, non_blocking=True)
            for name, value in batch["targets"].items()
        }
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(rgb, frequency=frequency, return_dict=True)
            raw_loss = criterion(outputs, targets)
            loss = (raw_loss[0] if isinstance(raw_loss, tuple) else raw_loss) / grad_accum_steps
        if scaler is not None and amp and device.type == "cuda":
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = step % grad_accum_steps == 0 or step == len(loader)
        if should_step:
            if scaler is not None and amp and device.type == "cuda":
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            last_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * grad_accum_steps * rgb.size(0)
    return running_loss / len(loader.dataset), last_lr


@torch.inference_mode()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = True,
    score_fusion_weights: dict[str, float] | None = None,
) -> tuple[float, BinaryMetrics]:
    model.eval()
    running_loss = 0.0
    labels_all: list[float] = []
    real_fake_all: list[float] = []
    inswapper_all: list[float] = []
    gan_all: list[float] = []
    boundary_all: list[float] = []
    for batch in loader:
        rgb = batch["rgb"].to(device, non_blocking=True)
        frequency = batch["frequency"].to(device, non_blocking=True)
        targets = {
            name: value.to(device, non_blocking=True)
            for name, value in batch["targets"].items()
        }
        with autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(rgb, frequency=frequency, return_dict=True)
            raw_loss = criterion(outputs, targets)
            loss = raw_loss[0] if isinstance(raw_loss, tuple) else raw_loss
        running_loss += loss.item() * rgb.size(0)
        labels_all.extend(targets["real_fake"].flatten().detach().cpu().tolist())
        real_fake_all.extend(torch.sigmoid(outputs["real_fake"]).flatten().detach().cpu().tolist())
        inswapper_all.extend(torch.sigmoid(outputs["inswapper"]).flatten().detach().cpu().tolist())
        gan_all.extend(torch.sigmoid(outputs["gan"]).flatten().detach().cpu().tolist())
        boundary_all.extend(torch.sigmoid(outputs["boundary"]).flatten().detach().cpu().tolist())
    fused = fuse_detection_scores(
        real_fake=torch.tensor(real_fake_all).numpy(),
        inswapper=torch.tensor(inswapper_all).numpy(),
        gan=torch.tensor(gan_all).numpy(),
        boundary=torch.tensor(boundary_all).numpy(),
        weights=score_fusion_weights,
    )
    return running_loss / len(loader.dataset), compute_binary_metrics(labels_all, fused.tolist())


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    metrics: BinaryMetrics,
    config: dict[str, Any],
    threshold: float | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics.__dict__,
            "threshold": threshold if threshold is not None else metrics.best_threshold,
            "config": config,
        },
        path,
    )

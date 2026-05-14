import argparse
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.model import ConvNeXtTinyDetector
from training.dataset import DeepfakeDataset, ZarrDeepfakeDataset, create_dataset
from training.losses import MultiTaskDetectionLoss
from training.trainer import EarlyStopping, save_checkpoint, train_epoch, val_epoch
from training.utils import (
    build_warmup_cosine_scheduler,
    count_labels,
    load_yaml,
    resolve_device,
    seed_worker,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train INSwapper detector.")
    parser.add_argument("--config", default="configs/convnext_tiny.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def build_loader(dataset: DeepfakeDataset | ZarrDeepfakeDataset, cfg: dict, train: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = train
    if train and cfg["train"].get("balanced_sampler", True):
        counts = count_labels(dataset.labels)
        weights = [1.0 / max(1, counts[label]) for label in dataset.labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False

    num_workers = int(cfg["train"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def auto_focal_alpha(labels: list[int]) -> float:
    counts = count_labels(labels)
    total = max(1, counts[0] + counts[1])
    return counts[0] / total


def append_history(path: str | Path, row: dict[str, float | int | str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def phase_for_epoch(epoch: int, cfg: dict) -> str:
    phases = cfg["train"].get("phases", {})
    freeze_until = int(phases.get("freeze_backbone_until", 3))
    partial_until = int(phases.get("unfreeze_last_stages_until", 15))
    if epoch < freeze_until:
        return "freeze_backbone"
    if epoch < partial_until:
        return "unfreeze_last_stages"
    return "unfreeze_full"


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg["train"].get("device", "auto"))
    torch.backends.cudnn.benchmark = bool(cfg["train"].get("cudnn_benchmark", True))
    checkpoint = torch.load(args.resume, map_location=device) if args.resume else None
    start_epoch = int(checkpoint["epoch"]) + 1 if checkpoint else 0

    train_ds = create_dataset(
        cfg["data"]["train_manifest"],
        image_size=cfg["model"]["image_size"],
        train=True,
        root_dir=cfg["data"].get("root_dir"),
        frequency_mode=cfg["model"].get("frequency_mode", "fft"),
    )
    val_ds = create_dataset(
        cfg["data"]["val_manifest"],
        image_size=cfg["model"]["image_size"],
        train=False,
        root_dir=cfg["data"].get("root_dir"),
        frequency_mode=cfg["model"].get("frequency_mode", "fft"),
    )
    train_loader = build_loader(train_ds, cfg, train=True, seed=seed)
    val_loader = build_loader(val_ds, cfg, train=False, seed=seed)

    model = ConvNeXtTinyDetector(
        pretrained=cfg["model"].get("pretrained", True),
        backbone=cfg["model"]["backbone"],
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    ).to(device)
    current_phase = phase_for_epoch(start_epoch, cfg)
    model.set_training_phase(current_phase)

    alpha = cfg["loss"].get("alpha", "auto")
    if alpha == "auto":
        alpha = auto_focal_alpha(train_ds.labels)
    criterion = MultiTaskDetectionLoss(
        focal_gamma=cfg["loss"].get("focal_gamma", 2.0),
        alpha=alpha,
        weights=cfg["loss"].get("task_weights"),
    )
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
    )
    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    update_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_steps = update_steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(cfg["scheduler"].get("warmup_epochs", 1) * update_steps_per_epoch)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg["scheduler"].get("min_lr_ratio", 0.05),
    )
    scaler = GradScaler(device.type, enabled=cfg["train"].get("amp", True) and device.type == "cuda")
    best_auc = -1.0
    best_product_metric = -1.0
    early_stopping = EarlyStopping(
        patience=int(cfg["train"].get("early_stopping_patience", 6)),
        min_delta=float(cfg["train"].get("early_stopping_min_delta", 0.0)),
    )

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("optimizer state shape changed after backbone unfreeze; continuing with a fresh optimizer")
        if checkpoint.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError:
                print("scheduler state shape changed after backbone unfreeze; continuing with a fresh scheduler")
        best_auc = float(checkpoint.get("metrics", {}).get("auc", -1.0))
        best_product_metric = float(checkpoint.get("metrics", {}).get("product_score", best_auc))
        early_stopping.best = best_product_metric

    alpha_display = "none" if alpha is None else f"{float(alpha):.4f}"
    print(
        f"training backbone={cfg['model']['backbone']} frequency={cfg['model'].get('frequency_mode', 'fft')} device={device} "
        f"train={len(train_ds)} val={len(val_ds)} alpha={alpha_display}"
    )

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        next_phase = phase_for_epoch(epoch, cfg)
        if next_phase != current_phase:
            current_phase = next_phase
            model.set_training_phase(current_phase)
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=cfg["optimizer"]["lr"],
                weight_decay=cfg["optimizer"]["weight_decay"],
                betas=tuple(cfg["optimizer"].get("betas", [0.9, 0.999])),
            )
            scheduler = build_warmup_cosine_scheduler(
                optimizer,
                total_steps=max(1, (cfg["train"]["epochs"] - epoch) * update_steps_per_epoch),
                warmup_steps=warmup_steps,
                min_lr_ratio=cfg["scheduler"].get("min_lr_ratio", 0.05),
            )
            print(f"switched training phase to {current_phase}")

        train_loss, lr = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            cfg["train"].get("amp", True),
            scheduler=scheduler,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=cfg["train"].get("max_grad_norm"),
        )
        val_loss, metrics = val_epoch(
            model,
            val_loader,
            criterion,
            device,
            cfg["train"].get("amp", True),
            score_fusion_weights=cfg.get("score_fusion"),
        )

        print(
            f"epoch={epoch:03d} phase={current_phase} lr={lr:.6g} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"product={metrics.product_score:.4f} auc={metrics.auc:.4f} f1={metrics.f1:.4f} eer={metrics.eer:.4f} "
            f"best_threshold={metrics.best_threshold:.3f}"
        )
        append_history(
            cfg["paths"]["history_csv"],
            {
                "epoch": epoch,
                "phase": current_phase,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "auc": metrics.auc,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "eer": metrics.eer,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "product_score": metrics.product_score,
                "best_threshold": metrics.best_threshold,
            },
        )
        save_checkpoint(cfg["paths"]["last_checkpoint"], model, optimizer, scheduler, epoch, metrics, cfg)
        if metrics.product_score > best_product_metric:
            best_product_metric = metrics.product_score
            best_auc = metrics.auc
            save_checkpoint(cfg["paths"]["best_checkpoint"], model, optimizer, scheduler, epoch, metrics, cfg)
        if early_stopping.step(metrics.product_score):
            print(f"early stopping at epoch={epoch} best_product_metric={early_stopping.best:.4f}")
            break


if __name__ == "__main__":
    main()

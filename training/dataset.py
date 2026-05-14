from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset

from core.frequency import frequency_features

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception:
    A = None
    ToTensorV2 = None


def build_transforms(image_size: int, train: bool = True):
    if A is None or ToTensorV2 is None:
        return None
    steps = [
        A.Resize(image_size, image_size),
    ]
    if train:
        try:
            image_compression = A.ImageCompression(quality_range=(45, 100), p=0.45)
        except TypeError:
            image_compression = A.ImageCompression(quality_lower=45, quality_upper=100, p=0.45)
        try:
            dropout = A.CoarseDropout(max_holes=2, max_height=24, max_width=24, p=0.15)
        except TypeError:
            dropout = A.CoarseDropout(p=0.15)
        steps.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=8, border_mode=0, p=0.35),
                image_compression,
                A.GaussNoise(p=0.2),
                A.ColorJitter(p=0.2),
                A.MotionBlur(blur_limit=3, p=0.12),
                A.Sharpen(p=0.12),
                dropout,
            ]
        )
    steps.extend(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(steps)


class DeepfakeDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 256,
        train: bool = True,
        root_dir: str | Path | None = None,
        frequency_mode: str = "fft",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root_dir = Path(root_dir) if root_dir else self.manifest_path.parent
        self.frame = pd.read_csv(self.manifest_path)
        self.frame = self.frame.dropna(how="all").reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Manifest has no samples: {self.manifest_path}")
        required = {"path", "label"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        self.transforms = build_transforms(image_size=image_size, train=train)
        self.image_size = image_size
        self.frequency_mode = frequency_mode
        self.labels = self.frame["label"].astype(int).tolist()
        invalid_labels = sorted(set(self.labels) - {0, 1})
        if invalid_labels:
            raise ValueError(f"Labels must be 0 or 1, found: {invalid_labels}")
        if "quality_label" in self.frame.columns:
            quality_values = set(self.frame["quality_label"].dropna().astype(int).tolist())
            invalid_quality = sorted(quality_values - {0, 1, 2})
            if invalid_quality:
                raise ValueError(f"quality_label must be 0, 1, or 2, found: {invalid_quality}")

    def __len__(self) -> int:
        return len(self.frame)

    def _load_image(self, path: str) -> Image.Image:
        image_path = Path(path)
        if not image_path.is_absolute():
            image_path = self.root_dir / image_path
        with Image.open(image_path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    def _target(self, row) -> dict[str, torch.Tensor]:
        label = float(row["label"])
        fake_type = str(row.get("fake_type", row.get("source", "real"))).lower()
        is_inswapper = float(row.get("is_inswapper", int("inswapper" in fake_type)))
        boundary = float(row.get("boundary_label", label))
        quality = int(row.get("quality_label", 0))
        return {
            "real_fake": torch.tensor(label, dtype=torch.float32),
            "inswapper": torch.tensor(is_inswapper, dtype=torch.float32),
            "boundary": torch.tensor(boundary, dtype=torch.float32),
            "quality": torch.tensor(quality, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = self._load_image(str(row["path"]))
        if self.transforms is not None:
            transformed = self.transforms(image=np.asarray(image))
            rgb = transformed["image"]
        else:
            image = image.resize((self.image_size, self.image_size))
            arr = torch.from_numpy(np.asarray(image).astype("float32")).permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (arr - mean) / std
        frequency = frequency_features(rgb.unsqueeze(0), mode=self.frequency_mode).squeeze(0)
        return {"rgb": rgb, "frequency": frequency, "targets": self._target(row)}


class ZarrDeepfakeDataset(Dataset):
    def __init__(
        self,
        zarr_path: str | Path,
        image_size: int = 256,
        train: bool = True,
        frequency_mode: str = "fft",
    ) -> None:
        try:
            import zarr
        except ImportError as exc:
            raise ImportError("zarr is required for ZarrDeepfakeDataset. Run: uv sync --extra train") from exc

        self.zarr_path = Path(zarr_path)
        self.group: Any = zarr.open_group(str(self.zarr_path), mode="r")
        self.images = self.group["images"]
        self.frame = pd.read_csv(self.zarr_path / "metadata.csv")
        self.transforms = build_transforms(image_size=image_size, train=train)
        self.image_size = image_size
        self.frequency_mode = frequency_mode
        self.labels = np.asarray(self.group["labels"][:], dtype=np.int64).tolist()
        invalid_labels = sorted(set(self.labels) - {0, 1})
        if invalid_labels:
            raise ValueError(f"Labels must be 0 or 1, found: {invalid_labels}")

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def _target(self, index: int) -> dict[str, torch.Tensor]:
        label = float(self.group["labels"][index])
        is_inswapper = float(self.group["is_inswapper"][index])
        boundary = float(self.group["boundary_label"][index])
        quality = int(self.group["quality_label"][index])
        return {
            "real_fake": torch.tensor(label, dtype=torch.float32),
            "inswapper": torch.tensor(is_inswapper, dtype=torch.float32),
            "boundary": torch.tensor(boundary, dtype=torch.float32),
            "quality": torch.tensor(quality, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = np.asarray(self.images[index], dtype=np.uint8)
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            rgb = transformed["image"]
        else:
            if image.shape[:2] != (self.image_size, self.image_size):
                pil_image = Image.fromarray(image).resize((self.image_size, self.image_size))
                image = np.asarray(pil_image, dtype=np.uint8)
            arr = torch.from_numpy(image.astype("float32")).permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb = (arr - mean) / std
        frequency = frequency_features(rgb.unsqueeze(0), mode=self.frequency_mode).squeeze(0)
        return {"rgb": rgb, "frequency": frequency, "targets": self._target(index)}


def create_dataset(
    manifest_path: str | Path,
    image_size: int = 256,
    train: bool = True,
    root_dir: str | Path | None = None,
    frequency_mode: str = "fft",
) -> DeepfakeDataset | ZarrDeepfakeDataset:
    path = Path(manifest_path)
    if path.suffix == ".zarr" or path.is_dir():
        return ZarrDeepfakeDataset(
            path,
            image_size=image_size,
            train=train,
            frequency_mode=frequency_mode,
        )
    return DeepfakeDataset(
        path,
        image_size=image_size,
        train=train,
        root_dir=root_dir,
        frequency_mode=frequency_mode,
    )

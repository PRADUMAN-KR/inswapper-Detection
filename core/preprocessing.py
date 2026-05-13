from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def decode_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def align_face(image: Image.Image) -> Image.Image:
    """Placeholder for face alignment; keep API stable for future RetinaFace/MTCNN."""
    return image


def normalize(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    return normalize(align_face(image.convert("RGB")), image_size=image_size)


def preprocess_image(image_bytes: bytes, image_size: int = 224) -> torch.Tensor:
    image = align_face(decode_image(image_bytes))
    return normalize(image, image_size=image_size)


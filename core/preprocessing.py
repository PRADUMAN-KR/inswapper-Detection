from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import torch

from core.face_detection import detect_face, expand_box

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def decode_image(image_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(image_bytes)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def align_face(image: Image.Image, require_face: bool = True) -> Image.Image:
    """Crop the dominant production-grade face detection region."""
    try:
        detection = detect_face(image)
    except RuntimeError:
        if require_face:
            raise
        return image
    if detection is None:
        if require_face:
            raise ValueError("No face detected in image.")
        return image
    return image.crop(expand_box(detection.box, scale=1.5, width=image.width, height=image.height))


def normalize(image: Image.Image, image_size: int = 256) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_pil_image(image: Image.Image, image_size: int = 256, require_face: bool = True) -> torch.Tensor:
    return normalize(align_face(image.convert("RGB"), require_face=require_face), image_size=image_size)


def preprocess_image(image_bytes: bytes, image_size: int = 256, require_face: bool = True) -> torch.Tensor:
    image = align_face(decode_image(image_bytes), require_face=require_face)
    return normalize(image, image_size=image_size)

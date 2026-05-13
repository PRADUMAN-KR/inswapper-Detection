import torch
from torch.nn import functional as F


def fft_magnitude_features(batch: torch.Tensor) -> torch.Tensor:
    """Return normalized log FFT magnitude maps."""
    spectrum = torch.fft.fft2(batch.float(), norm="ortho")
    magnitude = torch.log1p(torch.abs(torch.fft.fftshift(spectrum, dim=(-2, -1))))
    mean = magnitude.mean(dim=(-2, -1), keepdim=True)
    std = magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (magnitude - mean) / std


def high_pass_features(batch: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    """Approximate high-frequency residual maps for compression/manipulation artifacts."""
    padding = kernel_size // 2
    blurred = F.avg_pool2d(batch.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return batch.float() - blurred


def frequency_features(batch: torch.Tensor, mode: str = "fft") -> torch.Tensor:
    if mode == "fft":
        return fft_magnitude_features(batch)
    if mode == "high_pass":
        return high_pass_features(batch)
    raise ValueError(f"Unknown frequency mode: {mode}")

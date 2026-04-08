"""Device utilities with cached CUDA checks."""

import torch

_CUDA_AVAILABLE: bool = torch.cuda.is_available()
_DEVICE: str = "cuda" if _CUDA_AVAILABLE else "cpu"


def get_device() -> str:
    return _DEVICE


def is_cuda_available() -> bool:
    return _CUDA_AVAILABLE


def empty_cache() -> None:
    if _CUDA_AVAILABLE:
        torch.cuda.empty_cache()

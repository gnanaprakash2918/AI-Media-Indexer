import logging
from enum import Enum
from pydantic import Field, computed_field


class HardwareProfile(str, Enum):
    """Hardware profile for throughput tuning.

    NOTE: Profiles affect ONLY batch sizes and parallelism.
    Model quality and accuracy are IDENTICAL across all profiles.
    """
    LAPTOP = "laptop"  # < 8GB VRAM
    WORKSTATION = "workstation"  # 8-20GB VRAM
    SERVER = "server"  # > 20GB VRAM
    CPU_ONLY = "cpu_only"  # No GPU


def get_hardware_profile() -> dict:
    """Detect hardware and return optimal settings.

    Returns throughput settings based on available VRAM.
    NOTE: Model quality is NEVER reduced - only batch sizes change.
    """
    profile = {
        "name": HardwareProfile.CPU_ONLY,
        "batch_size": 4,
        "worker_count": 1,
        "embedding_batch_size": 4,
        "device": "cpu",
    }

    import torch
    if not torch.cuda.is_available():
        logging.info("No GPU detected. Using CPU profile (full accuracy, slower).")
        return profile

    profile["device"] = "cuda"

    try:
        # Get VRAM in GB
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)

        if vram_gb >= 20.0:  # Server (e.g. A100, 3090/4090 24GB)
            profile["name"] = HardwareProfile.SERVER
            profile["batch_size"] = 16
            profile["worker_count"] = 4
            profile["embedding_batch_size"] = 32
            logging.info(f"Detected SERVER profile ({vram_gb:.1f}GB VRAM)")
        elif vram_gb >= 8.0:  # Workstation (e.g. 3070/4070, 8-16GB)
            profile["name"] = HardwareProfile.WORKSTATION
            profile["batch_size"] = 8
            profile["worker_count"] = 2
            profile["embedding_batch_size"] = 16
            logging.info(f"Detected WORKSTATION profile ({vram_gb:.1f}GB VRAM)")
        else:  # Laptop / Low-end (< 8GB)
            profile["name"] = HardwareProfile.LAPTOP
            profile["batch_size"] = 4
            profile["worker_count"] = 1
            profile["embedding_batch_size"] = 8
            logging.info(f"Detected LAPTOP profile ({vram_gb:.1f}GB VRAM)")

    except Exception as e:
        logging.warning(f"Failed to detect detailed hardware specs: {e}")

    return profile


_HW_PROFILE = get_hardware_profile()


class HardwareSettings:
    """Hardware configuration and performance settings."""
    
    batch_size: int = Field(
        default=_HW_PROFILE["batch_size"],
        description="Batch size for inference",
    )
    embedding_batch_size: int = Field(
        default=_HW_PROFILE["embedding_batch_size"],
        description="Batch size for embedding generation",
    )
    worker_count: int = Field(
        default=_HW_PROFILE["worker_count"],
        description="Number of parallel ingestion workers",
    )
    device_override: str | None = None

    @computed_field
    @property
    def device(self) -> str:
        """Decide the device based on CPU or CUDA."""
        if self.device_override:
            return self.device_override
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @computed_field
    @property
    def compute_type(self) -> str:
        """Determine the compute type (float16/int8) based on device."""
        if self.device == "cuda":
            return "float16"  # SOTA precision
        return "int8"

    @computed_field
    @property
    def device_index(self) -> list[int]:
        """List of available device indices."""
        if self.device == "cuda":
            import torch
            return list(range(torch.cuda.device_count()))
        return []

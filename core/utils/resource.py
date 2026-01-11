"""Resource management utilities for monitoring system health."""

import asyncio
from typing import Literal

import psutil

from config import settings
from core.utils.logger import get_logger

log = get_logger(__name__)

TaskType = Literal["compute", "network", "io"]


class ResourceManager:
    """Monitors system resources (CPU, RAM, Temp) and throttles execution.

    Prevents overheating or system freeze by pausing tasks when limits are exceeded.
    """

    def __init__(self) -> None:
        """Initialize the resource manager using settings from config."""
        self.enabled = settings.enable_resource_monitoring
        self._last_log = 0.0

    async def throttle_if_needed(self, task_type: TaskType = "compute"):
        """Checks system health. If unsafe, pauses execution until safe.

        Args:
            task_type:
                - 'compute': Heavy local processing (Whisper,Pyannote). Checks CPU temp.
                - 'network': API calls. Ignores Temp/CPU, checks RAM.
                - 'io': File operations. Checks RAM.
        """
        if not self.enabled:
            return

        # We assume network tasks (API calls) don't heat up the CPU
        # But we still check RAM to prevent OOM crashes
        check_thermal = (task_type == "compute")

        while not self._is_safe(check_thermal=check_thermal):
            log.warning(
                f"System throttled! Cooling down for {settings.cool_down_seconds}s.. "
                f"({self._get_status_string()})"
            )
            await asyncio.sleep(settings.cool_down_seconds)

    def _is_safe(self, check_thermal: bool = True) -> bool:
        """Returns True if system resources are within safe limits."""
        # 1. Check RAM (Always critical)
        mem = psutil.virtual_memory()
        if mem.percent > settings.max_ram_percent:
            return False
        
        # 2. Check VRAM (GPU memory) if available
        try:
            from core.utils.hardware import get_vram_usage_percent
            vram_percent = get_vram_usage_percent()
            # Use 75% threshold for VRAM to leave headroom
            if vram_percent > 75.0:
                log.warning(f"VRAM usage high: {vram_percent:.1f}%")
                return False
        except Exception:
            pass  # No GPU or import failed

        # If it's just a network call, we don't care about CPU/Temp as much
        if not check_thermal:
            return True

        # 3. Check CPU Usage
        # interval=None is non-blocking (returns usage since last call)
        if psutil.cpu_percent(interval=None) > settings.max_cpu_percent:
            return False

        # 4. Check CPU Temperature (Best Effort)
        temp = self._get_cpu_temp()
        if temp and temp > settings.max_temp_celsius:
            return False
        
        # 5. Check GPU Temperature (NVIDIA via pynvml)
        gpu_temp = self._get_gpu_temp()
        if gpu_temp and gpu_temp > 80:  # 80°C threshold for GPU
            log.warning(f"GPU overheating: {gpu_temp}°C. Throttling...")
            return False

        return True

    def _get_gpu_temp(self) -> float | None:
        """Get NVIDIA GPU temperature using pynvml (cross-platform)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return float(temp)
        except ImportError:
            return None  # pynvml not installed
        except Exception:
            return None  # No NVIDIA GPU or driver issue

    def _get_cpu_temp(self) -> float | None:
        """Attempts to fetch CPU temperature in a cross-platform way."""
        try:
            # Linux / macOS
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()  # type: ignore
                if not temps:
                    return None

                # Common sensor names
                for name in ["coretemp", "cpu_thermal", "k10temp", "zenpower"]:
                    if name in temps:
                        # Return max core temp
                        return max(entry.current for entry in temps[name])

                # Fallback: just take the highest value found anywhere
                return max(
                    entry.current
                    for entry_list in temps.values()
                    for entry in entry_list
                )

            return None
        except Exception as e:
            log.debug(f"Failed to read CPU temp: {e}")
            return None

    def _get_status_string(self) -> str:
        """Returns a debug string of current stats."""
        mem = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent()
        temp = self._get_cpu_temp()
        temp_str = f"{temp:.1f}°C" if temp else "N/A"
        
        # Add VRAM if available
        try:
            from core.utils.hardware import get_used_vram, get_available_vram
            vram_used = get_used_vram()
            vram_total = get_available_vram()
            vram_str = f" | VRAM: {vram_used:.1f}/{vram_total:.1f}GB"
        except Exception:
            vram_str = ""
        
        return f"RAM: {mem}% | CPU: {cpu}% | Temp: {temp_str}{vram_str}"

# Singleton instance
resource_manager = ResourceManager()

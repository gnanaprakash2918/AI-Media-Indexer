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

        # Network tasks exclude thermal checks but require RAM verification.
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

        # Network calls skip thermal checks
        if not check_thermal:
            return True

        # 2. Check CPU Usage
        # interval=None is non-blocking
        if psutil.cpu_percent(interval=None) > settings.max_cpu_percent:
            return False

        # 3. Check Temperature
        temp = self._get_cpu_temp()
        if temp and temp > settings.max_temp_celsius:
            return False

        return True

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

                # Fallback: use max sensor value
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
        return f"RAM: {mem}% | CPU: {cpu}% | Temp: {temp_str}"

# Singleton instance
resource_manager = ResourceManager()


def get_resource_usage() -> dict[str, float]:
    """Returns a dictionary of current system resource usage.

    Used by tests and potential status monitoring.
    """
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": mem.percent,
        "memory_used_gb": mem.used / (1024**3),
        "memory_total_gb": mem.total / (1024**3),
    }

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
                - 'compute': Heavy local processing (Whisper,Pyannote). Checks CPU temp + VRAM.
                - 'network': API calls. Checks RAM only (VRAM won't change from waiting on HTTP).
                - 'io': File operations. Checks RAM only.
        """
        if not self.enabled:
            return

        # Only compute tasks should be gated on thermals and VRAM.
        # Network tasks blocked on VRAM will spin forever because no GPU work
        # is happening that could free memory — that's the "stuck at 6%" bug.
        check_thermal = task_type == "compute"
        check_vram = task_type == "compute"

        throttle_count = 0
        max_throttle_cycles = 10  # Hard ceiling to prevent infinite blocking

        while not self._is_safe(
            check_thermal=check_thermal, check_vram=check_vram
        ):
            throttle_count += 1
            log.warning(
                f"System throttled! Cooling down for {settings.cool_down_seconds}s.. "
                f"({self._get_status_string()}) [cycle {throttle_count}/{max_throttle_cycles}]"
            )

            # After first throttle, try to clear GPU memory
            if throttle_count == 1:
                await self._clear_gpu_memory()

            # CancelledError must propagate for graceful shutdown (^C fix)
            await asyncio.sleep(settings.cool_down_seconds)

            # If stuck for too long (3+ cycles), force aggressive cleanup
            if throttle_count >= 3 and check_vram:
                log.warning(
                    "Throttle stuck! Attempting aggressive GPU cleanup..."
                )
                await self._clear_gpu_memory(aggressive=True)

            # Hard ceiling: don't block forever — log and proceed
            if throttle_count >= max_throttle_cycles:
                log.error(
                    f"Throttle exhausted after {max_throttle_cycles} cycles "
                    f"({self._get_status_string()}). Proceeding to avoid deadlock."
                )
                break

    async def _clear_gpu_memory(self, aggressive: bool = False) -> None:
        """Force clear GPU memory by unloading models and clearing cache.

        Uses RESOURCE_ARBITER for centralized model lifecycle management.

        Args:
            aggressive: If True, force unload all models. If False, only clear cache.
        """
        try:
            import gc


            import torch

            if torch.cuda.is_available():
                # First, unload models if aggressive cleanup needed
                if aggressive or self._should_aggressive_cleanup():
                    try:
                        from core.utils.resource_arbiter import RESOURCE_ARBITER

                        # Force release all to free VRAM
                        await RESOURCE_ARBITER.force_release_all()
                        log.info(
                            "Force released all models via RESOURCE_ARBITER"
                        )
                    except Exception as e:
                        log.debug(f"Could not use RESOURCE_ARBITER: {e}")

                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Log actual VRAM freed
                try:
                    from core.utils.hardware import get_vram_usage_percent

                    log.info(
                        f"GPU memory cleared. VRAM now at {get_vram_usage_percent():.1f}%"
                    )
                except Exception:
                    log.info("GPU memory cache cleared")
        except ImportError:
            pass  # torch not available
        except Exception as e:
            log.debug(f"Failed to clear GPU memory: {e}")

    def _should_aggressive_cleanup(self) -> bool:
        """Check if aggressive model unloading is needed."""
        try:
            from core.utils.hardware import get_vram_usage_percent

            # If VRAM > 80%, do aggressive cleanup
            return get_vram_usage_percent() > 80.0
        except Exception:
            return False

    def _is_safe(self, check_thermal: bool = True, check_vram: bool = True) -> bool:
        """Returns True if system resources are within safe limits.

        Args:
            check_thermal: Whether to check CPU/GPU temperature and CPU usage.
            check_vram: Whether to check GPU VRAM usage.
        """
        # 1. Check RAM (Always critical)
        mem = psutil.virtual_memory()
        if mem.percent > settings.max_ram_percent:
            self.status = f"High RAM ({mem.percent:.1f}%)"
            return False

        # 2. Check VRAM (GPU memory) if available and requested
        if check_vram:
            try:
                # Check VRAM (Global usage is safer than just local)
                from core.utils.hardware import get_global_vram_usage_percent

                vram_percent = get_global_vram_usage_percent()

                if vram_percent > settings.max_vram_percent:
                    self.status = f"High VRAM ({vram_percent:.1f}%)"
                    return False
            except Exception:
                pass  # No GPU or import failed

        # If it's just a network call, we don't care about CPU/Temp as much
        if not check_thermal:
            return True


        # 3. Check CPU Usage
        cpu_usage = psutil.cpu_percent(interval=None)
        if cpu_usage > settings.max_cpu_percent:
            self.status = f"High CPU ({cpu_usage:.1f}%)"
            return False

        # 4. Check CPU Temperature (Best Effort)
        temp = self._get_cpu_temp()
        if temp and temp > settings.max_temp_celsius:
            self.status = f"High CPU Temp ({temp:.1f}°C)"
            return False

        # 5. Check GPU Temperature (NVIDIA via pynvml)
        gpu_temp = self._get_gpu_temp()
        gpu_max = getattr(settings, "max_gpu_temp_celsius", 80)
        if gpu_temp and gpu_temp > gpu_max:
            self.status = f"High GPU Temp ({gpu_temp:.1f}°C)"
            log.warning(
                f"GPU overheating: {gpu_temp}°C (limit: {gpu_max}°C). Throttling..."
            )
            return False

        return True

    def _get_gpu_temp(self) -> float | None:
        """Get NVIDIA GPU temperature using pynvml (cross-platform)."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
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
            from core.utils.hardware import (
                get_available_vram,
                get_global_vram_usage_percent,
                get_used_vram,
            )

            vram_used = get_used_vram()
            vram_total = get_available_vram()
            global_vram_pct = get_global_vram_usage_percent()
            vram_str = f" | VRAM: {global_vram_pct:.1f}% global ({vram_used:.1f}/{vram_total:.1f}GB local)"
        except Exception:
            vram_str = ""

        return f"RAM: {mem}% | CPU: {cpu}% | Temp: {temp_str}{vram_str}"


# Singleton instance
resource_manager = ResourceManager()

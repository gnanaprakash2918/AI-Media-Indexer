import asyncio
import subprocess
import time

import psutil
from func_timeout import FunctionTimedOut, func_set_timeout

from core.utils.logger import log


class ResourceExceededError(Exception):
    pass


class MediaSandbox:
    def __init__(self, max_memory_mb: int = 2048, timeout_seconds: int = 3600):
        self.max_memory_mb = max_memory_mb
        self.timeout_seconds = timeout_seconds

    async def _watchdog(self, proc: asyncio.subprocess.Process):
        """Monitors the subprocess for memory violations on Windows/Unix."""
        try:
            ps_proc = psutil.Process(proc.pid)
            start_time = time.time()

            while proc.returncode is None:
                # 1. Memory check
                mem_info = ps_proc.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)

                if mem_mb > self.max_memory_mb:
                    log(
                        f"[Sandbox] Memory limit exceeded! {mem_mb:.1f}MB > {self.max_memory_mb}MB. Killing PID {proc.pid}."
                    )
                    proc.kill()
                    raise ResourceExceededError(
                        f"Memory limit exceeded: {mem_mb:.1f}MB"
                    )

                # 2. Timeout check
                if time.time() - start_time > self.timeout_seconds:
                    log(
                        f"[Sandbox] Timeout exceeded! {self.timeout_seconds}s. Killing PID {proc.pid}."
                    )
                    proc.kill()
                    raise ResourceExceededError(
                        f"Timeout exceeded: {self.timeout_seconds}s"
                    )

                await asyncio.sleep(1)
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            if not isinstance(e, ResourceExceededError):
                log(f"[Sandbox] Watchdog error: {e}")

    async def create_process(self, *cmd: str) -> asyncio.subprocess.Process:
        """Creates a sandboxed process, attaching a watchdog that will kill it if limits are exceeded."""
        log(f"[Sandbox] Securing stream: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            limit=10 * 1024 * 1024,  # 10MB buffer
        )

        # Attach the watchdog to the event loop. It will exit when the process finishes.
        asyncio.create_task(self._watchdog(proc))
        return proc

    async def run_command(self, cmd: list[str]) -> tuple[int, bytes, bytes]:
        """Runs a command entirely inside the resource sandbox (blocking till end)."""
        proc = await self.create_process(*cmd)
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr

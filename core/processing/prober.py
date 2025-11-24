from typing import Any, Optional
import subprocess
import json
from pathlib import Path
import shutil
import functools

def requires_ffprobe(func):
    """
    Ensure that the `ffprobe` executable is available before calling the function.

    This decorator checks whether the `ffprobe` binary is accessible in the
    system PATH. If it is not found, a `RuntimeError` is raised before the
    wrapped function executes. This is typically used to guard methods that
    rely on `ffprobe`, such as media metadata extraction.

    Args:
        func (Callable): The function or method being wrapped.

    Returns:
        Callable: The wrapped function that performs the ffprobe availability
            check before invoking the original function.

    Raises:
        RuntimeError: If `ffprobe` is not installed or not found in the PATH.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrapper that performs the ffprobe availability check."""
        if not shutil.which("ffprobe"):
            raise RuntimeError("ffprobe is not installed or not in PATH.")
        return func(self, *args, **kwargs)

    return wrapper

class MediaProbeError(Exception):
    """
    Raised when probing media metadata with ffprobe fails.

    Attributes:
        code: A short, machine-friendly error code.
        message: Human-readable description of the error.
        details: Optional extra context (stderr output, return code, etc.).

    """
    def __init__(
            self,
            message: str,
            *,
            code: str = "media_probe_error",
            details: Optional[Any] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details

        super().__init__(message)

    def __str__(self) -> str:
        """Return a readable string representation of the error."""
        base = f"[{self.code}] {self.message}"
        if self.details is not None:
            return f"{base} | details={self.details!r}"

        return base


class MediaProber:
    """
    Probe media files using ffprobe and return parsed metadata.

    This class is responsible for running `ffprobe` on the provided media file
    and returning the parsed JSON metadata describing the available streams
    and format.
    """

    @requires_ffprobe
    def probe(self, file_path: str | Path) -> dict:
        """
        Probe a media file with ffprobe and return its metadata.

        This method invokes `ffprobe` with JSON output enabled and parses the
        result into a Python dictionary.

        Args:
            file_path: Path to the media file to probe. Either a string or Path object.

        Returns:
            dict: A dictionary representing the JSON output from ffprobe. It
            typically contains keys like "streams" and "format".

        Raises:
            MediaProbeError: If ffprobe fails to execute, returns a non-zero
                exit code, or produces invalid JSON output.
        """

        path = Path(file_path)

        if isinstance(file_path, str) and file_path.strip() == "":
            raise ValueError("Provided path is empty or whitespace.")

        # Check if that path actually exists
        # Also Check if it's an empty string (as empty string is a valid POSIX path)

        if not path.exists() or file_path.strip() == '':
            raise MediaProbeError(
                f"File does not exist: {file_path}",
                code="file_not_found",
                details={"path": file_path}
            )

        args_to_ffprobe = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            file_path
        ]

        process = subprocess.Popen(
            args=args_to_ffprobe,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=False
        )

        out, err = process.communicate()

        return_code = process.returncode

        if return_code != 0:
            # ffprobe failed to execute
            raise MediaProbeError(
                "ffprobe failed to execute",
                code="media_probe_error",
                details={
                    "return_code": return_code,
                    "stderr": err.strip() if err else "",
                    "stdout": out.strip() if out else "",
                },
            )

        try:
            result_dict = json.loads(out)
        except json.JSONDecodeError as exc:
            raise MediaProbeError(
                "Failed to parse ffprobe JSON output",
                code="media_probe_error",
                details={
                    "stdout": out,
                    "stderr": err,
                },
            ) from exc

        return result_dict

if __name__ == "__main__":
    scanner = MediaProber()
    path = ""
    print(scanner.probe(path))
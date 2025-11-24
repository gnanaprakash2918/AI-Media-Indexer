from typing import Any, Optional
import subprocess
import json

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
    A class that is responsible for probing the provided media file with `ffprobe`.
    Spits out a JSON format containing the details of the available streams and format.
    """

    def probe(self, file_path: str) -> dict:
        """
        Executes
        """

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
        result_dict = json.loads(out)

        if return_code != 0:
            # ffprobe failed to execute
            raise MediaProbeError(
                "ffprobe failed to execute",
                code="media_probe_error",
                details={"error": json.loads(err)}
            )

        return result_dict
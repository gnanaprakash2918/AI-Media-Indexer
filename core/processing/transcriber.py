"""Audio transcription utilities using Faster Whisper.

This module provides a small wrapper around Faster Whisper optimized for
local execution with fallback to online download.
"""

import io
import os
import sys
from pathlib import Path


def _add_torch_libs_to_path() -> None:
    try:
        # Navigate from: core/processing/transcriber.py -> project_root -> .venv
        project_root = Path(__file__).resolve().parent.parent.parent
        torch_lib = (
            project_root / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
        )

        if torch_lib.exists():
            os.environ["PATH"] = (
                str(torch_lib) + os.pathsep + os.environ["PATH"]
            )
    except Exception:
        pass


_add_torch_libs_to_path()

import torch
from faster_whisper import WhisperModel

# Force stdout to handle UTF-8 characters properly (Fixes Windows Terminal display)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


class AudioTranscriber:
    """Audio transcriber using Faster Whisper.

    Attributes:
        model_size: Model size identifier such as "tiny", "base", "small",
            "medium", or "large-v2".
        device: Device on which the model runs, e.g. "cuda" or "cpu".
        compute_type: Compute precision used by the model, such as "float16"
            or "int8".
        model: Underlying WhisperModel instance.
    """

    def __init__(self, model_size: str = "large-v2") -> None:
        """Initialize the Faster Whisper model.

        Checks for a local manual download first. If not found, attempts
        to download from Hugging Face to the local 'models' directory.

        Args:
            model_size: One of "tiny", "base", "small", "medium", "large-v2".
        """
        self.model_size = model_size

        project_root = Path(__file__).resolve().parent.parent.parent
        self.model_root_dir = project_root / "models"
        self.model_root_dir.mkdir(exist_ok=True)

        print(f"[{self.__class__.__name__}] Models dir: {self.model_root_dir}")

        self.local_model_path = self.model_root_dir / model_size

        is_local_available = (self.local_model_path / "model.bin").exists()

        # 3. Auto-detect GPU
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print(f"[{self.__class__.__name__}] CUDA detected. Running on GPU.")
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            print(
                f"[{self.__class__.__name__}] CUDA not found. Running on CPU."
            )

        try:
            if is_local_available:
                print(
                    f"[{self.__class__.__name__}] Found local model at: {self.local_model_path}"
                )
                print(f"[{self.__class__.__name__}] Loading offline...")

                # PASSING A PATH forces offline loading
                self.model = WhisperModel(
                    str(self.local_model_path),
                    device=self.device,
                    compute_type=self.compute_type,
                )
            else:
                print(
                    f"[{self.__class__.__name__}] Local model not found at: {self.local_model_path}"
                )
                print(
                    f"[{self.__class__.__name__}] Downloading {self.model_size} from Hugging Face..."
                )

                # We set download_root so it saves to D: drive (models folder)
                self.model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(self.model_root_dir),
                )

        except Exception as exc:  # pylint: disable=broad-except
            print(f"Error loading model: {exc}")
            print(
                "Tip: If downloading, check internet. "
                "If local, check if 'model.bin' exists in 'models/large-v2/'."
            )
            raise

    def transcribe(self, audio_path: str | Path) -> dict:
        """Transcribe audio and return recognized text with timestamps.

        Args:
            audio_path: Path to an audio file readable by Faster Whisper.

        Returns:
            A dictionary with the following keys:
                text: Full transcribed text.
                segments: List of segment dictionaries containing:
                    start: Segment start time in seconds.
                    end: Segment end time in seconds.
                    text: Transcribed text for the segment.
                language: Detected language code.
                language_probability: Probability associated with the
                    detected language.

        Raises:
            FileNotFoundError: If the audio file does not exist.
        """
        if isinstance(audio_path, Path):
            audio_path = str(audio_path)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(
            f"[{self.__class__.__name__}] Transcribing: "
            f"{os.path.basename(audio_path)}..."
        )

        # Perform transcription
        # Using beam_size=5 for better accuracy
        segments_generator, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,  # remove bg noise
            vad_parameters={"min_silence_duration_ms": 500},
            task="transcribe",
        )

        full_text_pieces = []
        segments_data = []

        for segment in segments_generator:
            full_text_pieces.append(segment.text)
            segments_data.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

        return {
            "text": "".join(full_text_pieces).strip(),
            "segments": segments_data,
            "language": info.language,
            "language_probability": info.language_probability,
        }


def main() -> None:
    """Simple manual test harness for AudioTranscriber.

    Initializes the transcriber and processes a test audio file.
    """
    transcriber = AudioTranscriber(model_size="large-v2")
    test_path = r"C:\Users\Gnana Prakash M\Downloads\Music\clip.mp3"

    try:
        result = transcriber.transcribe(test_path)

        print("\n Transcription Result Summary:")
        print(
            f"Language: {result['language'].upper()} "
            f"({result['language_probability']:.2%})"
        )
        print(f"Full Text: {result['text'][:200]}...")
        print(f"Segments Count: {len(result['segments'])}")
    except FileNotFoundError:
        print(f"Test audio file not found: {test_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print("Starting AudioTranscriber Test")
    main()

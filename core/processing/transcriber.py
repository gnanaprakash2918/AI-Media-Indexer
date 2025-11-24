"""AudioTranscriber so the system can understand what is spoken in your videos (English, Tamil, etc.)."""

from pathlib import Path

from faster_whisper import WhisperModel


class AudioTranscriber:
    """Audio Transcriber using faster-whisper.

    - Defaults to model_size "small" (smaller and faster).
    - Auto-detects CUDA and chooses appropriate compute_type.
    - Returns a dict with full text, list of segments (with start/end), language info.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str | None = None,
        model_path: str | Path | None = None,
        max_download_retries: int = 3,
    ) -> None:
        """Initialize a new AudioTranscriber instance.

        Args:
        model_size: Whisper model name (e.g., "small", "base", "tiny"). Defaults to "small".
        device: "cuda" or "cpu". If None, we try to auto-detect CUDA (torch.cuda).
        model_path: Optional local path or HF repo id. If you pre-downloaded model, set this to that path.
        max_download_retries: Number of times to retry model download on transient errors.
        """
        self.model_size = model_size
        self.model_path = None if model_path is None else str(model_path)
        self.max_download_retries = max_download_retries

        if device is None:
            pass
            # device = "cuda" if (torch is not None and torch.cuda.is_available())

        # Run on GPU with FP16
        # Other option : compute_type="int8_float16", cuda
        # Or : compute_type="int8", cpu
        self.model = WhisperModel(
            self.model_size, device="cuda", compute_type="float16"
        )

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe the given audio file and return the transcribed text.

        Args:
            audio_path (str | Path): Path to the audio file to be transcribed.

        Returns:
            str: The transcribed text from the audio file. Return the full text and a list of segments (with start/end timestamps).
        """
        segments, info = self.model.transcribe(
            audio=str(audio_path), beam_size=5
        )

        print(
            f"Detected language {info.language} with probability {info.language_probability:.2f}"
        )

        for segment in segments:
            print(
                f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
            )

        full_text = ""
        for segment in segments:
            full_text += segment.text + " "
        return full_text.strip()


if __name__ == "__main__":
    transcriber = AudioTranscriber()
    audio_file = "C:\\Users\\Gnana Prakash M\\Downloads\\Music\\clip.mp3"
    transcription = transcriber.transcribe(audio_file)
    print(transcription)

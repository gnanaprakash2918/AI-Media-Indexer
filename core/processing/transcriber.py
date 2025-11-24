"""AudioTranscriber so the system can understand what is spoken in your videos (English, Tamil, etc.)."""

from pathlib import Path

from faster_whisper import WhisperModel


class AudioTranscriber:
    """Audio Transcriber using Faster Whisper model."""

    def __init__(self) -> None:
        """Initialize the AudioTranscriber with a Whisper model."""
        self.model_size = "base"

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

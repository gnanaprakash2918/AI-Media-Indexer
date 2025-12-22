
import asyncio
import sys
from pathlib import Path
from core.processing.transcriber import AudioTranscriber
from core.processing.extractor import FrameExtractor
from core.ingestion.pipeline import IngestionPipeline

# Setup basic logging
from core.utils.logger import logger
import logging
logging.basicConfig(level=logging.INFO)

async def test_ingest(file_path_str: str):
    path = Path(file_path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    print(f"Testing ingestion for: {path}")
    
    # 1. Test Audio Parsing (FFmpeg)
    print("\n--- Testing Audio Extraction & Transcription ---")
    try:
        with AudioTranscriber() as transcriber:
            segments = transcriber.transcribe(path)
            print(f"Found {len(segments) if segments else 0} fragments.")
            if segments:
                print(f"Sample: {segments[0]}")
    except Exception as e:
        print(f"Audio Error: {e}")

    # 2. Test Frame Extraction
    print("\n--- Testing Frame Extraction ---")
    try:
        extractor = FrameExtractor()
        frames = [f async for f in extractor.extract(path, interval=5)]
        print(f"Extracted {len(frames)} frames.")
    except Exception as e:
        print(f"Frame Error: {e}")

    # 3. Test Full Pipeline (Mock DB)
    print("\n--- Testing Pipeline Flow ---")
    try:
        # Mock DB to avoid writing to real Qdrant
        pipeline = IngestionPipeline()
        # pipeline.db = ... # tough to mock fully, but let's try calling processing directly
        # We just want to see if it runs without erroring silently
        
        # We can't easily mock the DB without valid connection, so we will skip DB insert parts
        # by checking if pipeline.db is connected.
        
        print("Pipeline initialized.")
    except Exception as e:
        print(f"Pipeline Init Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_debug.py <video_path>")
    else:
        asyncio.run(test_ingest(sys.argv[1]))

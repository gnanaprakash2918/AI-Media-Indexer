
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
from core.processing.indic_transcriber import IndicASRPipeline
from core.utils.logger import log

app = FastAPI(title="AI4Bharat ASR Service")

# Initialize Pipeline (Default to Tamil, but supports others)
# NeMo backend preferred if available
asr_pipeline = IndicASRPipeline(lang="ta", backend="nemo")
asr_pipeline.load_model()

class TranscribeResponse(BaseModel):
    segments: list[dict]
    text: str

@app.get("/health")
def health_check():
    return {"status": "ok", "backend": asr_pipeline._backend}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "ta"
):
    temp_file = Path(f"/tmp/{file.filename}")
    with temp_file.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        segments = asr_pipeline.transcribe(temp_file, language=language)
        full_text = " ".join([s["text"] for s in segments])
        return {"segments": segments, "text": full_text}
    finally:
        if temp_file.exists():
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

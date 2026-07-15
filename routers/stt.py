"""/stt 엔드포인트. 로직은 services/whisper_stt.py에."""
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from services import whisper_stt

router = APIRouter(tags=["stt"])


class SttResponse(BaseModel):
    text: str


@router.post("/stt", response_model=SttResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")
    text = whisper_stt.transcribe(audio.filename, audio_bytes)
    return SttResponse(text=text)

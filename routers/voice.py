"""/voicepack, /tts 엔드포인트. 로직은 services/cosyvoice_tts.py에."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services import cosyvoice_tts

router = APIRouter(tags=["voice"])


class VoicepackResponse(BaseModel):
    voicepack_id: str


class TtsRequest(BaseModel):
    text: str
    voicepack_id: str
    output_name: str | None = None  # 지정 시 messages/{output_name}.wav, 미지정 시 uuid


class TtsResponse(BaseModel):
    audio_path: str  # 데이터 디렉토리 기준 상대경로 (예: messages/xxx.wav)


@router.post("/voicepack", response_model=VoicepackResponse)
async def create_voicepack(
    audio: UploadFile = File(...),
    script: str = Form(...),
    member_id: str = Form(...),
):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="오디오 파일이 비어 있습니다.")
    voicepack_id = cosyvoice_tts.save_voicepack(member_id, script, audio.filename, audio_bytes)
    return VoicepackResponse(voicepack_id=voicepack_id)


@router.post("/tts", response_model=TtsResponse)
def create_tts(request: TtsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")
    try:
        audio_path = cosyvoice_tts.synthesize_message(
            request.text, request.voicepack_id, request.output_name
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return TtsResponse(audio_path=audio_path)

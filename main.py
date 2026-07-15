"""familog-ai FastAPI 앱. 실행: venv/bin/uvicorn main:app --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import get_settings
from routers import avatar, stt, voice
from services import cosyvoice_tts, whisper_stt

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings().ensure_dirs()
    cosyvoice_tts.load_engine()  # 무거운 모델은 시작 시 1회 로드
    whisper_stt.load_model()
    yield


app = FastAPI(title="familog-ai", lifespan=lifespan)
app.include_router(avatar.router)
app.include_router(stt.router)
app.include_router(voice.router)


@app.get("/health")
def health():
    return {"status": "ok"}

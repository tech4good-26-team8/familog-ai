"""음성 메시지 STT — openai-whisper 로컬 추론.

- 모델은 시작 시 1회 로드 (02_TECH_FLOW §라이프사이클, cosyvoice와 동일 패턴).
- base 모델(~140MB)이면 짧은 한국어 음성 메시지에 충분. WHISPER_MODEL로 교체 가능.
"""
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()  # 동시 추론 금지 (메모리 보호)


def load_model() -> None:
    """프로세스 시작 시 1회 호출 (main.py lifespan)."""
    global _model
    import whisper

    name = os.getenv("WHISPER_MODEL", "base")
    logger.info("Whisper 로드 시작: %s", name)
    _model = whisper.load_model(name)
    logger.info("Whisper 로드 완료")


def transcribe(filename: str | None, audio_bytes: bytes) -> str:
    """업로드된 음성 파일을 한국어 텍스트로 변환."""
    if _model is None:
        raise RuntimeError("Whisper 모델이 로드되지 않았습니다 (load_model 미호출)")

    suffix = Path(filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        with _lock:
            result = _model.transcribe(tmp_path, language="ko", fp16=False)
        return str(result["text"]).strip()
    finally:
        os.unlink(tmp_path)

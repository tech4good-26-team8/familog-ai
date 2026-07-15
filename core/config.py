"""환경설정. .env → 환경변수로만 읽는다 (04_CONVENTIONS §5)."""
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        # 두 서버가 공유하는 데이터 디렉토리 (05_ARCHITECTURE §6)
        self.data_dir = Path(
            os.getenv("FAMILOG_DATA_DIR", str(Path.home() / "familog-data"))
        ).expanduser()
        # CosyVoice2 모델 가중치 위치 (gitignore 대상)
        self.model_dir = Path(
            os.getenv(
                "COSYVOICE_MODEL_DIR",
                str(BASE_DIR / "pretrained_models" / "CosyVoice2-0.5B"),
            )
        ).expanduser()
        # cosyvoice | mock (mock = macOS say — 모델 없이 계약만 확인용)
        self.tts_engine = os.getenv("TTS_ENGINE", "cosyvoice")

        self.voicepack_dir = self.data_dir / "voicepacks"
        self.message_dir = self.data_dir / "messages"

    def ensure_dirs(self) -> None:
        self.voicepack_dir.mkdir(parents=True, exist_ok=True)
        self.message_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

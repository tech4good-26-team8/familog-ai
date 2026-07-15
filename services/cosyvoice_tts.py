"""보이스팩 저장 + CosyVoice2 zero-shot TTS.

- 모델은 프로세스 시작 시 1회 로드해서 재사용 (16GB 메모리 — 요청마다 로드 금지).
- 보이스팩 = 참조 오디오 + 낭독 대사를 그대로 저장 (임베딩 추출 불필요, 02_TECH_FLOW §4).
"""
import logging
import re
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from core.config import get_settings

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
_COSYVOICE_REPO = BASE_DIR / "third_party" / "CosyVoice"
for _p in (str(_COSYVOICE_REPO), str(_COSYVOICE_REPO / "third_party" / "Matcha-TTS")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

VOICEPACK_PREFIX = "vp_"
REFERENCE_WAV = "reference.wav"
SCRIPT_TXT = "script.txt"


class CosyVoiceEngine:
    """CosyVoice2-0.5B zero-shot. M2에서는 CUDA가 없어 CPU로 동작."""

    def __init__(self, model_dir: Path) -> None:
        from cosyvoice.cli.cosyvoice import CosyVoice2

        logger.info("CosyVoice2 로드 시작: %s", model_dir)
        self.model = CosyVoice2(str(model_dir))
        self.sample_rate = self.model.sample_rate
        self._lock = threading.Lock()  # 동시 추론 금지 (메모리 보호)
        logger.info("CosyVoice2 로드 완료 (sample_rate=%d)", self.sample_rate)

    def synthesize(self, text: str, ref_wav: Path, prompt_text: str, out_path: Path) -> None:
        import torch
        import torchaudio

        with self._lock:
            chunks = [
                out["tts_speech"]
                for out in self.model.inference_zero_shot(text, prompt_text, str(ref_wav))
            ]
        speech = torch.concat(chunks, dim=1)
        torchaudio.save(str(out_path), speech, self.sample_rate)


class MockEngine:
    """macOS say 기반 mock. 모델 없이 엔드포인트 계약 확인용 (TTS_ENGINE=mock)."""

    def synthesize(self, text: str, ref_wav: Path, prompt_text: str, out_path: Path) -> None:
        aiff = out_path.with_suffix(".aiff")
        subprocess.run(["say", "-v", "Yuna", "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", str(aiff), str(out_path)],
            check=True,
        )
        aiff.unlink(missing_ok=True)


_engine = None


def load_engine() -> None:
    """프로세스 시작 시 1회 호출 (main.py lifespan)."""
    global _engine
    settings = get_settings()
    if settings.tts_engine == "mock":
        _engine = MockEngine()
        logger.warning("TTS_ENGINE=mock — macOS say로 동작 (CosyVoice2 미사용)")
    else:
        _engine = CosyVoiceEngine(settings.model_dir)


def _get_engine():
    if _engine is None:
        raise RuntimeError("TTS 엔진이 로드되지 않았습니다 (load_engine 미호출)")
    return _engine


def _trim_and_normalize(ref: Path) -> None:
    """참조 오디오 앞뒤 무음 제거 + 피크 정규화.

    zero-shot은 참조의 침묵 패턴·볼륨까지 복제하므로, 무음이 긴 참조를 주면
    출력 앞에 긴 무음이 생성돼 시간을 낭비한다. 클릭성 잡음에 속지 않도록
    100ms 이상 연속 활성 구간만 말소리 시작으로 인정한다.
    """
    import torchaudio

    wav, sr = torchaudio.load(str(ref))
    mono = wav.mean(dim=0)
    frame = int(sr * 0.02)  # 20ms
    n = len(mono) // frame
    if n < 10:
        return
    rms = mono[: n * frame].reshape(n, frame).pow(2).mean(dim=1).sqrt()
    thresh = max(rms.max().item() * 0.1, 1e-4)
    active = rms > thresh
    sustained = [
        i for i in range(n - 4) if bool(active[i : i + 5].all())
    ]  # 5프레임(100ms) 연속 활성
    if not sustained:
        return
    pad = int(0.2 / 0.02)  # 앞뒤 0.2초 여유
    start = max(0, sustained[0] - pad) * frame
    end = min(n, sustained[-1] + 5 + pad) * frame
    trimmed = mono[start:end]
    peak = trimmed.abs().max().item()
    if peak > 0:
        trimmed = trimmed * min(0.9 / peak, 10.0)  # 과도한 증폭은 10배로 제한
    torchaudio.save(str(ref), trimmed.unsqueeze(0), sr)
    logger.info(
        "참조 오디오 트리밍: %.2fs → %.2fs (게인 %.1fx)",
        len(mono) / sr, len(trimmed) / sr, min(0.9 / peak, 10.0) if peak > 0 else 1.0,
    )


def save_voicepack(member_id: str, script: str, filename: str | None, audio_bytes: bytes) -> str:
    """참조 오디오+대사를 ~/familog-data/voicepacks/{member_id}/에 저장."""
    settings = get_settings()
    pack_dir = settings.voicepack_dir / member_id
    pack_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename or "").suffix.lower() or ".wav"
    source = pack_dir / f"source{suffix}"
    source.write_bytes(audio_bytes)

    # CosyVoice 입력용으로 16kHz mono WAV 정규화 (m4a 등 녹음 포맷 대응, ffmpeg — macOS/리눅스 공통)
    ref = pack_dir / REFERENCE_WAV
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(source), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(ref)],
        capture_output=True,
    )
    if result.returncode != 0:
        if suffix == ".wav":
            shutil.copyfile(source, ref)  # 변환 실패해도 wav면 원본 그대로 사용
            logger.warning("ffmpeg 실패, 원본 wav 사용: %s", result.stderr.decode(errors="replace"))
        else:
            raise ValueError(f"참조 오디오 변환 실패: {result.stderr.decode(errors='replace')}")

    _trim_and_normalize(ref)

    (pack_dir / SCRIPT_TXT).write_text(script.strip(), encoding="utf-8")
    return f"{VOICEPACK_PREFIX}{member_id}"


def synthesize_message(text: str, voicepack_id: str, output_name: str | None = None) -> str:
    """저장된 보이스팩으로 zero-shot 낭독 생성 → data_dir 기준 상대경로 반환."""
    settings = get_settings()
    member_id = voicepack_id.removeprefix(VOICEPACK_PREFIX)
    pack_dir = settings.voicepack_dir / member_id
    ref = pack_dir / REFERENCE_WAV
    script_path = pack_dir / SCRIPT_TXT
    if not ref.exists() or not script_path.exists():
        raise FileNotFoundError(f"보이스팩이 없습니다: {voicepack_id}")

    name = _sanitize(output_name) if output_name else uuid.uuid4().hex
    out_path = settings.message_dir / f"{name}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _get_engine().synthesize(text, ref, script_path.read_text(encoding="utf-8"), out_path)
    return str(out_path.relative_to(settings.data_dir))


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not cleaned:
        raise ValueError(f"잘못된 output_name: {name}")
    return cleaned

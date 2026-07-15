"""CosyVoice2 M2 구동 검증 스크립트 (02_TECH_FLOW §5-2 3번 관문).

사용: venv/bin/python scripts/verify_cosyvoice.py <참조wav> <참조대사> <낭독텍스트> <출력wav>
"""
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "third_party" / "CosyVoice"))
sys.path.insert(0, str(BASE_DIR / "third_party" / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from cosyvoice.cli.cosyvoice import CosyVoice2  # noqa: E402


def main() -> None:
    ref_wav, prompt_text, tts_text, out_wav = sys.argv[1:5]
    model_dir = BASE_DIR / "pretrained_models" / "CosyVoice2-0.5B"

    t0 = time.time()
    model = CosyVoice2(str(model_dir))
    print(f"[verify] 모델 로드: {time.time() - t0:.1f}s, device={model.model.device}")

    t1 = time.time()
    chunks = [o["tts_speech"] for o in model.inference_zero_shot(tts_text, prompt_text, ref_wav)]
    speech = torch.concat(chunks, dim=1)
    elapsed = time.time() - t1
    audio_sec = speech.shape[1] / model.sample_rate

    torchaudio.save(out_wav, speech, model.sample_rate)
    print(f"[verify] 생성 {elapsed:.1f}s / 오디오 {audio_sec:.1f}s (RTF {elapsed / audio_sec:.2f}) → {out_wav}")


if __name__ == "__main__":
    main()

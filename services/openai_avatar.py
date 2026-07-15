import base64
from pathlib import Path

from openai import OpenAI

from core.config import get_settings

# 02_TECH_FLOW.md §8-2 결정: 3D 메시가 아닌 "3D풍 PNG", Memoji/AR이모지 느낌의 정적 캐릭터.
AVATAR_PROMPT = (
    "Turn this person's photo into a cute 3D-style iphone memoji/AR-emoji character portrait with only the face without neck. "
    "Keep recognizable facial features (hair, glasses, expression), soft 3D shading, " 
    "no background, front-facing, head especially only face framing without neck."
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=get_settings().openai_api_key)
    return _client


def generate_avatar(photo_bytes: bytes, photo_filename: str, member_id: str) -> str:
    """사진을 3D 캐릭터 PNG로 변환해 저장하고, data_dir 기준 상대 경로를 반환한다."""
    client = _get_client()

    response = client.images.edit(
        model="gpt-image-1",
        image=(photo_filename, photo_bytes),
        prompt=AVATAR_PROMPT,
        size="1024x1024",
    )
    image_bytes = base64.b64decode(response.data[0].b64_json)

    settings = get_settings()
    settings.avatars_dir.mkdir(parents=True, exist_ok=True)

    absolute_path = settings.avatars_dir / f"{member_id}.png"
    absolute_path.write_bytes(image_bytes)

    return str(Path("avatars") / f"{member_id}.png")

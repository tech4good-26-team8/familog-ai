# 아바타 생성 (`POST /avatar`) 구현

`familog-server/docs`의 `02_TECH_FLOW.md §4`, `03_API_SPEC.md`, `05_ARCHITECTURE.md §4·§6` 계약을 따라 구현.

## 구조

```
familog-ai/
├── main.py                    # FastAPI 앱 생성, /health, avatar 라우터 등록
├── routers/
│   └── avatar.py               # POST /avatar 엔드포인트 (요청 검증만)
├── services/
│   └── openai_avatar.py        # OpenAI 이미지 API 호출 + PNG 저장 로직
├── core/
│   └── config.py                # .env 로딩 (OPENAI_API_KEY, FAMILOG_DATA_DIR)
└── .env.example
```

## 계약

- **입력**: multipart/form-data — `photo`(파일), `member_id`(문자열)
- **출력**: `{"avatar_path": "avatars/{member_id}.png"}` (data_dir 기준 상대 경로, snake_case)
- **실패**: 500 + `{"detail": "..."}` → server가 `avatarStatus`를 `FAILED`로 전이 (05 §2)
- **스타일**: 프롬프트 고정, "3D 메시가 아닌 3D풍 PNG, Memoji/AR이모지 느낌의 정적 캐릭터" (02 §8-1, §8-2)

## 환경변수

| 변수 | 설명 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI 이미지 API 키 | (필수, `.env`) |
| `FAMILOG_DATA_DIR` | 생성 파일 저장 위치. server와 공유하는 `~/familog-data`를 가리키도록 설정 (05 §6) | `./data` (로컬 개발용) |

## 검증

- `uvicorn main:app`로 기동 후 `GET /health` → `200 {"status":"ok"}`
- `POST /avatar` (photo 누락) → `422` (FastAPI 자동 검증)
- `POST /avatar` (photo+member_id, `OPENAI_API_KEY` 미설정) → `500` — 계약대로 실패가 500으로 전파되는 것 확인. 실제 이미지 생성 자체는 유효한 API 키 필요해 로컬에서 별도 확인 필요.

## 범위 밖

- `/voicepack`, `/tts`, `/stt`는 이번 변경에 포함되지 않음 (03_API_SPEC.md 기준 여전히 "예정").
- `main.py`의 `/health`는 avatar 라우터가 동작하려면 앱 골격이 필요해 최소로 함께 추가함.

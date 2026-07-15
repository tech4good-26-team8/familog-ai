from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel

from services.openai_avatar import generate_avatar

router = APIRouter()


class AvatarResponse(BaseModel):
    avatar_path: str


@router.post("/avatar", response_model=AvatarResponse)
async def create_avatar(photo: UploadFile, member_id: str = Form(...)) -> AvatarResponse:
    photo_bytes = await photo.read()
    try:
        relative_path = generate_avatar(photo_bytes, photo.filename or "photo.png", member_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return AvatarResponse(avatar_path=relative_path)

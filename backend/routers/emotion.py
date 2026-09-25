"""Emotion recognition endpoint."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.emotion_service import detect_emotion

router = APIRouter(tags=["agents"])
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@router.post("/emotion")
async def emotion(image: UploadFile = File(...)):
    """Take a frame (JPEG/PNG); return the dominant emotion and scores."""
    data = await image.read()
    if not data:
        raise HTTPException(422, "La imagen está vacía")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            413, "La imagen es demasiado grande (máximo 8 MB)"
        )
    return await detect_emotion(data)

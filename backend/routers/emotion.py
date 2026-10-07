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
        raise HTTPException(422, "The image is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            413, "The image is too large (8 MB at most)"
        )
    return await detect_emotion(data)

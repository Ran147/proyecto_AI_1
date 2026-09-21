from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.emotion_service import detect_emotion

router = APIRouter(tags=["agentes"])
MAX_BYTES = 8 * 1024 * 1024


@router.post("/emotion")
async def emotion(image: UploadFile = File(...)):
    """Recibe un frame (JPEG/PNG) y devuelve la emoción dominante y los puntajes."""
    data = await image.read()
    if not data:
        raise HTTPException(422, "La imagen está vacía")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "La imagen es demasiado grande (máximo 8 MB)")
    return await detect_emotion(data)

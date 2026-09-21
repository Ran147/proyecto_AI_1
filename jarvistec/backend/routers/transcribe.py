from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.speech_service import transcribe as run_transcribe

router = APIRouter(tags=["agentes"])
MAX_BYTES = 10 * 1024 * 1024  # la API síncrona de Google admite ~1 minuto / 10 MB


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Recibe audio (webm/ogg/wav) y devuelve el texto transcrito."""
    data = await audio.read()
    if not data:
        raise HTTPException(422, "El audio está vacío")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "El audio es demasiado largo (máximo ~1 minuto)")
    return await run_transcribe(data, audio.content_type or "")

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(tags=["agentes"])


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """PENDIENTE (paso 5): recibe audio y llama a Google Speech-to-Text."""
    raise HTTPException(501, "Transcripción aún no implementada")

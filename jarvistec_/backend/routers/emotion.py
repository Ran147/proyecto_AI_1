from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(tags=["agentes"])


@router.post("/emotion")
async def emotion(image: UploadFile = File(...)):
    """PENDIENTE (paso 5): recibe un frame y llama a Azure Face / Google Vision."""
    raise HTTPException(501, "Detección de emociones aún no implementada")

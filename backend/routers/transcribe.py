"""Speech-to-text endpoint."""
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.speech_service import transcribe as run_transcription

router = APIRouter(tags=["agents"])
# Google's synchronous API accepts about one minute / 10 MB of audio.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Take audio and return the transcribed text.

    The local model takes WAV (16-bit PCM); Google also takes webm/ogg.
    """
    data = await audio.read()
    if not data:
        raise HTTPException(422, "The audio is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            413, "The audio is too long (about 1 minute at most)"
        )
    return await run_transcription(data, audio.content_type or "")

"""Speech-to-text endpoint."""
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..services.speech_service import transcribe as run_transcription

router = APIRouter(tags=["agents"])
# Google's synchronous API accepts about one minute / 10 MB of audio.
MAX_AUDIO_BYTES = 10 * 1024 * 1024


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
    engine: str | None = Form(None),
):
    """Take audio and return the transcribed text.

    The local model takes WAV (16-bit PCM); Google also takes webm/ogg.
    ``language`` is "en" (default) or "es". ``engine`` optionally picks
    the server engine: "local" (Vosk), "google", or "scratch" (our model,
    only when SPEECH_PROVIDER=scratch).
    """
    data = await audio.read()
    if not data:
        raise HTTPException(422, "The audio is empty")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(
            413, "The audio is too long (about 1 minute at most)"
        )
    return await run_transcription(
        data, audio.content_type or "", language, engine
    )

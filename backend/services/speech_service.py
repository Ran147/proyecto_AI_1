"""Speech-to-text on the server.

Single entry point: ``transcribe(audio_bytes, content_type, language)``.
Output: ``{"text": "...", "confidence": 0.93, "provider": "local",
"language": "en"}``. Commands are understood in English ("en") and Spanish
("es").

Providers:

- ``local``: Vosk, an offline speech recognizer (Kaldi acoustic model), with
  its small English and Spanish models stored in artifacts/. It needs no API
  key and no internet connection. It takes WAV audio (16-bit PCM), which is
  what the web interface records.
- ``google``: Google Cloud Speech-to-Text (REST v1, needs GOOGLE_API_KEY).
- ``browser``: the server has no speech engine (no local model, no key).
  The web interface transcribes with the browser's Web Speech API and sends
  only the text to ``POST /command``; ``POST /transcribe`` answers 503 with
  that explanation.
- ``mock``: canned text, only when SPEECH_PROVIDER=mock is set explicitly
  (development and tests).
"""
import asyncio
import base64
import io
import json
import os
import threading
import wave
from pathlib import Path

import httpx
from fastapi import HTTPException

from ..core.orchestrator import get_orchestrator

SPEECH_URL = "https://speech.googleapis.com/v1/speech:recognize"
# Google accepts at most 500 hint phrases per request.
MAX_HINT_PHRASES = 500


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
# Vosk model folder and Google language code of each supported language.
LOCAL_MODEL_DIRS = {
    "en": ARTIFACTS_DIR / "vosk-model-small-en-us-0.15",
    "es": ARTIFACTS_DIR / "vosk-model-small-es-0.42",
}
GOOGLE_LANGUAGES = {"en": "en-US", "es": "es-US"}
DEFAULT_LANGUAGE = "en"
# Sample rate the Vosk model was trained on.
LOCAL_SAMPLE_RATE = 16000
# Longest recording the local model accepts, in seconds.
MAX_LOCAL_SECONDS = 60

NO_PROVIDER_MESSAGE = (
    "No speech-to-text provider is available on the server: the local "
    "model is missing (run scripts/download_speech_models.py) and "
    "GOOGLE_API_KEY is not set. The web interface can still transcribe in "
    "the browser and send the text to POST /command."
)
NOT_UNDERSTOOD_MESSAGE = (
    "I could not understand the audio. Speak closer to the microphone and "
    "try again."
)


def local_model_available(language: str = DEFAULT_LANGUAGE) -> bool:
    """Tell whether the local speech model of ``language`` is in place."""
    folder = LOCAL_MODEL_DIRS.get(language)
    return folder is not None and (folder / "am" / "final.mdl").exists()


def check_language(language: str | None) -> str:
    """Return a supported language code; unknown ones answer 422."""
    code = (language or DEFAULT_LANGUAGE).lower()[:2]
    if code not in LOCAL_MODEL_DIRS:
        raise HTTPException(
            422, f"Unsupported language '{language}': use 'en' or 'es'"
        )
    return code


def provider_name() -> str:
    """Return the active provider, or the explicit SPEECH_PROVIDER.

    In ``auto`` mode: google with GOOGLE_API_KEY, otherwise the local model
    when its files are present, otherwise browser (no engine on the server).
    """
    choice = (os.getenv("SPEECH_PROVIDER") or "auto").lower()
    if choice != "auto":
        return choice
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return "local" if local_model_available() else "browser"


def _encoding_for(content_type: str) -> str | None:
    """Map the browser recording format to a Google encoding.

    Chrome, Edge and Firefox record webm/opus. Safari records mp4, which
    Google does not accept.
    """
    content_type = (content_type or "").lower()
    if "webm" in content_type:
        return "WEBM_OPUS"
    if "ogg" in content_type:
        return "OGG_OPUS"
    # wav/flac: Google detects the encoding from the file header.
    return None


def build_config(
    content_type: str, language: str = DEFAULT_LANGUAGE
) -> dict:
    """Build the recognition config, including command hint phrases."""
    # Hint phrases greatly improve recognition of "JarvisTEC" and commands.
    phrases = [
        "JarvisTEC",
        "Jarvis TEC",
        *get_orchestrator().command_phrases(),
    ]
    config = {
        # SPEECH_LANGUAGE overrides the code used for English commands.
        "languageCode": (
            GOOGLE_LANGUAGES["es"]
            if language == "es"
            else os.getenv("SPEECH_LANGUAGE") or GOOGLE_LANGUAGES["en"]
        ),
        "enableAutomaticPunctuation": False,
        "maxAlternatives": 1,
        "speechContexts": [
            {"phrases": phrases[:MAX_HINT_PHRASES], "boost": 15}
        ],
    }
    encoding = _encoding_for(content_type)
    if encoding:
        config["encoding"] = encoding
    return config


async def _call_speech(
    audio_bytes: bytes, content_type: str, language: str = DEFAULT_LANGUAGE
) -> dict:
    """HTTP call to Google Speech-to-Text (isolated for tests)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "GOOGLE_API_KEY is missing in .env")
    payload = {
        "config": build_config(content_type, language),
        "audio": {"content": base64.b64encode(audio_bytes).decode()},
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                SPEECH_URL, params={"key": key}, json=payload
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"Could not reach Google Speech-to-Text: {exc}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            502, "Google Speech-to-Text returned an unreadable response"
        ) from exc


def parse_speech_response(data: dict) -> dict:
    """Convert a raw Google Speech-to-Text response to the project format."""
    if "error" in data:
        message = data["error"].get("message", "unknown error")
        raise HTTPException(502, f"Google Speech-to-Text: {message}")
    results = data.get("results") or []
    alternatives = results[0].get("alternatives") if results else None
    if not alternatives:
        raise HTTPException(422, NOT_UNDERSTOOD_MESSAGE)
    best = alternatives[0]
    return {
        "text": best["transcript"].strip(),
        "confidence": best.get("confidence"),
        "provider": "google",
    }


async def _transcribe_google(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    raw = await _call_speech(audio_bytes, content_type, language)
    return parse_speech_response(raw)


async def _transcribe_mock(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    """Simulated provider: returns MOCK_TRANSCRIPT and ignores the audio."""
    text = os.getenv("MOCK_TRANSCRIPT") or "JarvisTEC bitcoin price"
    return {"text": text, "confidence": None, "provider": "mock"}


async def _transcribe_browser(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    """No engine on the server: say so instead of inventing a text."""
    raise HTTPException(503, NO_PROVIDER_MESSAGE)


def read_wav(audio_bytes: bytes) -> bytes:
    """Return 16 kHz mono 16-bit PCM from a WAV file of any rate or layout."""
    import numpy as np

    try:
        with wave.open(io.BytesIO(audio_bytes)) as wav:
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise HTTPException(
            415,
            "The local speech model needs WAV audio (16-bit PCM). The web "
            "interface records it in that format.",
        ) from exc
    if width != 2 or channels < 1 or rate < 8000:
        raise HTTPException(
            415, "The WAV audio must be 16-bit PCM at 8 kHz or more."
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples[: len(samples) // channels * channels]
        samples = samples.reshape(-1, channels).mean(axis=1)
    if len(samples) > rate * MAX_LOCAL_SECONDS:
        raise HTTPException(
            413, f"The audio is too long ({MAX_LOCAL_SECONDS} s at most)"
        )
    if len(samples) < rate // 10:
        raise HTTPException(422, NOT_UNDERSTOOD_MESSAGE)
    if rate != LOCAL_SAMPLE_RATE:
        # Linear interpolation is enough for speech at these rates.
        length = int(len(samples) * LOCAL_SAMPLE_RATE / rate)
        samples = np.interp(
            np.linspace(0, len(samples) - 1, length),
            np.arange(len(samples)),
            samples,
        )
    return samples.astype(np.int16).tobytes()


class LocalSpeechModel:
    """Vosk recognizer over one of the small models in artifacts/."""

    def __init__(self, language: str) -> None:
        # Imported here so the API still starts without Vosk installed.
        import vosk

        vosk.SetLogLevel(-1)
        self.vosk = vosk
        self.model = vosk.Model(str(LOCAL_MODEL_DIRS[language]))

    def transcribe(self, audio_bytes: bytes) -> dict:
        pcm = read_wav(audio_bytes)
        # A recognizer keeps state: one per request; the model is shared.
        recognizer = self.vosk.KaldiRecognizer(self.model, LOCAL_SAMPLE_RATE)
        recognizer.SetWords(True)
        recognizer.AcceptWaveform(pcm)
        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            raise HTTPException(422, NOT_UNDERSTOOD_MESSAGE)
        words = result.get("result") or []
        confidence = (
            round(sum(w["conf"] for w in words) / len(words), 4)
            if words
            else None
        )
        return {"text": text, "confidence": confidence, "provider": "local"}


_local_models: dict[str, LocalSpeechModel] = {}
_local_model_lock = threading.Lock()


def _get_local_model(language: str) -> LocalSpeechModel:
    """Load the local model of ``language`` once, on first use."""
    if language not in _local_models:
        with _local_model_lock:
            if language not in _local_models:
                if not local_model_available(language):
                    raise HTTPException(503, NO_PROVIDER_MESSAGE)
                try:
                    _local_models[language] = LocalSpeechModel(language)
                except ImportError as exc:
                    raise HTTPException(
                        503,
                        "Vosk is missing. Install the dependencies in "
                        "requirements.txt.",
                    ) from exc
    return _local_models[language]


async def _transcribe_local(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    # CPU-bound (loading and decoding): run outside the event loop.
    def run() -> dict:
        return _get_local_model(language).transcribe(audio_bytes)

    return await asyncio.to_thread(run)


PROVIDERS = {
    "local": _transcribe_local,
    "google": _transcribe_google,
    "browser": _transcribe_browser,
    "mock": _transcribe_mock,
}


async def transcribe(
    audio_bytes: bytes, content_type: str = "", language: str | None = None
) -> dict:
    """Transcribe audio with the active provider, in ``language``."""
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"Invalid SPEECH_PROVIDER: {name}")
    code = check_language(language)
    result = await PROVIDERS[name](audio_bytes, content_type, code)
    return {**result, "language": code}

"""Speech-to-text on the server.

Single entry point: ``transcribe(audio_bytes, content_type, language)``.
Output: ``{"text": "...", "confidence": 0.93, "provider": "local",
"language": "en"}``. Commands are understood in English ("en") and Spanish
("es").

Providers (engines):

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
- ``scratch``: our own keyword spotter, a convolutional network trained from
  random initialization on Speech Commands (training/speech_commands/). It
  recognizes the spoken number of a model ("JarvisTEC, three" runs the third
  model), in English, not free sentences, and answers with that model's
  command text, so the rest of the pipeline is the same. Optional: it is
  used only with SPEECH_PROVIDER=scratch.

The engine is chosen by SPEECH_PROVIDER (``auto``: google with a key,
otherwise local); a request can ask for another available engine with
``engine``.
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
# Our keyword spotter and its feature settings (written by the notebook).
SCRATCH_MODEL = ARTIFACTS_DIR / "speech_commands.onnx"
SCRATCH_META = ARTIFACTS_DIR / "speech_commands.meta.json"
DIGIT_WORDS = (
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
)
NO_NUMBER_MESSAGE = (
    "I did not hear the number of a model. Say JarvisTEC and a number from "
    "one to nine, or zero for the tenth model."
)

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


def scratch_model_available() -> bool:
    """Tell whether our keyword spotter and its settings are in place."""
    return SCRATCH_MODEL.exists() and SCRATCH_META.exists()


def command_numbers() -> dict[str, str]:
    """Spoken number of each model for the scratch engine.

    "one" is the first registered model, ..., "nine" the ninth and "zero"
    the tenth, like the keys of a phone.
    """
    order = DIGIT_WORDS[1:] + DIGIT_WORDS[:1]
    return dict(zip(order, get_orchestrator().names()))


def available_engines() -> list[str]:
    """Server engines that can transcribe right now.

    Our scratch model is offered only when SPEECH_PROVIDER=scratch.
    """
    engines = []
    configured = (os.getenv("SPEECH_PROVIDER") or "auto").lower()
    if configured == "scratch" and scratch_model_available():
        engines.append("scratch")
    if local_model_available():
        engines.append("local")
    if os.getenv("GOOGLE_API_KEY"):
        engines.append("google")
    return engines


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

    In ``auto`` mode: google with GOOGLE_API_KEY, otherwise the local
    (Vosk) model when its files are present, otherwise browser (no engine on
    the server). Our scratch model is never chosen automatically.
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


def mel_filterbank(sample_rate, n_fft, n_mels, low_hz):
    """Triangular mel filters (HTK formula), shape (n_mels, n_fft // 2 + 1).

    Same function as in training/speech_commands/notebook.ipynb; keep both
    identical.
    """
    import numpy as np

    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    edges = to_hz(
        np.linspace(to_mel(low_hz), to_mel(sample_rate / 2), n_mels + 2)
    )
    bins = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    bank = np.zeros((n_mels, len(bins)), np.float32)
    for m in range(n_mels):
        left, center, right = edges[m], edges[m + 1], edges[m + 2]
        rising = (bins - left) / (center - left)
        falling = (right - bins) / (right - center)
        bank[m] = np.maximum(0, np.minimum(rising, falling))
    return bank


class ScratchSpeechModel:
    """Our keyword spotter (ONNX, run with OpenCV) over sliding windows.

    A one-second window moves over the recording every 100 ms; every window
    is classified and the digit with the highest probability is taken when
    it reaches the threshold chosen in the notebook.
    """

    def __init__(self) -> None:
        # Imported here so the API still starts without OpenCV installed.
        import cv2
        import numpy as np

        self.np = np
        meta = json.loads(SCRATCH_META.read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        self.settings = meta["features"]
        self.threshold = meta["detection"]["threshold"]
        self.step = meta["detection"]["step_samples"]
        f = self.settings
        self.mel = mel_filterbank(
            f["sample_rate"], f["n_fft"], f["n_mels"], f["mel_low_hz"]
        )
        window = np.arange(f["window"])
        self.hann = (
            0.5 - 0.5 * np.cos(2 * np.pi * window / f["window"])
        ).astype(np.float32)
        self.network = cv2.dnn.readNetFromONNX(str(SCRATCH_MODEL))
        # OpenCV networks are not safe to run from two threads at once.
        self.lock = threading.Lock()

    def log_mel(self, samples):
        """Log-mel spectrogram of one window; same formula as the notebook."""
        np, f = self.np, self.settings
        count = 1 + (len(samples) - f["window"]) // f["hop"]
        index = (
            np.arange(f["window"])[None]
            + f["hop"] * np.arange(count)[:, None]
        )
        spectrum = np.abs(
            np.fft.rfft(samples[index] * self.hann, f["n_fft"])
        ) ** 2
        return np.log(spectrum @ self.mel.T + 1e-6).T.astype(np.float32)

    def detect(self, pcm: bytes) -> tuple[str | None, float]:
        """Most confident spoken digit in the audio, or None."""
        np, f = self.np, self.settings
        audio = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
        size = f["clip_samples"]
        if len(audio) < size:
            audio = np.pad(audio, (0, size - len(audio)))
        starts = range(0, len(audio) - size + 1, self.step)
        batch = np.stack([
            (self.log_mel(audio[s:s + size]) - f["mean"]) / f["std"]
            for s in starts
        ])[:, None].astype(np.float32)
        with self.lock:
            self.network.setInput(batch)
            logits = self.network.forward().reshape(len(batch), -1)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        digits = [self.classes.index(word) for word in DIGIT_WORDS]
        scores = probabilities[:, digits]
        window, position = np.unravel_index(scores.argmax(), scores.shape)
        best = float(scores[window, position])
        word = DIGIT_WORDS[position] if best >= self.threshold else None
        return word, round(best, 4)

    def transcribe(self, audio_bytes: bytes) -> dict:
        word, confidence = self.detect(read_wav(audio_bytes))
        if word is None:
            raise HTTPException(422, NO_NUMBER_MESSAGE)
        name = command_numbers()[word]
        phrase = get_orchestrator().describe(name)["keywords"][0]
        return {
            "text": f"JarvisTEC {phrase}",
            "confidence": confidence,
            "provider": "scratch",
            "heard": word,
        }


_scratch_model: ScratchSpeechModel | None = None
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


def _get_scratch_model() -> ScratchSpeechModel:
    """Load our keyword spotter once, on first use."""
    global _scratch_model
    if _scratch_model is None:
        with _local_model_lock:
            if _scratch_model is None:
                if not scratch_model_available():
                    raise HTTPException(
                        503,
                        "Our speech model is missing: run "
                        "training/speech_commands/notebook.ipynb.",
                    )
                try:
                    _scratch_model = ScratchSpeechModel()
                except ImportError as exc:
                    raise HTTPException(
                        503,
                        "OpenCV is missing. Install the dependencies in "
                        "requirements.txt.",
                    ) from exc
    return _scratch_model


async def _transcribe_scratch(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    """Our keyword spotter; it listens for English numbers in any case."""
    def run() -> dict:
        return _get_scratch_model().transcribe(audio_bytes)

    return await asyncio.to_thread(run)


async def _transcribe_local(
    audio_bytes: bytes, content_type: str, language: str
) -> dict:
    # CPU-bound (loading and decoding): run outside the event loop.
    def run() -> dict:
        return _get_local_model(language).transcribe(audio_bytes)

    return await asyncio.to_thread(run)


PROVIDERS = {
    "scratch": _transcribe_scratch,
    "local": _transcribe_local,
    "google": _transcribe_google,
    "browser": _transcribe_browser,
    "mock": _transcribe_mock,
}


async def transcribe(
    audio_bytes: bytes,
    content_type: str = "",
    language: str | None = None,
    engine: str | None = None,
) -> dict:
    """Transcribe audio in ``language`` with ``engine`` or the default one."""
    name = provider_name()
    if engine:
        name = engine.lower()
        if name != provider_name() and name not in available_engines():
            options = ", ".join(available_engines()) or "none"
            raise HTTPException(
                422,
                f"Speech engine '{engine}' is not available "
                f"(available: {options})",
            )
    if name not in PROVIDERS:
        raise HTTPException(500, f"Invalid SPEECH_PROVIDER: {name}")
    code = check_language(language)
    result = await PROVIDERS[name](audio_bytes, content_type, code)
    return {**result, "language": code}

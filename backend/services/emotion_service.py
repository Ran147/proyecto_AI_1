"""Emotion detection from an image.

Single entry point: ``detect_emotion(image_bytes)``. Providers:

- ``local``: runs in the backend with OpenCV. YuNet finds the face and two
  pretrained networks classify its expression: FER+ (grayscale, 64x64) and
  HSEmotion (EfficientNet-B0 trained on AffectNet, color, 224x224). Their
  probabilities are averaged, which recognizes sadness, anger and fear better
  than FER+ alone (see scripts/evaluate_emotion_model.py). No API key; it
  needs the ONNX files fetched by ``scripts/download_emotion_models.py``.
- ``google``: Google Cloud Vision face detection (needs GOOGLE_API_KEY).
- ``mock``: random answer, for development only.

To add a provider, write a ``_detect_<name>`` function and register it in
PROVIDERS.

Normalized output (the format the frontend expects)::

    {"emotion": "happiness", "scores": {"happiness": 0.95, ...},
     "provider": "local"}
"""
import asyncio
import base64
import os
import random
import threading
from pathlib import Path

import httpx
from fastapi import HTTPException

MODELS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
FACE_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = MODELS_DIR / "emotion-ferplus-8.onnx"
HSEMOTION_MODEL = MODELS_DIR / "enet_b0_8_best_afew.onnx"
# Output order of the FER+ network.
FERPLUS_CLASSES = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)
# FER+ takes a 64x64 grayscale face with pixel values 0-255.
FERPLUS_SIZE = 64
# Output order of the HSEmotion network.
HSEMOTION_CLASSES = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
# HSEmotion takes a 224x224 RGB face normalized with the ImageNet statistics.
HSEMOTION_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Minimum YuNet confidence to accept a detection as a face.
FACE_THRESHOLD = 0.6
# Frames are downscaled to this width before detection; faces stay large
# enough and detection gets faster.
MAX_DETECTION_WIDTH = 640
NO_FACE_MESSAGE = (
    "No face was detected. Look at the camera, in good light."
)
UNREADABLE_IMAGE_MESSAGE = "Could not read the image. Send a JPEG or PNG."
# Frames smaller than this on either side cannot contain a usable face.
MIN_IMAGE_SIDE = 32

VISION_URL = "https://vision.googleapis.com/v1/images:annotate"

# Google Vision returns likelihood levels, not probabilities; they are
# mapped to a 0-1 score.
LIKELIHOOD_SCORES = {
    "UNKNOWN": 0.0,
    "VERY_UNLIKELY": 0.03,
    "UNLIKELY": 0.2,
    "POSSIBLE": 0.5,
    "LIKELY": 0.8,
    "VERY_LIKELY": 0.95,
}
# Google field -> normalized project emotion name.
VISION_FIELDS = {
    "joyLikelihood": "happiness",
    "sorrowLikelihood": "sadness",
    "angerLikelihood": "anger",
    "surpriseLikelihood": "surprise",
}
# Below "POSSIBLE" for every emotion, the face is considered neutral.
NEUTRAL_THRESHOLD = 0.5


def local_models_available() -> bool:
    """Tell whether the ONNX files the local provider needs are in place.

    HSEmotion is optional: without it only FER+ classifies.
    """
    return FACE_MODEL.exists() and EMOTION_MODEL.exists()


def local_classifiers() -> list[str]:
    """Return the classifiers the local provider uses: EMOTION_MODEL env.

    ``ensemble`` (default) averages FER+ and HSEmotion; ``ferplus`` and
    ``hsemotion`` use a single network.
    """
    choice = os.getenv("EMOTION_MODEL", "ensemble").lower()
    wanted = {
        "ferplus": ["ferplus"],
        "hsemotion": ["hsemotion"],
    }.get(choice, ["ferplus", "hsemotion"])
    if not HSEMOTION_MODEL.exists():
        return ["ferplus"]
    return wanted


def provider_name() -> str:
    """Return the active provider, or the explicit EMOTION_PROVIDER.

    In ``auto`` mode: google with GOOGLE_API_KEY, otherwise the local model
    when its files are present, otherwise mock.
    """
    choice = (os.getenv("EMOTION_PROVIDER") or "auto").lower()
    if choice != "auto":
        return choice
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    return "local" if local_models_available() else "mock"


async def _call_vision(image_bytes: bytes) -> dict:
    """HTTP call to Google Vision (isolated so tests can replace it)."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(500, "GOOGLE_API_KEY is missing in .env")
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(image_bytes).decode()},
            "features": [{"type": "FACE_DETECTION", "maxResults": 1}],
        }]
    }
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                VISION_URL, params={"key": key}, json=payload
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, f"Could not reach Google Vision: {exc}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            502, "Google Vision returned an unreadable response"
        ) from exc


def _vision_error(error: dict) -> HTTPException:
    message = error.get("message", "unknown error")
    return HTTPException(502, f"Google Vision: {message}")


def parse_vision_response(data: dict) -> dict:
    """Convert a raw Google Vision response into the project format."""
    if "error" in data:
        raise _vision_error(data["error"])
    response = (data.get("responses") or [{}])[0]
    if "error" in response:
        raise _vision_error(response["error"])
    faces = response.get("faceAnnotations") or []
    if not faces:
        raise HTTPException(422, NO_FACE_MESSAGE)
    face = faces[0]
    scores = {
        name: LIKELIHOOD_SCORES.get(face.get(field, "UNKNOWN"), 0.0)
        for field, name in VISION_FIELDS.items()
    }
    top, top_score = max(scores.items(), key=lambda item: item[1])
    emotion = top if top_score >= NEUTRAL_THRESHOLD else "neutral"
    return {
        "emotion": emotion,
        "scores": scores,
        "provider": "google",
        "face_confidence": face.get("detectionConfidence"),
    }


async def _detect_google(image_bytes: bytes) -> dict:
    return parse_vision_response(await _call_vision(image_bytes))


async def _detect_mock(image_bytes: bytes) -> dict:
    """Simulated provider for work without credentials; ignores the image."""
    emotion = os.getenv("MOCK_EMOTION") or random.choice(
        ["happiness", "neutral", "sadness", "surprise", "anger"]
    )
    scores = {name: 0.05 for name in VISION_FIELDS.values()}
    if emotion in scores:
        scores[emotion] = 0.9
    return {"emotion": emotion, "scores": scores, "provider": "mock"}


class LocalEmotionModel:
    """YuNet face detector plus the expression classifiers (OpenCV)."""

    def __init__(self) -> None:
        # Imported here so the API still starts without OpenCV installed.
        import cv2
        import numpy as np

        self.cv2, self.np = cv2, np
        self.detector = cv2.FaceDetectorYN.create(
            str(FACE_MODEL), "", (320, 320), FACE_THRESHOLD
        )
        self.ferplus = cv2.dnn.readNetFromONNX(str(EMOTION_MODEL))
        self.hsemotion = (
            cv2.dnn.readNetFromONNX(str(HSEMOTION_MODEL))
            if HSEMOTION_MODEL.exists()
            else None
        )
        # OpenCV networks are not safe to run from two threads at once.
        self.lock = threading.Lock()

    def predict(self, image_bytes: bytes) -> dict:
        cv2, np = self.cv2, self.np
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise HTTPException(422, UNREADABLE_IMAGE_MESSAGE)
        height, width = image.shape[:2]
        if min(height, width) < MIN_IMAGE_SIDE:
            raise HTTPException(422, NO_FACE_MESSAGE)
        if width > MAX_DETECTION_WIDTH:
            scale = MAX_DETECTION_WIDTH / width
            image = cv2.resize(image, None, fx=scale, fy=scale)
            height, width = image.shape[:2]
            if min(height, width) < MIN_IMAGE_SIDE:
                raise HTTPException(422, NO_FACE_MESSAGE)
        with self.lock:
            self.detector.setInputSize((width, height))
            _, faces = self.detector.detect(image)
            if faces is None or len(faces) == 0:
                raise HTTPException(422, NO_FACE_MESSAGE)
            # Largest face: the user sitting in front of the camera.
            face = max(faces, key=lambda row: row[2] * row[3])
            scores, used = self._classify(self._crop(image, face[:4]))
        return {
            "emotion": max(scores, key=scores.get),
            "scores": scores,
            "provider": "local",
            "model": "+".join(used),
            "face_confidence": round(float(face[14]), 4),
        }

    def classify(self, face) -> dict[str, float]:
        """Classify an already cropped BGR face (used by the evaluation)."""
        with self.lock:
            return self._classify(face)[0]

    def _classify(self, face) -> tuple[dict[str, float], list[str]]:
        """Average the probabilities of the active classifiers."""
        used = local_classifiers()
        if self.hsemotion is None:
            used = ["ferplus"]
        runs = {"ferplus": self._run_ferplus, "hsemotion": self._run_hsemotion}
        results = [runs[name](face) for name in used]
        scores = {
            emotion: round(
                sum(result[emotion] for result in results) / len(results), 4
            )
            for emotion in FERPLUS_CLASSES
        }
        return scores, used

    def _softmax(self, logits):
        exp = self.np.exp(logits - logits.max())
        return exp / exp.sum()

    def _run_ferplus(self, face) -> dict[str, float]:
        cv2, np = self.cv2, self.np
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blob = cv2.resize(gray, (FERPLUS_SIZE, FERPLUS_SIZE)).astype(
            np.float32
        )
        self.ferplus.setInput(blob.reshape(1, 1, FERPLUS_SIZE, FERPLUS_SIZE))
        probabilities = self._softmax(self.ferplus.forward().ravel())
        return dict(zip(FERPLUS_CLASSES, map(float, probabilities)))

    def _run_hsemotion(self, face) -> dict[str, float]:
        cv2, np = self.cv2, self.np
        rgb = cv2.cvtColor(
            cv2.resize(face, (HSEMOTION_SIZE, HSEMOTION_SIZE)),
            cv2.COLOR_BGR2RGB,
        )
        blob = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        self.hsemotion.setInput(
            blob.astype(np.float32).transpose(2, 0, 1)[None]
        )
        probabilities = self._softmax(self.hsemotion.forward().ravel())
        return dict(zip(HSEMOTION_CLASSES, map(float, probabilities)))

    def _crop(self, image, box):
        """Return the face as a square BGR crop, clipped to the frame."""
        x, y, w, h = box
        side = max(w, h)
        left = int(round(x + w / 2 - side / 2))
        top = int(round(y + h / 2 - side / 2))
        right, bottom = left + int(round(side)), top + int(round(side))
        left, top = max(left, 0), max(top, 0)
        right = min(right, image.shape[1])
        bottom = min(bottom, image.shape[0])
        if right - left < 2 or bottom - top < 2:
            raise HTTPException(422, NO_FACE_MESSAGE)
        return image[top:bottom, left:right]


_local_model: LocalEmotionModel | None = None
_local_model_lock = threading.Lock()


def _get_local_model() -> LocalEmotionModel:
    """Load the local model once, on first use."""
    global _local_model
    if _local_model is None:
        with _local_model_lock:
            if _local_model is None:
                if not local_models_available():
                    raise HTTPException(
                        503,
                        "The emotion models are missing. Run "
                        "scripts/download_emotion_models.py.",
                    )
                try:
                    _local_model = LocalEmotionModel()
                except ImportError as exc:
                    raise HTTPException(
                        503,
                        "OpenCV is missing. Install the dependencies "
                        "in requirements.txt.",
                    ) from exc
    return _local_model


async def _detect_local(image_bytes: bytes) -> dict:
    # CPU-bound: run outside the event loop.
    return await asyncio.to_thread(_get_local_model().predict, image_bytes)


PROVIDERS = {
    "local": _detect_local,
    "google": _detect_google,
    "mock": _detect_mock,
}


async def detect_emotion(image_bytes: bytes) -> dict:
    """Detect the dominant emotion with the active provider."""
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"Invalid EMOTION_PROVIDER: {name}")
    return await PROVIDERS[name](image_bytes)

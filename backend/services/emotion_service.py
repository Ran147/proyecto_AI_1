"""Emotion detection from an image.

Single entry point: ``detect_emotion(image_bytes)``. Providers:

- ``local``: runs in the backend with OpenCV. YuNet (pretrained) finds the
  face and a classifier reads its expression. The classifier is chosen with
  the EMOTION_MODEL setting (.env), or per request through the API:

  - ``ensemble`` (default): the average of two pretrained networks, FER+
    (grayscale, 64x64) and HSEmotion (EfficientNet-B0 trained on AffectNet,
    color, 224x224).
  - ``ferplus`` or ``hsemotion``: one pretrained network alone.
  - ``scratch``: our own convolutional network, trained from random
    initialization on FER-2013 (training/emotion_cnn/), grayscale 48x48.
    Optional: set EMOTION_MODEL=scratch to use it.

  No API key. The pretrained ONNX files are fetched by
  ``scripts/download_emotion_models.py``; ours is committed in artifacts/.
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
# Our network, trained from scratch (training/emotion_cnn/notebook.ipynb).
SCRATCH_MODEL = MODELS_DIR / "emotion_cnn.onnx"
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
# Output order of our network (training/emotion_cnn/notebook.ipynb). It takes
# a 48x48 grayscale face with pixel values scaled to 0-1, like FER-2013.
SCRATCH_CLASSES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)
SCRATCH_SIZE = 48
# FER-2013 faces are cropped tighter than YuNet's box: our network reads the
# central 85 % of the box (the best share on FER-2013 validation faces).
SCRATCH_CROP_SCALE = 0.85
# Classifier choices: name -> networks whose probabilities are averaged.
CLASSIFIER_CHOICES = {
    "ensemble": ["ferplus", "hsemotion"],
    "ferplus": ["ferplus"],
    "hsemotion": ["hsemotion"],
    "scratch": ["scratch"],
}
DEFAULT_CLASSIFIER = "ensemble"
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


NETWORK_FILES = {
    "scratch": SCRATCH_MODEL,
    "ferplus": EMOTION_MODEL,
    "hsemotion": HSEMOTION_MODEL,
}


def local_models_available() -> bool:
    """Tell whether the face detector and at least one classifier exist."""
    return FACE_MODEL.exists() and any(
        path.exists() for path in NETWORK_FILES.values()
    )


def available_classifiers() -> list[str]:
    """Classifier choices whose ONNX files are all in place."""
    return [
        choice
        for choice, networks in CLASSIFIER_CHOICES.items()
        if all(NETWORK_FILES[name].exists() for name in networks)
    ]


def check_classifier(choice: str | None) -> str | None:
    """Validate a per-request classifier choice; unknown ones answer 422."""
    if not choice:
        return None
    code = choice.lower()
    if code not in CLASSIFIER_CHOICES:
        options = ", ".join(CLASSIFIER_CHOICES)
        raise HTTPException(
            422, f"Unknown emotion model '{choice}': use one of {options}"
        )
    return code


def local_classifiers(choice: str | None = None) -> list[str]:
    """Networks the local provider runs for ``choice``.

    Without a choice, EMOTION_MODEL decides (default ``ensemble``). A choice
    whose files are missing falls back to the first available one.
    """
    choice = (
        choice or os.getenv("EMOTION_MODEL") or DEFAULT_CLASSIFIER
    ).lower()
    available = available_classifiers()
    if choice not in available:
        choice = available[0] if available else DEFAULT_CLASSIFIER
    return CLASSIFIER_CHOICES[choice]


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


async def _detect_google(image_bytes: bytes, model: str | None) -> dict:
    return parse_vision_response(await _call_vision(image_bytes))


async def _detect_mock(image_bytes: bytes, model: str | None) -> dict:
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
        # Every classifier whose file is present; each runs on demand.
        self.networks = {
            name: cv2.dnn.readNetFromONNX(str(path))
            for name, path in NETWORK_FILES.items()
            if path.exists()
        }
        # OpenCV networks are not safe to run from two threads at once.
        self.lock = threading.Lock()

    def predict(self, image_bytes: bytes, choice: str | None = None) -> dict:
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
            x, y, w, h = face[:4]
            shrink = SCRATCH_CROP_SCALE
            tight = self._crop(image, (
                x + w * (1 - shrink) / 2, y + h * (1 - shrink) / 2,
                w * shrink, h * shrink,
            ))
            scores, used = self._classify(
                self._crop(image, face[:4]), choice, tight
            )
        return {
            "emotion": max(scores, key=scores.get),
            "scores": scores,
            "provider": "local",
            "model": "+".join(used),
            "face_confidence": round(float(face[14]), 4),
        }

    def classify(self, face, choice: str | None = None) -> dict[str, float]:
        """Classify an already cropped BGR face (used by the evaluation)."""
        with self.lock:
            return self._classify(face, choice)[0]

    def _classify(
        self, face, choice: str | None = None, tight_face=None
    ) -> tuple[dict[str, float], list[str]]:
        """Average the probabilities of the chosen classifiers.

        ``tight_face`` is the tighter crop our network reads; without it
        (an already cropped face) every network reads ``face``.
        """
        used = [
            name for name in local_classifiers(choice)
            if name in self.networks
        ]
        runs = {
            "scratch": self._run_scratch,
            "ferplus": self._run_ferplus,
            "hsemotion": self._run_hsemotion,
        }
        results = [
            runs[name](
                tight_face if name == "scratch" and tight_face is not None
                else face
            )
            for name in used
        ]
        # Our network has no "contempt" class: it counts as 0 there.
        scores = {
            emotion: round(
                sum(result.get(emotion, 0.0) for result in results)
                / len(results),
                4,
            )
            for emotion in FERPLUS_CLASSES
        }
        return scores, used

    def _softmax(self, logits):
        exp = self.np.exp(logits - logits.max())
        return exp / exp.sum()

    def _run_scratch(self, face) -> dict[str, float]:
        cv2, np = self.cv2, self.np
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blob = cv2.resize(gray, (SCRATCH_SIZE, SCRATCH_SIZE)).astype(
            np.float32
        ) / 255
        network = self.networks["scratch"]
        network.setInput(blob.reshape(1, 1, SCRATCH_SIZE, SCRATCH_SIZE))
        probabilities = self._softmax(network.forward().ravel())
        return dict(zip(SCRATCH_CLASSES, map(float, probabilities)))

    def _run_ferplus(self, face) -> dict[str, float]:
        cv2, np = self.cv2, self.np
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blob = cv2.resize(gray, (FERPLUS_SIZE, FERPLUS_SIZE)).astype(
            np.float32
        )
        network = self.networks["ferplus"]
        network.setInput(blob.reshape(1, 1, FERPLUS_SIZE, FERPLUS_SIZE))
        probabilities = self._softmax(network.forward().ravel())
        return dict(zip(FERPLUS_CLASSES, map(float, probabilities)))

    def _run_hsemotion(self, face) -> dict[str, float]:
        cv2, np = self.cv2, self.np
        rgb = cv2.cvtColor(
            cv2.resize(face, (HSEMOTION_SIZE, HSEMOTION_SIZE)),
            cv2.COLOR_BGR2RGB,
        )
        blob = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        network = self.networks["hsemotion"]
        network.setInput(blob.astype(np.float32).transpose(2, 0, 1)[None])
        probabilities = self._softmax(network.forward().ravel())
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


async def _detect_local(image_bytes: bytes, model: str | None) -> dict:
    # CPU-bound: run outside the event loop.
    return await asyncio.to_thread(
        _get_local_model().predict, image_bytes, model
    )


PROVIDERS = {
    "local": _detect_local,
    "google": _detect_google,
    "mock": _detect_mock,
}


async def detect_emotion(image_bytes: bytes, model: str | None = None) -> dict:
    """Detect the dominant emotion with the active provider.

    ``model`` picks the classifier of the local provider for this request
    (see CLASSIFIER_CHOICES); the other providers ignore it.
    """
    name = provider_name()
    if name not in PROVIDERS:
        raise HTTPException(500, f"Invalid EMOTION_PROVIDER: {name}")
    return await PROVIDERS[name](image_bytes, check_classifier(model))

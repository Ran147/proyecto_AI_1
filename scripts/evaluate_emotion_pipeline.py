"""Evaluate the whole local emotion pipeline as the application runs it.

Usage, from the repository root::

    python scripts/evaluate_emotion_pipeline.py

scripts/evaluate_emotion_model.py feeds the classifiers the FER-2013 crops
directly. Here 60 random test faces per emotion (420, fixed seed) are
enlarged and pasted into a 640x480 frame, so the YuNet detector has to find
and crop each face first, as with the camera. Every classifier choice is
scored on the same detected faces. Needs ``data/fer2013/test/``.
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services import emotion_service as es  # noqa: E402

FOLDERS = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "surprise": "surprise",
}
random.seed(1)
items = []
for folder, emotion in FOLDERS.items():
    paths = sorted((ROOT / "data/fer2013/test" / folder).glob("*.jpg"))
    items += [(p, emotion) for p in random.sample(paths, min(60, len(paths)))]
model = es._get_local_model()
results = {c: [0, 0] for c in es.available_classifiers()}
missed = 0
for path, emotion in items:
    face = cv2.resize(
        cv2.imread(str(path)), (260, 260), interpolation=cv2.INTER_CUBIC
    )
    frame = np.full((480, 640, 3), 90, np.uint8)
    frame[110:370, 190:450] = face
    data = cv2.imencode(".jpg", frame)[1].tobytes()
    try:
        for choice in results:
            out = model.predict(data, choice)
            results[choice][0] += out["emotion"] == emotion
            results[choice][1] += 1
    except HTTPException:
        missed += 1
print(f"{len(items)} faces, YuNet missed {missed}")
for choice, (ok, n) in results.items():
    print(f"{choice:10s} accuracy {ok / n:.3f} on {n} detected faces")

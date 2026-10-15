"""Evaluate the local emotion classifiers on the FER-2013 test images.

Usage, from the repository root::

    python scripts/evaluate_emotion_model.py

It needs the Kaggle dataset ``msambare/fer2013`` unzipped in
``data/fer2013/`` (folders ``test/<emotion>/*.jpg``) and the ONNX files from
``scripts/download_emotion_models.py``.

The images are 48x48 grayscale faces, already cropped, so this measures the
classifiers only, not the face detector. The labels are the original
FER-2013 ones, which are known to be noisy. FER+ was trained on these
images' relabeled training split; HSEmotion was trained on AffectNet
(larger color photos), so this test is out of its domain. ``contempt`` has
no FER-2013 label and is ignored.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services import emotion_service  # noqa: E402

TEST_DIR = Path(__file__).resolve().parents[1] / "data" / "fer2013" / "test"
# Folder name -> emotion name used by the API.
FOLDERS = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "neutral": "neutral",
    "sad": "sadness",
    "surprise": "surprise",
}


def main() -> int:
    if not TEST_DIR.exists():
        print(f"Missing {TEST_DIR}; see the module docstring.")
        return 1
    model = emotion_service._get_local_model()
    faces, truth = [], []
    for folder, emotion in FOLDERS.items():
        for path in sorted((TEST_DIR / folder).glob("*")):
            faces.append(cv2.imread(str(path), cv2.IMREAD_COLOR))
            truth.append(emotion)
    truth = np.array(truth)
    emotions = list(FOLDERS.values())
    print(f"{len(faces)} test faces\n")
    header = "model      accuracy  macro recall  " + "  ".join(
        f"{e[:8]:>8}" for e in emotions
    )
    print(header)
    for choice in ("scratch", "ferplus", "hsemotion", "ensemble"):
        os.environ["EMOTION_MODEL"] = choice
        predicted = []
        for face in faces:
            scores = model.classify(face)
            predicted.append(max(emotions, key=scores.get))
        predicted = np.array(predicted)
        recalls = [
            float((predicted[truth == e] == e).mean()) for e in emotions
        ]
        print(
            f"{choice:<10} {float((predicted == truth).mean()):8.3f}"
            f"  {np.mean(recalls):12.3f}  "
            + "  ".join(f"{r:8.3f}" for r in recalls)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

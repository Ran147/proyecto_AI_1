"""Download the ONNX models of the local emotion provider.

Usage, from the repository root::

    python scripts/download_emotion_models.py

Files are saved in artifacts/ and checked against a SHA-256
hash. Models:

- YuNet face detector, OpenCV Zoo (MIT license).
- FER+ emotion classifier, ONNX Model Zoo (MIT license); Barsoum et al.
  (2016), trained on the FER+ relabeling of FER-2013.
- HSEmotion emotion classifier (EfficientNet-B0 trained on AffectNet),
  av-savchenko/face-emotion-recognition (Apache 2.0 license); Savchenko et
  al. (2022).
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODELS = [
    (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "emotion-ferplus-8.onnx",
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c",
    ),
    (
        "enet_b0_8_best_afew.onnx",
        "https://github.com/av-savchenko/face-emotion-recognition/raw/main/"
        "models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx",
        "7aa2ea31c1311f4f8aa9d3fdb085d418dd4e7a48c4b9ed41df8c044f91d0213f",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ARTIFACTS.mkdir(exist_ok=True)
    for name, url, expected in MODELS:
        target = ARTIFACTS / name
        if target.exists() and sha256(target) == expected:
            print(f"{name}: already present")
            continue
        print(f"{name}: downloading...")
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        if sha256(partial) != expected:
            partial.unlink()
            print(f"{name}: hash mismatch, file discarded", file=sys.stderr)
            return 1
        partial.replace(target)
        print(f"{name}: saved ({target.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the local speech-to-text models (Vosk, small English and Spanish).

Usage, from the repository root::

    python scripts/download_speech_models.py

The models are already committed in artifacts/; this script re-fetches any
that is missing. Source: https://alphacephei.com/vosk/models (Apache 2.0
license, about 40 MB zipped each).
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
BASE_URL = "https://alphacephei.com/vosk/models/"
MODELS = {
    "vosk-model-small-en-us-0.15": (
        "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"
    ),
    "vosk-model-small-es-0.42": (
        "09b239888f633ef2f0b4e09736e3d9936acfd810bc65d53fad45261762c6511f"
    ),
}


def download(name: str, expected: str) -> bool:
    target = ARTIFACTS / name
    if (target / "am" / "final.mdl").exists():
        print(f"{name}: already present")
        return True
    print(f"{name}: downloading...")
    with urllib.request.urlopen(f"{BASE_URL}{name}.zip", timeout=300) as reply:
        content = reply.read()
    if hashlib.sha256(content).hexdigest() != expected:
        print(f"{name}: hash mismatch, download discarded", file=sys.stderr)
        return False
    ARTIFACTS.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(ARTIFACTS)
    print(f"{name}: saved in {target}")
    return True


def main() -> int:
    results = [download(name, digest) for name, digest in MODELS.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

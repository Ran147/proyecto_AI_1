"""Download the local speech-to-text model (Vosk, small English).

Usage, from the repository root::

    python scripts/download_speech_model.py

The model is already committed in artifacts/; this script re-fetches it if
the folder is missing. Source: https://alphacephei.com/vosk/models
(vosk-model-small-en-us-0.15, Apache 2.0 license, 40 MB zipped).
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
NAME = "vosk-model-small-en-us-0.15"
URL = f"https://alphacephei.com/vosk/models/{NAME}.zip"
SHA256 = "30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498"


def main() -> int:
    target = ARTIFACTS / NAME
    if (target / "am" / "final.mdl").exists():
        print(f"{NAME}: already present")
        return 0
    print(f"{NAME}: downloading...")
    with urllib.request.urlopen(URL, timeout=300) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        print(f"{NAME}: hash mismatch, download discarded", file=sys.stderr)
        return 1
    ARTIFACTS.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        archive.extractall(ARTIFACTS)
    print(f"{NAME}: saved in {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

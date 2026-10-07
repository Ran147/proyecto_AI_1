"""Download the datasets the training notebooks read.

Usage, from the repository root::

    python scripts/download_datasets.py            # the ten model datasets
    python scripts/download_datasets.py --emotion  # also FER-2013 (60 MB)

Files are saved under data/ (git-ignored) at the paths each notebook expects.
The datasets are only needed to re-run the notebooks; the trained models are
already in artifacts/. Every source is public and needs no account: Kaggle
serves public datasets as a zip from its download endpoint.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/"

# folder under data/ -> (url, kind); kind is "zip" or the file name to save.
DATASETS = {
    "wine_quality": (KAGGLE + "rajyellow46/wine-quality", "zip"),
    "car_price": (
        "https://raw.githubusercontent.com/amankharwal/Website-data/master/"
        "car%20data.csv",
        "car data.csv",
    ),
    "churn": (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv",
        "Telco-Customer-Churn.csv",
    ),
    "stroke": (KAGGLE + "fedesoriano/stroke-prediction-dataset", "zip"),
    "avocado": (KAGGLE + "neuromusic/avocado-prices", "zip"),
    "bitcoin": (KAGGLE + "team-ai/bitcoin-price-prediction", "zip"),
    "body_fat": (KAGGLE + "fedesoriano/body-fat-prediction-dataset", "zip"),
    "movies": (
        "https://drive.google.com/uc?export=download"
        "&id=1Dn1BZD3YxgBQJSIjbfNnmCFlDW2jdQGD",
        "zip",
    ),
    "hepatitis": (KAGGLE + "fedesoriano/hepatitis-c-dataset", "zip"),
    "cirrhosis": (KAGGLE + "fedesoriano/cirrhosis-prediction-dataset", "zip"),
}
EMOTION = {"fer2013": (KAGGLE + "msambare/fer2013", "zip")}


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download(folder: str, url: str, kind: str) -> bool:
    target = DATA / folder
    if target.exists() and any(target.iterdir()):
        print(f"{folder}: already present")
        return True
    print(f"{folder}: downloading...")
    try:
        content = fetch(url)
        target.mkdir(parents=True, exist_ok=True)
        if kind == "zip":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                archive.extractall(target)
        else:
            (target / kind).write_bytes(content)
    except Exception as error:  # noqa: BLE001 - report and continue
        print(f"{folder}: FAILED ({error}); download it by hand from {url}")
        return False
    files = sorted(p.name for p in target.iterdir())
    print(f"{folder}: {', '.join(files[:4])}")
    return True


def main() -> int:
    datasets = dict(DATASETS)
    if "--emotion" in sys.argv[1:]:
        datasets.update(EMOTION)
    DATA.mkdir(exist_ok=True)
    results = [download(name, *spec) for name, spec in datasets.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

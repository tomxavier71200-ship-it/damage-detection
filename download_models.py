from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODELS = [
    {
        "path": BASE_DIR / "car_damage_seg.pt",
        "url": "https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11/resolve/main/best.pt",
    },
    {
        "path": BASE_DIR / "car_parts_seg.pt",
        "url": "https://huggingface.co/Majorburn/yolov11-carparts-seg/resolve/main/best.pt",
    },
]


def sha256_of_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_missing(target_path: Path, url: str) -> None:
    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"Model already present: {target_path.name}")
        return

    print(f"Downloading {target_path.name} from {url}")
    urllib.request.urlretrieve(url, target_path)
    print(f"Saved {target_path.name} ({target_path.stat().st_size} bytes)")


if __name__ == "__main__":
    for model in MODELS:
        download_if_missing(model["path"], model["url"])

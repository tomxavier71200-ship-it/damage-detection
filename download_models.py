from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# ONNX exports of the original YOLO11 .pt models (see the model card for export details).
# Pinned to a commit of https://huggingface.co/tommy33355/damagelens-onnx and checked by SHA-256.
MODEL_REPO = "https://huggingface.co/tommy33355/damagelens-onnx/resolve/50bcc7ecc01ae2a572a85b8f62c13844e9ff64d1"

MODELS = [
    {
        "path": BASE_DIR / "car_damage_seg.onnx",
        "url": f"{MODEL_REPO}/car_damage_seg.onnx",
        "sha256": "74d589b389abea1d78d5243a01b9e3acef8c599099c9e02f321b81c000372034",
    },
    {
        "path": BASE_DIR / "car_parts_seg.onnx",
        "url": f"{MODEL_REPO}/car_parts_seg.onnx",
        "sha256": "aea07464e059d820b2a459ef0a4289bdbaf3ef7eb8da6d3af3a5fe3fda6fb1ad",
    },
]


def sha256_of_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_if_missing(target_path: Path, url: str, expected_sha256: str) -> None:
    if target_path.exists() and sha256_of_file(target_path) == expected_sha256:
        print(f"Model already present: {target_path.name}")
        return

    print(f"Downloading {target_path.name} from {url}")
    urllib.request.urlretrieve(url, target_path)
    actual_sha256 = sha256_of_file(target_path)
    if actual_sha256 != expected_sha256:
        target_path.unlink()
        raise RuntimeError(f"Checksum mismatch for {target_path.name}: {actual_sha256}")
    print(f"Saved {target_path.name} ({target_path.stat().st_size} bytes)")


if __name__ == "__main__":
    for model in MODELS:
        download_if_missing(model["path"], model["url"], model["sha256"])

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
CACHE_DIR = BASE_DIR / ".runtime-cache"
MPL_CONFIG_DIR = CACHE_DIR / "matplotlib"
YOLO_CONFIG_DIR = CACHE_DIR / "ultralytics"
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_DAMAGE_CONFIDENCE = 0.18
DEFAULT_PARTS_CONFIDENCE = 0.22
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 960
DAMAGE_MODEL_PATH = BASE_DIR / "car_damage_seg.pt"
PARTS_MODEL_PATH = BASE_DIR / "car_parts_seg.pt"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))

import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

MODEL_LOCK = threading.Lock()
MODEL_CACHE: dict[str, YOLO] = {}


@dataclass
class DetectionReport:
    image_name: str
    annotated_image: str
    findings: list[dict[str, Any]]
    parts_found: list[str]
    summary: dict[str, Any]
    no_damage: bool
    damage_model_label: str
    parts_model_label: str


def is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def save_upload(file_storage, destination: Path) -> Path:
    filename = secure_filename(file_storage.filename or "")
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    output_path = destination / unique_name
    file_storage.save(output_path)
    return output_path


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def parse_float(value: str | None, default: float, lower: float, upper: float) -> float:
    try:
        return clamp(float(value or default), lower, upper)
    except (TypeError, ValueError):
        return default


def load_model(model_path: Path) -> YOLO:
    model_key = str(model_path)
    with MODEL_LOCK:
        if model_key not in MODEL_CACHE:
            MODEL_CACHE[model_key] = YOLO(str(model_path))
        return MODEL_CACHE[model_key]


def model_files_ready() -> tuple[bool, list[str]]:
    missing = []
    if not DAMAGE_MODEL_PATH.exists():
        missing.append(DAMAGE_MODEL_PATH.name)
    if not PARTS_MODEL_PATH.exists():
        missing.append(PARTS_MODEL_PATH.name)
    return len(missing) == 0, missing


def confidence_band(confidence_ratio: float) -> str:
    if confidence_ratio >= 0.85:
        return "Very High"
    if confidence_ratio >= 0.70:
        return "High"
    if confidence_ratio >= 0.50:
        return "Medium"
    return "Low"


def estimate_size(area_ratio: float) -> str:
    if area_ratio < 0.003:
        return "Hairline"
    if area_ratio < 0.01:
        return "Small"
    if area_ratio < 0.025:
        return "Medium"
    if area_ratio < 0.06:
        return "Large"
    return "Severe"


def estimate_severity(area_ratio: float, confidence_ratio: float, damage_type: str) -> str:
    weighted_score = (area_ratio * 100) + (confidence_ratio * 0.35)
    if damage_type in {"glass shatter", "lamp broken", "tire flat"}:
        weighted_score += 1.2

    if weighted_score < 0.7:
        return "Low"
    if weighted_score < 1.8:
        return "Moderate"
    if weighted_score < 4.0:
        return "High"
    return "Critical"


def titleize_label(label: str) -> str:
    return label.replace("_", " ").title()


def normalize_mask(mask_array: np.ndarray | None) -> np.ndarray | None:
    if mask_array is None:
        return None
    return (mask_array > 0.5).astype(np.uint8)


def extract_prediction_entries(result, image_width: int, image_height: int) -> list[dict[str, Any]]:
    boxes = result.boxes
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    entries: list[dict[str, Any]] = []

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence_ratio = float(box.conf)
        class_id = int(box.cls)
        label = result.names[class_id]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        image_area = image_width * image_height
        area_ratio = area / image_area
        center_x = x1 + width / 2
        center_y = y1 + height / 2

        entries.append(
            {
                "label": label,
                "label_pretty": titleize_label(label),
                "confidence_ratio": confidence_ratio,
                "confidence": round(confidence_ratio * 100, 1),
                "bbox": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "area_ratio": area_ratio,
                "mask": normalize_mask(masks[index]) if masks is not None else None,
            }
        )

    return entries


def intersection_over_union(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mask_overlap_ratio(damage_mask: np.ndarray | None, part_mask: np.ndarray | None) -> float:
    if damage_mask is None or part_mask is None:
        return 0.0
    overlap = np.logical_and(damage_mask, part_mask).sum()
    damage_area = damage_mask.sum()
    return float(overlap / damage_area) if damage_area else 0.0


def choose_part_for_damage(damage_entry: dict[str, Any], parts_entries: list[dict[str, Any]]) -> tuple[str, float]:
    if not parts_entries:
        return "Unmatched Body Area", 0.0

    best_label = "Unmatched Body Area"
    best_score = 0.0

    for part in parts_entries:
        overlap_score = mask_overlap_ratio(damage_entry["mask"], part["mask"])
        iou_score = intersection_over_union(damage_entry["bbox"], part["bbox"])
        score = max(overlap_score, iou_score * 0.7)
        if score > best_score:
            best_score = score
            best_label = part["label_pretty"]

    return best_label, best_score


def build_damage_findings(
    damage_entries: list[dict[str, Any]],
    parts_entries: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    for damage in damage_entries:
        x1, y1, x2, y2 = damage["bbox"]
        center_x, center_y = damage["center"]
        matched_part, match_score = choose_part_for_damage(damage, parts_entries)
        x_ratio = center_x / image_width
        y_ratio = center_y / image_height

        findings.append(
            {
                "type": damage["label_pretty"],
                "confidence": damage["confidence"],
                "confidence_band": confidence_band(damage["confidence_ratio"]),
                "severity": estimate_severity(damage["area_ratio"], damage["confidence_ratio"], damage["label"]),
                "size": estimate_size(damage["area_ratio"]),
                "area_percent": round(damage["area_ratio"] * 100, 2),
                "component": matched_part,
                "component_match": round(match_score * 100, 1),
                "zone": estimate_zone(x_ratio, y_ratio),
                "side": estimate_side(x_ratio),
                "bbox": (x1, y1, x2, y2),
            }
        )

    severity_rank = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    findings.sort(key=lambda item: (severity_rank[item["severity"]], item["confidence"]), reverse=True)
    return findings


def estimate_zone(x_ratio: float, y_ratio: float) -> str:
    if y_ratio < 0.22:
        vertical = "Upper"
    elif y_ratio < 0.70:
        vertical = "Mid"
    else:
        vertical = "Lower"

    if x_ratio < 0.18:
        horizontal = "Front Edge"
    elif x_ratio < 0.40:
        horizontal = "Front Section"
    elif x_ratio < 0.60:
        horizontal = "Center Section"
    elif x_ratio < 0.82:
        horizontal = "Rear Section"
    else:
        horizontal = "Rear Edge"

    return f"{vertical} {horizontal}"


def estimate_side(x_ratio: float) -> str:
    if x_ratio < 0.35:
        return "Left Side of Photo"
    if x_ratio > 0.65:
        return "Right Side of Photo"
    return "Center of Photo"


def annotate_image(
    image: np.ndarray,
    findings: list[dict[str, Any]],
    parts_entries: list[dict[str, Any]],
) -> np.ndarray:
    output = image.copy()
    part_color = (117, 156, 255)
    severity_colors = {
        "Low": (57, 176, 105),
        "Moderate": (74, 163, 255),
        "High": (0, 166, 255),
        "Critical": (51, 83, 232),
    }

    for part in parts_entries:
        x1, y1, x2, y2 = part["bbox"]
        cv2.rectangle(output, (x1, y1), (x2, y2), part_color, 1)
        cv2.putText(
            output,
            part["label_pretty"],
            (x1, max(18, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            part_color,
            1,
        )

    for finding in findings:
        x1, y1, x2, y2 = finding["bbox"]
        color = severity_colors.get(finding["severity"], (42, 208, 122))
        label = f"{finding['type']} on {finding['component']}"
        badge = f"{finding['severity']} | {finding['confidence']:.1f}%"
        badge_width = min(output.shape[1] - x1, 340)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(output, (x1, max(0, y1 - 54)), (x1 + badge_width, y1), color, -1)
        cv2.putText(
            output,
            label[:40],
            (x1 + 8, max(22, y1 - 32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            output,
            badge,
            (x1 + 8, max(42, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return output


def summarize_findings(findings: list[dict[str, Any]], parts_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not findings:
        return {
            "damage_count": 0,
            "overall_severity": "None",
            "dominant_type": "No damage detected",
            "avg_confidence": 0,
            "largest_area_percent": 0,
            "parts_count": len(parts_entries),
        }

    severity_rank = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}
    overall_severity = max(findings, key=lambda item: severity_rank[item["severity"]])["severity"]
    dominant_type = max(
        {item["type"]: sum(1 for entry in findings if entry["type"] == item["type"]) for item in findings},
        key=lambda label: sum(1 for entry in findings if entry["type"] == label),
    )
    avg_confidence = round(sum(item["confidence"] for item in findings) / len(findings), 1)
    largest_area_percent = max(item["area_percent"] for item in findings)

    return {
        "damage_count": len(findings),
        "overall_severity": overall_severity,
        "dominant_type": dominant_type,
        "avg_confidence": avg_confidence,
        "largest_area_percent": largest_area_percent,
        "parts_count": len(parts_entries),
    }


def build_report(
    damage_model: YOLO,
    parts_model: YOLO,
    image_path: Path,
    damage_confidence: float,
    parts_confidence: float,
    iou_threshold: float,
) -> DetectionReport:
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError(f"Could not read image: {image_path.name}")

    image_height, image_width = original.shape[:2]

    damage_result = damage_model.predict(
        source=original,
        conf=damage_confidence,
        iou=iou_threshold,
        imgsz=DEFAULT_IMAGE_SIZE,
        verbose=False,
    )[0]
    parts_result = parts_model.predict(
        source=original,
        conf=parts_confidence,
        iou=iou_threshold,
        imgsz=DEFAULT_IMAGE_SIZE,
        verbose=False,
    )[0]

    damage_entries = extract_prediction_entries(damage_result, image_width, image_height)
    parts_entries = extract_prediction_entries(parts_result, image_width, image_height)
    findings = build_damage_findings(damage_entries, parts_entries, image_width, image_height)
    parts_found = sorted({part["label_pretty"] for part in parts_entries})

    annotated = annotate_image(original, findings, parts_entries)
    result_name = f"{uuid.uuid4().hex}_{image_path.stem}.jpg"
    result_path = RESULTS_DIR / result_name
    cv2.imwrite(str(result_path), annotated)

    return DetectionReport(
        image_name=image_path.name,
        annotated_image=f"results/{result_name}",
        findings=findings,
        parts_found=parts_found,
        summary=summarize_findings(findings, parts_entries),
        no_damage=len(findings) == 0,
        damage_model_label="Car Damage Segmentation",
        parts_model_label="Car Parts Segmentation",
    )


def summarize_batch(reports: list[DetectionReport]) -> dict[str, Any]:
    total_detections = sum(report.summary["damage_count"] for report in reports)
    images_with_damage = sum(0 if report.no_damage else 1 for report in reports)
    largest_damage = max((report.summary["largest_area_percent"] for report in reports), default=0)
    avg_confidence = round(
        sum(report.summary["avg_confidence"] for report in reports if report.summary["damage_count"] > 0)
        / max(1, images_with_damage),
        1,
    )

    severity_rank = {"None": 0, "Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
    top_severity = "None"
    for report in reports:
        if severity_rank[report.summary["overall_severity"]] > severity_rank[top_severity]:
            top_severity = report.summary["overall_severity"]

    return {
        "image_count": len(reports),
        "images_with_damage": images_with_damage,
        "total_detections": total_detections,
        "highest_severity": top_severity,
        "largest_damage": largest_damage,
        "avg_confidence": avg_confidence if images_with_damage else 0,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    info = None
    reports: list[DetectionReport] = []
    batch_summary = None
    damage_confidence = DEFAULT_DAMAGE_CONFIDENCE
    parts_confidence = DEFAULT_PARTS_CONFIDENCE
    iou_threshold = DEFAULT_IOU
    ready, missing_models = model_files_ready()

    if request.method == "POST":
        image_files = request.files.getlist("image_files")
        damage_confidence = parse_float(request.form.get("damage_confidence"), DEFAULT_DAMAGE_CONFIDENCE, 0.05, 0.95)
        parts_confidence = parse_float(request.form.get("parts_confidence"), DEFAULT_PARTS_CONFIDENCE, 0.05, 0.95)
        iou_threshold = parse_float(request.form.get("iou_threshold"), DEFAULT_IOU, 0.10, 0.95)

        if not ready:
            error = f"Required model files are missing: {', '.join(missing_models)}"
            return render_template(
                "index.html",
                error=error,
                info=info,
                reports=reports,
                batch_summary=batch_summary,
                damage_confidence=damage_confidence,
                parts_confidence=parts_confidence,
                iou_threshold=iou_threshold,
                ready=ready,
                missing_models=missing_models,
            )

        valid_images = [
            image_file
            for image_file in image_files
            if image_file and image_file.filename and is_allowed(image_file.filename)
        ]

        if not valid_images:
            error = "Please upload at least one image file (.jpg, .jpeg, .png, .webp)."
            return render_template(
                "index.html",
                error=error,
                info=info,
                reports=reports,
                batch_summary=batch_summary,
                damage_confidence=damage_confidence,
                parts_confidence=parts_confidence,
                iou_threshold=iou_threshold,
                ready=ready,
                missing_models=missing_models,
            )

        try:
            damage_model = load_model(DAMAGE_MODEL_PATH)
            parts_model = load_model(PARTS_MODEL_PATH)
            info = (
                "Using local task-specific models: car damage segmentation plus car parts segmentation. "
                "This is a much better setup than a generic object detector for this task."
            )

            for image_file in valid_images:
                image_path = save_upload(image_file, UPLOAD_DIR)
                try:
                    reports.append(
                        build_report(
                            damage_model,
                            parts_model,
                            image_path,
                            damage_confidence,
                            parts_confidence,
                            iou_threshold,
                        )
                    )
                finally:
                    if image_path.exists():
                        image_path.unlink()

            batch_summary = summarize_batch(reports)
        except Exception as exc:
            error = f"Detection failed: {exc}"

    return render_template(
        "index.html",
        error=error,
        info=info,
        reports=reports,
        batch_summary=batch_summary,
        damage_confidence=damage_confidence,
        parts_confidence=parts_confidence,
        iou_threshold=iou_threshold,
        ready=ready,
        missing_models=missing_models,
    )


if __name__ == "__main__":
    app.run(debug=True)

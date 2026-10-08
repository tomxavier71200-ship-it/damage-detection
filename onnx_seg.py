"""ONNX Runtime inference for Ultralytics YOLO segmentation exports.

Replaces ``ultralytics.YOLO(...).predict`` in production so the app does not need
PyTorch at runtime. Pre- and post-processing mirror Ultralytics 8.3 (LetterBox,
non_max_suppression, process_mask, scale_boxes) for a dynamic-shape export (rect letterbox, like Ultralytics PyTorch predict) with
outputs ``output0`` [1, 4 + nc + 32, N] and ``output1`` [1, 32, mh, mw].
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MAX_WH = 7680  # class offset used for batched NMS (same as Ultralytics)
MAX_NMS = 30000
MAX_DET = 300
STRIDE = 32


@dataclass
class SegmentationResult:
    boxes_xyxy: np.ndarray  # (n, 4) float, original image pixels
    confidences: np.ndarray  # (n,) float
    class_ids: np.ndarray  # (n,) int
    masks: np.ndarray | None  # (n, H, W) float 0/1 in model input space, or None
    names: dict[int, str]


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Keep peak memory low on small instances: no arena pre-allocation or memory-pattern planning.
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # The published .onnx files are already optimized offline (ORT_ENABLE_EXTENDED); skipping
    # optimization at load avoids a second in-memory copy of the weights (~250 MB peak saved).
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def read_names(session: ort.InferenceSession) -> dict[int, str]:
    metadata = session.get_modelmeta().custom_metadata_map
    return {int(key): str(value) for key, value in ast.literal_eval(metadata["names"]).items()}


def letterbox(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad to the minimum stride-32 rectangle (Ultralytics ``auto=True``)."""
    height, width = image.shape[:2]
    ratio = min(image_size / height, image_size / width)
    new_unpad = int(round(width * ratio)), int(round(height * ratio))
    dw = np.mod(image_size - new_unpad[0], STRIDE) / 2
    dh = np.mod(image_size - new_unpad[1], STRIDE) / 2
    if (width, height) != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        current = order[0]
        keep.append(int(current))
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = intersection / (areas[current] + areas[rest] - intersection + 1e-12)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def process_masks(protos: np.ndarray, coefficients: np.ndarray, boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    channels, mask_h, mask_w = protos.shape
    input_h, input_w = shape
    masks = (coefficients @ protos.reshape(channels, -1)).reshape(-1, mask_h, mask_w)

    scaled = boxes.copy()
    scaled[:, [0, 2]] *= mask_w / input_w
    scaled[:, [1, 3]] *= mask_h / input_h
    cols = np.arange(mask_w, dtype=np.float32)[None, None, :]
    rows = np.arange(mask_h, dtype=np.float32)[None, :, None]
    x1, y1, x2, y2 = (scaled[:, i, None, None] for i in range(4))
    masks = masks * ((cols >= x1) & (cols < x2) & (rows >= y1) & (rows < y2))

    upsampled = np.stack(
        [cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR) for mask in masks]
    )
    return (upsampled > 0.0).astype(np.float32)


def predict(
    session: ort.InferenceSession,
    names: dict[int, str],
    image: np.ndarray,
    confidence: float,
    iou_threshold: float,
    image_size: int,
) -> SegmentationResult:
    original_h, original_w = image.shape[:2]

    padded = letterbox(image, image_size)
    input_h, input_w = padded.shape[:2]
    blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    output0, output1 = session.run(None, {session.get_inputs()[0].name: blob})
    del blob

    num_classes = len(names)
    predictions = output0[0].T  # (N, 4 + nc + nm)
    class_scores = predictions[:, 4 : 4 + num_classes]
    best_class = class_scores.argmax(1)
    best_score = class_scores[np.arange(len(predictions)), best_class]
    keep = best_score > confidence
    predictions, best_class, best_score = predictions[keep], best_class[keep], best_score[keep]

    empty = SegmentationResult(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int), None, names)
    if not len(predictions):
        return empty

    if len(predictions) > MAX_NMS:
        top = best_score.argsort()[::-1][:MAX_NMS]
        predictions, best_class, best_score = predictions[top], best_class[top], best_score[top]

    xywh = predictions[:, :4]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    kept = nms(boxes + best_class[:, None] * MAX_WH, best_score, iou_threshold)[:MAX_DET]
    boxes, best_class, best_score = boxes[kept], best_class[kept], best_score[kept]
    coefficients = predictions[kept, 4 + num_classes :]

    masks = process_masks(output1[0], coefficients, boxes, (input_h, input_w))

    gain = min(input_h / original_h, input_w / original_w)
    pad_x = round((input_w - original_w * gain) / 2 - 0.1)
    pad_y = round((input_h - original_h * gain) / 2 - 0.1)
    scaled = boxes.copy()
    scaled[:, [0, 2]] -= pad_x
    scaled[:, [1, 3]] -= pad_y
    scaled /= gain
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, original_w)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, original_h)

    return SegmentationResult(scaled, best_score.astype(np.float32), best_class.astype(int), masks, names)

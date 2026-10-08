# DamageLens AI

AI-assisted vehicle damage assessment and inspection reporting.

DamageLens AI analyzes vehicle inspection photos with two computer-vision segmentation models and turns
the detections into structured inspection findings: damage type, affected vehicle component, confidence,
estimated size, heuristic severity, image zone, an annotated image, and a batch summary.

Live demo: <https://damagelens.pntr.dev> (served by Render at <https://damage-lens-ai.onrender.com>)

## How it works

```text
Vehicle inspection photo(s)
        ↓
Damage segmentation      (car_damage_seg.onnx, ONNX Runtime)
        ↓  model released from memory
Vehicle-part segmentation (car_parts_seg.onnx, ONNX Runtime)
        ↓
Damage ↔ component matching (best of mask overlap and weighted box IoU)
        ↓
Confidence band, size, zone, heuristic severity
        ↓
Annotated image + structured findings + batch summary
```

Inference runs on **ONNX Runtime** (CPU). PyTorch and Ultralytics are not used at runtime.
`onnx_seg.py` implements the pre- and post-processing for the YOLO segmentation exports
(letterbox resize, non-maximum suppression, mask decoding and box rescaling), mirroring the
Ultralytics pipeline so the results match the original models.

To stay within the 512 MB memory limit of Render's free tier, the two models are loaded one at a time
for each image and released after use.

## Features

- Upload one or more vehicle images (`.jpg`, `.jpeg`, `.png`, `.webp`)
- Damage segmentation: crack, dent, glass shatter, lamp broken, scratch, tire flat
- Vehicle-part segmentation (23 parts, e.g. bumpers, doors, lights, hood, mirrors, wheels)
- Damage-to-component matching
- Confidence bands and heuristic severity
- Estimated damage size and image zone
- Annotated inspection images
- Batch-level summary
- Adjustable damage confidence, parts confidence and IoU thresholds

## Technology

- Python, Flask
- ONNX Runtime
- OpenCV, NumPy
- Gunicorn
- Render (free web service), Hugging Face Hub (model hosting)

## Models

The ONNX model files are not stored in this repository. They are hosted at
[tommy33355/damagelens-onnx](https://huggingface.co/tommy33355/damagelens-onnx) and downloaded by
`download_models.py`, which pins a specific Hugging Face commit and verifies each file's **SHA-256**
checksum (the build fails on a mismatch). No Hugging Face token is needed.

| File | Source model | Upstream license |
|---|---|---|
| `car_damage_seg.onnx` | [harpreetsahota/car-dd-segmentation-yolov11](https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11) (YOLO11x-seg, trained on CarDD) | AGPL-3.0 (stated on the model card) |
| `car_parts_seg.onnx` | [Majorburn/yolov11-carparts-seg](https://huggingface.co/Majorburn/yolov11-carparts-seg) (YOLO11n-seg) | AGPL-3.0 |

The ONNX files are format conversions of these models (weights unchanged). Credit to Harpreet Sahota
(damage model), Majorburn (parts model), [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO11),
and the authors of the [CarDD dataset](https://cardd-ustc.github.io).

**Licensing note:** the damage model was trained on CarDD, whose license requires prior authorization from
the PIC Lab for commercial use. See [LICENSE-REVIEW.md](LICENSE-REVIEW.md) before any commercial use.

## Run locally

```bash
pip install -r requirements.txt
python download_models.py
python app.py
```

Open <http://127.0.0.1:5000>.

## Deploy on Render

`render.yaml` is a Render Blueprint (free web service).

Build command:

```bash
pip install -r requirements.txt && python download_models.py
```

Start command (also in `Procfile`):

```bash
gunicorn --workers 1 --timeout 180 app:app
```

## Project structure

```text
damage-detection/
├── app.py               # Flask app: upload handling, findings, severity, annotation, summaries
├── onnx_seg.py          # ONNX Runtime inference + YOLO segmentation post-processing
├── download_models.py   # Downloads the pinned ONNX models and verifies SHA-256
├── requirements.txt
├── render.yaml          # Render Blueprint
├── Procfile
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── README.md
└── LICENSE-REVIEW.md
```

## Current limitations

- Analysis takes roughly **20–25 seconds per image** on Render's free CPU.
- A **single Gunicorn worker** handles requests one at a time; concurrent users wait in a queue.
- Large photos or multi-image uploads can exceed request time limits (the custom domain's proxy times out
  after about 100 seconds; Gunicorn's timeout is 180 seconds) and use more memory.
- Free Render instances sleep when idle; the first request afterwards can take 50 seconds or more.
- Annotated result images are stored on the instance's temporary disk and are lost on redeploy or restart.
- **Severity is a heuristic estimate** based on detected damage area, model confidence and damage type.
  It is not a validated insurance, repair-cost or safety assessment.

## Roadmap

Claude is **not yet integrated**. It is planned as a reporting and reasoning layer on top of the
structured findings:

- Claude-generated, human-readable inspection reports from the structured findings
- Multi-image case summaries
- Exportable inspection reports
- Inspection history and case management
- API access
- Human-review workflow

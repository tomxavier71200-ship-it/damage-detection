# DamageLens AI

AI-assisted vehicle damage assessment and reporting.

DamageLens AI analyzes vehicle inspection images using two task-specific computer-vision models:

1. **Vehicle damage segmentation** to identify visible damage such as scratches, dents, cracks, broken lamps, glass damage, and flat tires.
2. **Vehicle-part segmentation** to identify vehicle components and associate detected damage with the most likely affected component.

The application then produces structured inspection findings including damage type, confidence, estimated size, severity, component match, image zone, and batch-level summary metrics.

## Product workflow

```text
Vehicle inspection photos
        ↓
Damage segmentation
        ↓
Vehicle-part segmentation
        ↓
Damage/component matching
        ↓
Severity + confidence estimation
        ↓
Structured inspection findings
```

The current application is an MVP focused on computer-vision-assisted inspection. A planned Claude layer can transform structured findings into human-readable inspection reports and assist with inspection workflow automation.

## Current features

- Upload one or multiple vehicle images
- Automatic use of the bundled task-specific models
- Damage segmentation
- Vehicle-part segmentation
- Damage-to-component matching
- Confidence bands
- Heuristic severity estimation
- Estimated damage size and image zone
- Annotated inspection images
- Batch-level inspection summary
- Adjustable damage confidence, parts confidence, and IoU thresholds

## Technology

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- NumPy
- Gunicorn
- Render

## Run locally

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Download the model weights:

```bash
python download_models.py
```

Start the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

The repository contains a `render.yaml` Blueprint configuration.

Build command:

```bash
pip install -r requirements.txt && python download_models.py
```

Start command:

```bash
gunicorn app:app
```

The model weights are downloaded during the build instead of being committed as large binary files to GitHub.

## Project structure

```text
damage-detection/
├── app.py
├── download_models.py
├── render.yaml
├── Procfile
├── requirements.txt
├── templates/
│   └── index.html
└── static/
    └── styles.css
```

## Product notes

The severity value is currently a **heuristic estimate** based on detected damage area, model confidence, and damage type. It is not a validated insurance, repair-cost, or safety assessment.

Model weights are downloaded from their respective model repositories during deployment. Review the applicable model and framework licenses before commercial use.

## Roadmap

- Claude-assisted inspection report generation
- Multi-image case summaries
- Exportable inspection reports
- Inspection history and case management
- API access
- Human-review workflow

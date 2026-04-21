


---
title: Car Damage Detection
emoji: 🚗
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Car Damage Detection Website

This project turns your Google Colab YOLO workflow into a local website.

## What it does

- Upload a YOLO model file like `trained.pt`
- Upload one or more car images
- Run damage detection in the browser-backed web app
- Show annotated images with generated damage reports

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
python app.py
```

4. Open the local URL shown in the terminal, usually `http://127.0.0.1:5000`.

## Deploy publicly on Render

This project is prepared for Render with `render.yaml` and a `Procfile`.

1. Push this folder to a GitHub repository.
2. Sign in to [Render](https://render.com/).
3. Create a new `Web Service` from your GitHub repo.
4. Render should detect:

```text
Build Command: pip install -r requirements.txt && python download_models.py
Start Command: gunicorn app:app
```

5. Deploy the service and wait for the first build to finish.
6. Open the public `onrender.com` URL Render gives you.

## Public deployment notes

- This project downloads the two task-specific model files during the Render build step.
- Do not commit large `.pt` weights into GitHub directly; they are ignored by `.gitignore`.
- The public site uses a damage-segmentation model plus a parts-segmentation model automatically.

## Notes

- The app expects a trained Ultralytics YOLO `.pt` model.
- Uploaded model files are deleted after processing.
- Result images are saved under `static/results/`.

# License review – DamageLens AI

> **Status: NEEDS LICENSE REVIEW BEFORE COMMERCIAL USE**
>
> This file records what the upstream sources state, as checked on 2026-10-08. It is not legal advice
> and it does not grant or change any license. Adding a license file to this repository does not remove
> obligations or restrictions that come with the models or the data they were trained on.

## Summary

| Component | License / terms | Commercial-use concern |
|---|---|---|
| Damage model (`car_damage_seg`) | AGPL-3.0 (model card) | **Yes – trained on CarDD, which requires PIC Lab authorization for commercial use** |
| CarDD dataset | CarDD Dataset Licensing (PIC Lab, CAS) | **Yes – commercial use, including testing commercial systems, needs prior authorization** |
| Parts model (`car_parts_seg`) | AGPL-3.0 (model card) | AGPL obligations; training-data license not verified |
| Ultralytics YOLO11 (base architecture, pipeline logic) | AGPL-3.0 (enterprise license available) | AGPL obligations for network use |
| ONNX conversions (`tommy33355/damagelens-onnx`) | Published as AGPL-3.0 | Carry the upstream terms above |

## 1. Damage model

- **Source:** <https://huggingface.co/harpreetsahota/car-dd-segmentation-yolov11> (`best.pt`, YOLO11x-seg).
- **License:** the model card states: "Since this model was trained using Ultralytics, it falls under the
  AGPL-3.0 License." The Hub metadata has no `license` field set.
- **Relevant restriction:** fine-tuned on the CarDD dataset (see section 2). The model card itself says the
  dataset is intended for non-commercial research and educational purposes.
- **What DamageLens uses:** an ONNX conversion of this model (`car_damage_seg.onnx`, weights unchanged),
  downloaded from `tommy33355/damagelens-onnx` at build time and run on every analysis.
- **Attribution required:** credit the author (Harpreet Sahota), the source model, Ultralytics and CarDD;
  keep the AGPL-3.0 notice; cite the CarDD paper in any publication or report that uses it.
- **Commercial-use concern:** **yes** – inherited from the CarDD training data (section 2).
- **Recommended next action:** before any commercial use (paid product, customer pilot, commercial
  testing), either obtain written authorization from the CarDD PIC Lab covering models trained on CarDD,
  or replace this model with one trained on data that permits commercial use.

## 2. CarDD dataset

- **Source:** <https://cardd-ustc.github.io>, licensing form
  <https://cardd-ustc.github.io/docs/CarDD_license.pdf>.
- **License / terms (summary of the licensing form):**
  - The dataset is the property of the PIC Lab, Chinese Academy of Sciences, and is protected by copyright.
  - It can be used for statistical and scientific research; prior consent from the PIC Lab is required.
  - **Any commercial use must be authorized by the PIC Lab first.** Commercial use includes, but is not
    limited to, *testing commercial systems*, using dataset images in advertisements, selling or
    broadcasting the data.
  - No redistribution of all or part of the dataset to third parties without prior authorization.
  - Publications using the data must cite: Wang, Li, Wu, "CarDD: A New Dataset for Vision-Based Car Damage
    Detection", IEEE T-ITS 24(7), 2023, doi:10.1109/TITS.2023.3258480.
- **What DamageLens uses:** no CarDD images are stored in or distributed by this repository or the model
  repository. The damage model was trained on CarDD. During development, four CarDD images from a public
  Hugging Face mirror (`harpreetsahota/CarDD`) were used locally to verify the ONNX conversion; they were
  not committed or published.
- **Commercial-use concern:** **yes.** The terms do not explicitly address trained model weights, but
  "making any commercial use of the dataset" and "testing commercial systems" are broad enough that a
  commercial product built on a CarDD-trained model needs review.
- **Recommended next action:** contact the PIC Lab (contact listed on the licensing form) to ask whether
  commercial use of models trained on CarDD is permitted and on what terms; do not use CarDD images in
  marketing material or demos without authorization.

## 3. Vehicle-parts model

- **Source:** <https://huggingface.co/Majorburn/yolov11-carparts-seg> (`best.pt`, YOLO11n-seg).
- **License:** AGPL-3.0 (model card metadata `license: agpl-3.0`).
- **Relevant restriction:** AGPL-3.0 obligations. The model card does not name its training data; its
  metadata points to Ultralytics' `carparts-seg` dataset (Ultralytics docs credit the Roboflow Universe
  dataset by Gianmarco Russo). **The license of that dataset was not verified in this review.**
- **What DamageLens uses:** an ONNX conversion (`car_parts_seg.onnx`, weights unchanged).
- **Attribution required:** credit the author (Majorburn), the source model and Ultralytics; keep the
  AGPL-3.0 notice.
- **Commercial-use concern:** no dataset restriction identified so far, but the training-data license is
  unverified and AGPL-3.0 obligations apply.
- **Recommended next action:** confirm the license of the carparts-seg / Roboflow dataset.

## 4. Ultralytics YOLO11 and `onnx_seg.py`

- **Source:** <https://github.com/ultralytics/ultralytics>, license AGPL-3.0; Ultralytics also offers a
  separate Enterprise license for closed-source/commercial use.
- **What DamageLens uses:** no Ultralytics package at runtime. The models are YOLO11 architectures, and
  `onnx_seg.py` reimplements Ultralytics' pre/post-processing (letterbox, NMS, mask decoding, box scaling)
  to reproduce its results, so it should be treated as derived from Ultralytics code.
- **Relevant obligations (AGPL-3.0):** if a modified/derived work is offered to users over a network, its
  complete corresponding source must be made available to those users under AGPL-3.0.
- **Current state:** this repository is public, but it has **no LICENSE file** and the web app does not
  link to its source code.
- **Recommended next action:** decide on the repository's license with the AGPL-3.0 obligations of the
  models and `onnx_seg.py` in mind (or obtain an Ultralytics Enterprise license); consider adding a
  "source code" link in the web app.

## 5. ONNX conversions (`tommy33355/damagelens-onnx`)

- Format conversions of the two models above; weights unchanged.
- Published as AGPL-3.0 with a model card that names both source models, Ultralytics and the CarDD
  non-commercial note.
- **They keep all upstream obligations and restrictions** (AGPL-3.0, and the CarDD terms for the damage
  model). Converting the format does not change the license or remove the dataset restriction.

## 6. Attribution / notices in this repository

- README credits both source models, Ultralytics and CarDD, and links here.
- Still missing: a LICENSE file for the repository's own code (see section 4) and a link to the source
  code from the running app.

## 7. Planned startup / commercial use

**NEEDS LICENSE REVIEW BEFORE COMMERCIAL USE.**

- Using the current damage model commercially has an apparent conflict with the CarDD terms unless the
  PIC Lab authorizes it.
- AGPL-3.0 does not forbid commercial use, but requires that users of the network service can obtain the
  complete corresponding source under AGPL-3.0.
- Demonstrating the project as a non-commercial research/prototype is the use that best matches the
  current terms; describe it that way until the review is done.

## Recommended next actions

1. Contact the CarDD PIC Lab about commercial use of a CarDD-trained model.
2. Get a legal review of the AGPL-3.0 implications and choose a repository license accordingly.
3. Verify the license of the carparts-seg (Roboflow) training dataset.
4. Plan a commercially clean damage model (training data with commercial rights) for any paid product.
5. Until then, present DamageLens AI as a research prototype and avoid using CarDD images in demos or
   marketing.

# Models

Only two fine-tuned models are part of this project. Weights are **not** in git (the server model file is 110 MB); download them from Hugging Face.

| Model | Role | Folder | Weights |
|---|---|---|---|
| Server model (`livestock-v1`) | server detector behind the Vision API | `livestock-v1/` | https://huggingface.co/enockstack/livestock-detector |
| YOLO11n (`yolo11n-weak-v1`) | on-device detector in the mobile app (exported to `.tflite`, bundled at `mobile/assets/models/`) | `yolo11n-weak-v1/` | `mobile/assets/models/model.tflite` (3 MB) |

Each folder holds `labels.json` and `model_card.json` (training run, dataset counts, SHA-256 of the weights).
Both were trained on weak whole-animal labels (3 classes: healthy, lumpy skin disease, foot-and-mouth disease), so they are development models, not lesion detectors.

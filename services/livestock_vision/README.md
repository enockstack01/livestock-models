# Livestock Vision API (Service A)

FastAPI service that hosts the fine-tuned server model and implements `/v1/detect`, `/v1/feedback`, `/v1/model`, `/health`, `/ready`
from the Logistack master plan (Part 3). Code: `src/livestock_vision/`. Shared class files: `shared/labels.json`,
`shared/livestock.yaml`, `shared/model_manifest.json` (training, API and apps must all read these).

```bash
# 1. register a fine-tuned model from a Kaggle run (writes models/<version>/{best.pt,labels.json,model_card.json} + updates shared/model_manifest.json)
python -m livestock_vision.register --run-dir runs/kaggle/server --version livestock-v1
# 2. serve it. Local testing: --dev makes a random secret and prints a token (open /docs, Authorize, paste it)
PYTHONPATH=src python -m livestock_vision.serve --version livestock-v1 --port 8001 --dev
# production: AGRICODERS_JWT_SECRET=<secret shared with the Agricoders login service> DATABASE_URL=postgresql+psycopg2://... python -m livestock_vision.serve --version ... --host 0.0.0.0
```
Optional: `VISION_MODEL_CARD` (JSON `{"thresholds": {"tick": 0.45}, "metrics": {...}}`: per-class thresholds, default 0.35), `VISION_CONF`,
`VISION_IMGSZ`, `VISION_DEVICE`, `PUBLIC_BASE_URL`, `VISION_RATE_LIMIT`, `LABELS_PATH`, `MANIFEST_PATH`. `VISION_AUTH_DISABLED=1` and
`VISION_ALLOW_UNMATCHED_CLASSES=1` exist for local development only.

## Behaviour (all covered by `tests/test_vision_api.py`, 18 tests)
- **JWT (HS256) auth** on every `/v1` route, fail-closed: no secret configured means 503, never open. `alg=none`, wrong signature, expired and
  expiry-less tokens are rejected. Roles `vet`/`admin` may read other users' records; everyone else gets 404 for records that are not theirs.
- **Never serves the wrong model.** At startup the model's class names must all exist in `labels.json`. A generic COCO model is refused
  (`/ready` 503 with the reason, `/v1/detect` 503 `model_not_ready`); `/health` stays 200. Checked against the real downloaded `server-base.pt`.
- **One model instance, one worker thread** (a test proves 6 concurrent requests never overlap inside the model). Run one uvicorn worker per container.
- **Input handling**: decodes the upload (never trusts filename/content-type), JPEG/PNG/WebP only, 10 MB limit, min 64 px, longest side capped at 1280,
  decompression-bomb guard, **EXIF/GPS stripped** from the stored copy. Species must be one of `labels.json`'s list.
- **Per-class thresholds**, a `flags` list (`possible_notifiable_disease`, `possible_zoonotic_risk`) driven by `labels.json`, annotated image per diagnosis.
- **Offline confirmation**: `/v1/detect` accepts the phone's YOLO11n result (`offline_result` form field). It is compared by finding class,
  the response carries `offline_comparison`, and disagreements are stored for priority retraining.
- **Feedback loop**: `/v1/feedback` (`confirmed`/`corrected`, corrected boxes). `python -m livestock_vision.export --out DIR [--only-disagreements]`
  writes YOLO `images/ labels/ classes.txt` that `kaggle_train/launch.py --annotated DIR` and `detect.dataset.build_dataset` consume.
- Common error format with `request_id`, `X-Request-ID` propagation, per-user rate limit, `/metrics` (Prometheus text).

## Differences from the plan's Appendix A skeleton
- `half=True` is deprecated in the pinned Ultralytics (8.4.191); the service uses `quantize=16` on GPU. The plan's `format="tflite", int8=True` export
  call is likewise deprecated: use `format="litert", quantize="int8"` (Linux/macOS only). Core ML export is macOS only.
- `save_detection` is implemented (SQLAlchemy, SQLite by default). The `diagnosis_id` is the key the LLM API reads.
- Species is a request field, not a detector class. If you want species *detection*, add classes with new ids.

## Measured with the real fine-tuned server model (`livestock-v1`, 2026-11-02)
The running service (real HTTP, real weights, CPU) was scored on the exact 191 validation photos from the Kaggle run, with their ground-truth boxes, at the plan's 0.35 threshold:
**precision 0.747, recall 0.775** overall (TP 148, FP 50, FN 43); per class: healthy P 0.76 / R 0.82, lumpy skin disease P 0.85 / R 0.64, FMD P 0.50 / R 0.62 (only 13 validation boxes: noise).
This agrees with the training-time numbers (mAP50 0.807, mAP50-95 0.566). **CPU latency is about 1.1 s per photo**; the plan's ~60 ms assumes an L4 GPU, which was not measured. The model was trained on weak
whole-animal boxes, so these numbers say nothing about finding lesions. The loop `/v1/feedback` -> `livestock_vision.export` -> `kaggle_train` `--annotated` was run on real detections and the
dataset builder accepted the export. Running the live server (rather than the in-process tests) found one bug that the tests missed (ultralytics patches `Image.open`; invalid uploads became 500s): fixed and covered by tests.

## Not built / not verified
- **There is no trained *lesion* model yet.** The registered models are weak whole-animal detectors; each carries its own generated `labels.json` (scope `animal-level (weak labels)`). The plan's lesion classes in
  `shared/labels.json` have no model: `/ready` is 503 for any model whose classes are not in the labels file. That draft class list from the plan's examples; the notifiable/zoonotic flags and the French names need veterinary review and the
  Kinyarwanda names are null on purpose (a native-speaking vet must supply them).
- Not run: the Docker image, PostgreSQL, S3/object storage for images (local disk is used), real JWTs from the Agricoders login service (only tokens signed with the shared secret in tests), GPU/FP16 inference, load/latency on an L4.
- Not part of this service: the LLM API, the vet advice pack, YOLO11n training/export, the mobile offline queue.

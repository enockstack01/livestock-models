# Livestock models

Livestock disease detection from photos, with two fine-tuned detection models and a Qwen-written advice report.

| Part | What it is | Where |
|---|---|---|
| **Server model** (`server-weak-v1`) | Server detector. Fine-tuned on Kaggle (T4, 226 min): mAP50 0.807 / mAP50-95 0.566 on 191 validation photos | `models/`, weights on [Hugging Face](https://huggingface.co/enockstack/livestock-detector) |
| **YOLO11n** (`yolo11n-weak-v1`) | Phone detector used when offline, exported to TFLite (3 MB): mAP50 0.769 / mAP50-95 0.512 | `models/`, `mobile/assets/models/` |
| Vision API | FastAPI around the server model: `/v1/detect`, feedback, JWT auth | `src/livestock_vision/`, `services/livestock_vision/` |
| LLM API | FastAPI wrapper that calls Qwen through any OpenAI-compatible endpoint (hosted or vLLM) and returns a safety-checked report | `src/livestock_llm/`, `services/livestock_llm/` |
| Web app | Photo, then boxes, then streamed report and chat | `web_app/` |
| Mobile app | React Native. Online: server model + Qwen. Offline: YOLO11n + bundled advice, queued and re-checked online | `mobile/` |
| Training | Kaggle GPU pipeline: pseudo-labels, fine-tune, LiteRT export | `kaggle_train/`, `src/livestock_dx/detect/`, `configs/` |

## Read this first: what the models are and are not
Both models were trained on **weak whole-animal labels** (one box per animal, labelled by the photo's folder: healthy / lumpy skin disease / foot-and-mouth disease; FMD has only 30 training boxes).
They do **not** locate lesions and are development models, not for field decisions. The advice text (`shared/advice_pack.json`) and translations have not been reviewed by a veterinarian.

## Run locally
```
pip install -e ".[vision,dev]"
huggingface-cli download enockstack/livestock-detector best.pt --local-dir models/server-weak-v1
python -m livestock_vision.serve --version server-weak-v1 --port 8001 --dev
python -m livestock_llm.dev_mock                       # MOCK Qwen for local demos only
python -m livestock_llm.serve --llm-url http://127.0.0.1:8003/v1 --model mock-qwen --port 8002 --dev
python web_app/serve.py                                # http://127.0.0.1:8080
pytest -q                                              # 75 tests
```
For a real Qwen, see `services/livestock_llm/README.md` (set `LLM_API_KEY` in the environment).

## Status and limits
- Python tests pass (75). Mobile logic has 31 Node tests (`cd mobile && node --test tests/*.test.ts`); the web helpers have 10.
- **No real Qwen call has been made** (only mock servers). The mobile screens and adapters have never been compiled or run on a device.
- Licensing is undecided: Ultralytics YOLO11 is AGPL-3.0, which applies to anything built on it. Choose a license before releasing.

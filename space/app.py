"""ZeroGPU demo of the fine-tuned livestock health detector, plus a plain-JSON API (api_name "detect") for LivestockPro's server."""
import os

import gradio as gr
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

try:
    import spaces
    gpu = spaces.GPU(duration=30)
except ImportError:                                   # running locally without the ZeroGPU package
    def gpu(fn):
        return fn

WEIGHTS = "best.pt"                                   # bundled in the Space repo, so no token is needed at runtime
if not os.path.exists(WEIGHTS):                       # local development: fetch from the model repo (needs a token while it is private)
    WEIGHTS = hf_hub_download(os.environ["MODEL_REPO"], "best.pt", token=os.environ.get("HF_TOKEN"))
model = YOLO(WEIGHTS)                                 # loaded on CPU; moved to the GPU inside _infer()

REPORTABLE = {"foot_and_mouth_disease", "lumpy_skin_disease"}
MODEL_VERSION = os.environ.get("MODEL_VERSION", "livestock-v1")


@gpu
def _infer(image, conf):
    """Runs on the GPU and returns only plain Python data: CUDA tensors cannot cross back from the ZeroGPU worker."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r = model.predict(image, conf=conf, imgsz=640, device=device, verbose=False)[0]
    return [(tuple(int(v) for v in b.xyxy[0].tolist()), r.names[int(b.cls)], float(b.conf)) for b in r.boxes]


def predict(image, conf):
    if image is None:
        raise gr.Error("Upload a photo first.")
    dets = _infer(image, conf)
    boxes = [(xyxy, f"{name.replace('_', ' ')} {score:.0%}") for xyxy, name, score in dets]
    msg = ""
    if {name for _, name, _ in dets} & REPORTABLE:
        msg = "**Possible reportable disease flagged.** Isolate the animal and contact the sector veterinary officer."
    elif not dets:
        msg = "Nothing recognised. That does not mean the animal is healthy: retake a closer, well-lit photo, or ask a vet."
    return (image, boxes), msg


def detect_json(image, conf):
    """Machine-readable endpoint for the LivestockPro server (api_name="detect"): plain JSON, no annotated image.
    Box values are pixels of the photo as uploaded."""
    if image is None:
        raise gr.Error("No image received.")
    conf = 0.25 if conf is None else min(0.9, max(0.05, float(conf)))
    dets = _infer(image, conf)
    return {
        "model_version": MODEL_VERSION,
        "image": {"width": image.width, "height": image.height},
        "detections": [
            {"label": name, "confidence": round(score, 4), "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
             "reportable": name in REPORTABLE}
            for (x1, y1, x2, y2), name, score in dets
        ],
    }


with gr.Blocks(title="Livestock Health Detector") as demo:
    gr.Markdown("# Livestock health detector")
    with gr.Row():
        inp = gr.Image(type="pil", label="Photo of the animal")
        out = gr.AnnotatedImage(label="Detections")
    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
    msg = gr.Markdown()
    gr.Button("Detect", variant="primary").click(predict, [inp, conf], [out, msg])
    gr.Examples([["example.jpg"]], inputs=inp)

    # Hidden components that only exist to expose /gradio_api/call/detect (used by the LivestockPro server).
    api_img = gr.Image(type="pil", visible=False)
    api_conf = gr.Number(value=0.25, visible=False)
    api_out = gr.JSON(visible=False)
    gr.Button(visible=False).click(detect_json, [api_img, api_conf], api_out, api_name="detect")

if __name__ == "__main__":
    demo.launch(show_error=True)

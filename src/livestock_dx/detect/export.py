"""Export a trained YOLO11 model for the offline mobile app and write the app's labels/metadata file.

  python -m livestock_dx.detect.export --weights runs/livestock/weights/best.pt --format onnx
  python -m livestock_dx.detect.export --weights ... --format litert --int8 --data data_yolo/dataset.yaml   (Linux/macOS; 'tflite' is an alias)
  python -m livestock_dx.detect.export --weights ... --format coreml --int8          (macOS only)

INT8 post-training quantisation needs representative images (the `data` yaml's val split is used for calibration);
export the real fine-tuned weights, and re-check mAP on the quantised model before shipping it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_for_mobile(weights: str, fmt: str, imgsz: int = 640, int8: bool = False, data: str | None = None,
                      out_dir: str | Path = "mobile/assets/models") -> Path:
    from ultralytics import YOLO

    model = YOLO(weights)
    fmt = "litert" if fmt == "tflite" else fmt                  # ultralytics >= 8.4.83: 'tflite' is a deprecated alias of 'litert'
    kw = {"format": fmt, "imgsz": imgsz, "nms": False}          # raw [1, 4+nc, N] output; NMS is done in the app
    if int8:
        kw.update(quantize="int8", data=data)                   # replaces the deprecated int8=True
    produced = Path(model.export(**kw))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / produced.name
    if produced.is_dir():
        shutil.copytree(produced, dest, dirs_exist_ok=True)
    else:
        shutil.copyfile(produced, dest)
    names = [model.names[i] for i in sorted(model.names)]
    (out_dir / "model.json").write_text(json.dumps({
        "installed": True,                                     # the app shows "no model" unless this is true
        "file": dest.name, "format": fmt, "imgsz": imgsz, "int8": int8, "task": model.task, "labels": names,
        # Ultralytics raw detection output is (1, 4+nc, anchors). The app's decoder also accepts channels_last and either
        # normalised or pixel boxes, but VERIFY on one real photo after export: a wrong layout gives scrambled boxes.
        "output_layout": "channels_first", "input_dtype": "float32",
        "input_layout": "nchw",       # verified on the LiteRT export: planar [1,3,H,W], NOT the NHWC most TFLite models use
        "output": "raw_yolo (1, 4+nc, anchors); boxes are cx,cy,w,h",
        "source_weights": str(weights)}, indent=2))
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", choices=["onnx", "tflite", "litert", "coreml"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--out-dir", default="mobile/assets/models")
    a = ap.parse_args()
    print(export_for_mobile(a.weights, a.format, a.imgsz, a.int8, a.data, a.out_dir))

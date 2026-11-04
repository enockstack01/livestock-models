"""Fine-tune server model.   python -m livestock_dx.detect.train --data data_yolo/dataset.yaml [--task segment]

Hyper-parameters live in configs/yolo_train.yaml; CLI flags override them (e.g. --epochs 1 --imgsz 320 for a smoke test).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = {"detect": ROOT / "pretrained" / "server-base.pt", "segment": ROOT / "pretrained" / "server-base-seg.pt"}


def load_hyp(path: Path = ROOT / "configs" / "yolo_train.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_weights(task: str) -> Path:
    """The COCO-pretrained base weights are a file you provide (see kaggle_train/README.md); nothing is downloaded behind your back."""
    w = WEIGHTS[task]
    if not w.exists():
        raise SystemExit(f"{w} not found: place the COCO-pretrained base weights there (the Kaggle launcher checks their SHA-256).")
    return w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=["detect", "segment"], default="detect")
    ap.add_argument("--weights", default=None, help="override the starting weights")
    ap.add_argument("--project", default=str(ROOT / "runs"))
    ap.add_argument("--name", default="livestock")
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1 or cpu (default: GPU if present)")
    for k in ("epochs", "imgsz", "batch", "workers", "patience"):
        ap.add_argument(f"--{k}", type=int)
    for k in ("lr0", "mixup", "mosaic", "cls", "box", "dfl"):
        ap.add_argument(f"--{k}", type=float)
    ap.add_argument("--optimizer")
    ap.add_argument("--freeze", type=int, help="freeze the first N layers (10 = whole YOLO11 backbone)")
    ap.add_argument("--time", type=float, help="wall-clock budget in hours; the epoch count and LR schedule are fitted to it")
    ap.add_argument("--fraction", type=float, help="train on this fraction of the training images")
    ap.add_argument("--profile", choices=["default", "cpu"], default="default",
                    help="'cpu' applies configs/yolo_train_cpu.yaml (small images, frozen backbone, low RAM)")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run (same --project/--name)")
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    hyp = load_hyp()
    if a.profile == "cpu":
        hyp.update(load_hyp(ROOT / "configs" / "yolo_train_cpu.yaml"))
    hyp.update({k: v for k, v in vars(a).items() if k in hyp and v is not None})
    hyp.update({k: getattr(a, k) for k in ("freeze", "time", "fraction") if getattr(a, k) is not None})
    last = Path(a.project) / a.name / "weights" / "last.pt"
    if a.resume:
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} does not exist")
        return YOLO(str(last)).train(resume=True)
    model = YOLO(a.weights or str(ensure_weights(a.task)))
    return model.train(data=a.data, project=a.project, name=a.name, device=a.device, exist_ok=True, **hyp)


if __name__ == "__main__":
    main()

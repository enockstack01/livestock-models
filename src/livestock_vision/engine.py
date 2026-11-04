"""Detection engine: class registry, the YOLO wrapper, image preparation, annotation, flags and offline comparison."""
from __future__ import annotations

import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from .config import Settings

Image.MAX_IMAGE_PIXELS = 60_000_000          # decompression-bomb guard
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageError(ValueError):
    pass


class Labels:
    """labels.json: permanent class ids, display names and screening flags."""

    def __init__(self, path: Path):
        self.raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.classes = self.raw["classes"]
        self.by_name = {c["name"]: c for c in self.classes}
        self.species = set(self.raw["species"])
        ids = [c["id"] for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate class ids in labels.json (ids must never be reused)")

    def active_names(self) -> list[str]:
        return [c["name"] for c in self.classes if c["status"] == "active"]


def prepare_image(data: bytes, max_side: int) -> Image.Image:
    """Validate by actually decoding (never trust the filename/content-type), strip EXIF (incl. GPS), cap the size."""
    try:
        im = Image.open(io.BytesIO(data))
        fmt = im.format
        im.load()
    except MemoryError:
        raise
    except Exception:
        # Deliberately broad. Pillow raises UnidentifiedImageError/OSError/SyntaxError/DecompressionBombError, but importing
        # ultralytics monkey-patches Image.open, and for an unrecognised file the patched version tries an optional HEIC
        # loader and raises ModuleNotFoundError. Found only on the live server: untrusted bytes must never become a 500.
        raise ImageError("The file is not a readable image (JPEG, PNG or WebP).")
    if fmt not in ALLOWED_FORMATS:
        raise ImageError(f"Unsupported image format {fmt}; use JPEG, PNG or WebP.")
    if min(im.size) < 64:
        raise ImageError("The image is too small (minimum 64 px).")
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def annotate(im: Image.Image, dets: list[dict]) -> Image.Image:
    out = im.copy()
    d = ImageDraw.Draw(out)
    w = max(2, round(max(out.size) / 300))
    for x in dets:
        x1, y1, x2, y2 = x["box"]
        d.rectangle([x1, y1, x2, y2], outline=(229, 57, 53), width=w)
        label = f"{x['class']} {x['confidence']:.2f}"
        d.rectangle([x1, max(0, y1 - 12), x1 + 6 * len(label) + 4, y1], fill=(229, 57, 53))
        d.text((x1 + 2, max(0, y1 - 11)), label, fill=(255, 255, 255))
    return out


def summarise(dets: list[dict]) -> dict:
    s: dict[str, int] = {}
    for d in dets:
        s[d["class"]] = s.get(d["class"], 0) + 1
    return s


def flags_for(dets: list[dict], labels: Labels) -> list[str]:
    f = []
    names = {d["class"] for d in dets}
    if any(labels.by_name.get(n, {}).get("notifiable_suspect") for n in names):
        f.append("possible_notifiable_disease")
    if any(labels.by_name.get(n, {}).get("zoonotic_suspect") for n in names):
        f.append("possible_zoonotic_risk")
    return f


def compare_offline(online: list[dict], offline: list[dict]) -> dict:
    """Does the on-device YOLO11n result agree with the YOLO11x result (compared by finding class)?"""
    on, off = {d["class"] for d in online}, {d["class"] for d in offline}
    return {"agrees": on == off, "only_online": sorted(on - off), "only_offline": sorted(off - on)}


class YoloDetector:
    """Ultralytics YOLO, loaded once. One model instance must not be called from several threads at once, so every
    call goes through the engine's single-worker executor."""

    def __init__(self, weights: str, device: str = "auto", imgsz: int = 640):
        import torch
        from ultralytics import YOLO
        cuda = torch.cuda.is_available()
        self.device = (0 if cuda else "cpu") if device == "auto" else (int(device) if device.isdigit() else device)
        self.half = cuda and self.device != "cpu"                 # FP16 only on GPU
        self.imgsz = imgsz
        self.model = YOLO(weights)
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.model.predict(Image.new("RGB", (imgsz, imgsz)), **self._kw(imgsz))

    def _kw(self, imgsz: int) -> dict:
        kw = {"imgsz": imgsz, "device": self.device, "verbose": False}
        if self.half:
            kw["quantize"] = 16            # FP16 on GPU. Replaces the deprecated half=True (ultralytics >= 8.4)
        return kw

    def predict(self, im: Image.Image, conf: float) -> list[dict]:
        r = self.model.predict(im, conf=conf, **self._kw(self.imgsz))[0]
        return [{"class": self.names[int(c)], "confidence": round(float(p), 4), "box": [round(v, 1) for v in b]}
                for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


class Engine:
    def __init__(self, settings: Settings, detector=None):
        self.s = settings
        self.labels = Labels(settings.labels_path)
        self.exec = ThreadPoolExecutor(max_workers=1)
        self.detector, self.ready, self.reason = None, False, "model not loaded"
        self.thresholds: dict[str, float] = {}
        self.metrics: dict = {}
        try:
            self._load(detector)
        except Exception as e:                              # the service stays up (/health) but reports not ready
            self.reason = f"{type(e).__name__}: {str(e)[:200]}"

    def _load(self, detector):
        if detector is None:
            if not self.s.weights or not Path(self.s.weights).exists():
                self.reason = "no weights configured (set VISION_WEIGHTS)"
                return
            detector = YoloDetector(self.s.weights, self.s.device, self.s.imgsz)
        unknown = sorted(set(detector.names.values()) - set(self.labels.by_name))
        if unknown and not self.s.allow_unmatched_classes:
            self.reason = (f"model classes {unknown[:5]} are not in labels.json: refusing to serve (a generic COCO model must "
                           "not be presented as a livestock-finding detector)")
            return
        if self.s.model_card_path and Path(self.s.model_card_path).exists():
            card = json.loads(Path(self.s.model_card_path).read_text(encoding="utf-8"))
            self.thresholds = {k: float(v) for k, v in card.get("thresholds", {}).items()}
            self.metrics = card.get("metrics", {})
        self.detector, self.ready, self.reason = detector, True, "ok"

    def threshold(self, cls: str) -> float:
        return self.thresholds.get(cls, self.s.default_threshold)

    def _detect_sync(self, im: Image.Image) -> list[dict]:
        floor = min([self.s.default_threshold, *self.thresholds.values()])
        return [d for d in self.detector.predict(im, floor) if d["confidence"] >= self.threshold(d["class"])]

    async def detect(self, im: Image.Image) -> list[dict]:
        return await asyncio.get_running_loop().run_in_executor(self.exec, self._detect_sync, im)

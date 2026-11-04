"""Build a YOLO-format dataset (detect or segment) + dataset.yaml from annotated images.

Source layout (the standard YOLO / Roboflow / CVAT "YOLO 1.1" export):  <src>/images/*.jpg  <src>/labels/*.txt
Label line, detection:     class_id x_center y_center width height           (all normalised to 0..1)
Label line, segmentation:  class_id x1 y1 x2 y2 x3 y3 ...                    (polygon, >= 3 points, 0..1)

Nothing is trusted: malformed lines/files are dropped and counted, near-duplicate images are removed with a
perceptual hash BEFORE the train/val split, and the split is a hash of the image content, so the same photo can
never land on both sides.
"""
from __future__ import annotations

import hashlib
import shutil
import zlib
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from ..clean.images import NearDupIndex, dhash

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Target pathology classes. Lesion-level classes need real box/polygon annotations (see README: none are
# publicly available, so these must be annotated by you or a vet before lesion-level training is meaningful).
DEFAULT_CLASSES = ["lumpy_skin_nodules", "foot_and_mouth_lesions", "tick_clusters", "injuries"]


def parse_label_line(line: str, nc: int) -> tuple[int, list[float]]:
    """-> (class_id, coords). Raises ValueError with the reason if the line is not valid YOLO."""
    t = line.split()
    if len(t) < 5:
        raise ValueError("too few values")
    try:
        cid, v = int(t[0]), [float(x) for x in t[1:]]
    except ValueError:
        raise ValueError("non-numeric value")
    if not 0 <= cid < nc:
        raise ValueError(f"class id {cid} outside 0..{nc - 1}")
    if any(not (0.0 <= x <= 1.0) for x in v):
        raise ValueError("coordinate outside 0..1 (labels must be normalised)")
    if len(v) == 4:
        if v[2] <= 0 or v[3] <= 0:
            raise ValueError("zero-size box")
    elif len(v) >= 6 and len(v) % 2 == 0:
        pass                                                   # polygon with >= 3 points
    else:
        raise ValueError("neither a box (4 values) nor a polygon (even count >= 6)")
    return cid, v


def polygon_to_box(v: list[float]) -> list[float]:
    xs, ys = v[0::2], v[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def clean_label_file(path: Path, nc: int, task: str) -> tuple[list[str], list[str]]:
    """-> (valid lines, error messages). task='detect' converts polygons to boxes;
    task='segment' keeps polygons and drops plain boxes (segmentation training needs masks)."""
    good, errs = [], []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cid, v = parse_label_line(line, nc)
        except ValueError as e:
            errs.append(f"{Path(path).name}:{i}: {e}")
            continue
        is_poly = len(v) > 4
        if task == "detect" and is_poly:
            v, is_poly = polygon_to_box(v), False
        if task == "segment" and not is_poly:
            errs.append(f"{Path(path).name}:{i}: box in a segmentation dataset (needs a polygon)")
            continue
        good.append(f"{cid} " + " ".join(f"{x:.6f}" for x in v))
    return good, errs


def split_of(key: str, val_pct: int = 15, test_pct: int = 0) -> str:
    u = zlib.crc32(key.encode()) % 100
    return "test" if u < test_pct else "val" if u < test_pct + val_pct else "train"


def build_dataset(sources: list[Path], out_dir: Path, classes: list[str], task: str = "detect",
                  val_pct: int = 15, dedupe_radius: int = 3, keep_negatives: bool = True) -> dict:
    """Validate, de-duplicate and split annotated images into out_dir (train/val) and write dataset.yaml.
    keep_negatives: images with an empty label file are kept as background examples (reduces false alarms)."""
    assert task in ("detect", "segment")
    out_dir = Path(out_dir)
    idx = NearDupIndex(dedupe_radius)
    stats: Counter = Counter()
    per_class: Counter = Counter()
    errors: list[str] = []
    for sp in ("train", "val"):
        (out_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for src in map(Path, sources):
        for img in sorted(p for p in (src / "images").rglob("*") if p.suffix.lower() in IMG_EXT):
            stats["images_seen"] += 1
            lab = src / "labels" / img.relative_to(src / "images").with_suffix(".txt")
            try:
                with Image.open(img) as im:
                    im.load()
                    h = dhash(im)
            except Exception:
                stats["dropped_unreadable_image"] += 1
                continue
            lines: list[str] = []
            if lab.exists():
                lines, errs = clean_label_file(lab, len(classes), task)
                errors += errs
                stats["dropped_bad_label_lines"] += len(errs)
            if not lines and not keep_negatives:
                stats["dropped_no_labels"] += 1
                continue
            if not idx.add_if_new(h):
                stats["dropped_duplicate_image"] += 1
                continue
            digest = hashlib.sha1(img.read_bytes()).hexdigest()[:16]
            sp = split_of(digest, val_pct)
            name = f"{digest}{img.suffix.lower()}"
            shutil.copyfile(img, out_dir / "images" / sp / name)
            (out_dir / "labels" / sp / f"{digest}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[f"kept_{sp}"] += 1
            if not lines:
                stats["kept_background"] += 1
            for ln in lines:
                per_class[classes[int(ln.split()[0])]] += 1
    write_dataset_yaml(out_dir, classes)
    return {**stats, "instances_per_class": dict(per_class), "label_errors": errors[:50]}


def write_dataset_yaml(out_dir: Path, classes: list[str]) -> Path:
    cfg = {"path": str(Path(out_dir).resolve()), "train": "images/train", "val": "images/val",
           "names": {i: c for i, c in enumerate(classes)}}
    p = Path(out_dir) / "dataset.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p

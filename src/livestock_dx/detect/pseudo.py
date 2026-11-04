"""WEAK, animal-level pseudo-labels from image-level data.

Our classification folders (vetdata/<Species>___<Condition>/*.jpg) say *what condition the animal in the photo has*, not
*where*. This script runs the COCO-pretrained labeller to find the animal (COCO has horse/sheep/cow, not pig or goat)
and writes ONE box around the largest animal, labelled with the folder's condition.

What this is good for: a pipeline test, a diseased-animal detector, and a starting point you can then correct by hand.
What it is NOT: lesion localisation. A box around a whole cow labelled 'lumpy_skin_disease' does not teach the
network where the nodules are. Lesion-level classes (lumpy_skin_nodules, tick_clusters, ...) need human annotation.
"""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

SEP = "___"   # class folders are <Species>___<Condition> (same constant as livestock_dx.web.labels, kept local so this
                # module imports cleanly on Kaggle, where importing the web package would try to create folders)

COCO_ANIMALS = {17: "horse", 18: "sheep", 19: "cow"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slug(condition: str) -> str:
    return condition.lower().replace(" ", "_").replace("-", "_")


def pseudo_label_animals(weights: str, data_dir: Path, out_src: Path, conf: float = 0.35, imgsz: int = 640,
                         limit_per_class: int | None = None, batch: int = 4) -> tuple[list[str], dict]:
    """-> (class names, stats). Writes out_src/images and out_src/labels (input to build_dataset)."""
    from ultralytics import YOLO

    data_dir, out_src = Path(data_dir), Path(out_src)
    folders = sorted(d for d in data_dir.iterdir() if d.is_dir() and SEP in d.name and not d.name.startswith(("_", ".")))
    classes = sorted({slug(d.name.split(SEP, 1)[1]) for d in folders})
    (out_src / "images").mkdir(parents=True, exist_ok=True)
    (out_src / "labels").mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    stats: Counter = Counter()
    for d in folders:
        cid = classes.index(slug(d.name.split(SEP, 1)[1]))
        files = [p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT][:limit_per_class]
        # Chunk the file list ourselves. Passing the whole list with stream=True and relying on ultralytics' `batch` argument
        # sent all 167 FMD photos through the GPU as ONE batch (6 GB tensors, CUDA out of memory on a 16 GB T4).
        def results():
            for i in range(0, len(files), batch):
                yield from model.predict([str(p) for p in files[i:i + batch]], conf=conf, classes=list(COCO_ANIMALS),
                                         imgsz=imgsz, batch=batch, verbose=False)

        for r in results():
            stats["images"] += 1
            if len(r.boxes) == 0:
                stats["no_animal_found"] += 1
                continue
            areas = (r.boxes.xywhn[:, 2] * r.boxes.xywhn[:, 3])
            x, y, w, h = r.boxes.xywhn[int(areas.argmax())].tolist()
            name = f"{d.name}_{Path(r.path).stem}".replace(" ", "_")
            shutil.copyfile(r.path, out_src / "images" / (name + Path(r.path).suffix.lower()))
            (out_src / "labels" / f"{name}.txt").write_text(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            stats["labelled"] += 1
            stats[f"class:{classes[cid]}"] += 1
    return classes, dict(stats)


if __name__ == "__main__":
    import argparse
    import json

    from .dataset import build_dataset
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="pretrained/server-base.pt")
    ap.add_argument("--data-dir", default="vetdata")
    ap.add_argument("--out", default="data_yolo_weak")
    ap.add_argument("--limit-per-class", type=int)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    import psutil, sys
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10)   # stay polite
    out = Path(a.out)
    names, st = pseudo_label_animals(a.weights, Path(a.data_dir), out / "_src", imgsz=a.imgsz, limit_per_class=a.limit_per_class)
    print(json.dumps(st, indent=1))
    print(json.dumps(build_dataset([out / "_src"], out, names, "detect"), indent=1))

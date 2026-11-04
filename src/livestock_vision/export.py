"""Turn stored feedback into YOLO training data (the retraining loop).

  python -m livestock_vision.export --out data_yolo_feedback [--only-disagreements]

Writes  <out>/images/*.jpg  <out>/labels/*.txt  <out>/classes.txt  : the layout livestock_dx.detect.dataset.build_dataset
and kaggle_train/launch.py --annotated both consume. The latest feedback per diagnosis wins. Boxes are only as good as the
vet who drew them: review a sample before training on them.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from sqlalchemy import select

from .config import Settings
from .db import Detection, Feedback, make_session_factory
from .engine import Labels


def export_corrections(settings: Settings, out: Path, only_disagreements: bool = False) -> dict:
    labels = Labels(settings.labels_path)
    ids = {c["name"]: c["id"] for c in labels.classes}
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    names = [c["name"] for c in sorted(labels.classes, key=lambda c: c["id"])]
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")   # position == class id (ids are 0..n-1, stable)
    assert [ids[n] for n in names] == list(range(len(names))), "labels.json ids must be 0..n-1 for classes.txt"
    stats = {"exported": 0, "skipped_missing_image": 0, "skipped_disagreement_filter": 0, "boxes": 0}
    Session = make_session_factory(settings.db_url)
    with Session() as db:
        latest: dict[str, Feedback] = {}
        for fb in db.execute(select(Feedback).order_by(Feedback.id)).scalars():
            latest[fb.diagnosis_id] = fb
        for did, fb in latest.items():
            rec = db.get(Detection, did)
            if only_disagreements and not rec.disagreement:
                stats["skipped_disagreement_filter"] += 1
                continue
            if not Path(rec.image_path).exists():
                stats["skipped_missing_image"] += 1
                continue
            with Image.open(rec.image_path) as im:
                w, h = im.size
            lines = []
            for f in json.loads(fb.findings):
                x1, y1, x2, y2 = (max(0.0, f["box"][0]), max(0.0, f["box"][1]), min(float(w), f["box"][2]), min(float(h), f["box"][3]))
                if x2 <= x1 or y2 <= y1 or f["class"] not in ids:
                    continue
                lines.append(f"{ids[f['class']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            shutil.copyfile(rec.image_path, out / "images" / f"{did}.jpg")
            (out / "labels" / f"{did}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))   # empty file = verified background
            stats["exported"] += 1
            stats["boxes"] += len(lines)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-disagreements", action="store_true", help="only photos where YOLO11n and server model disagreed")
    a = ap.parse_args()
    print(export_corrections(Settings.from_env(), Path(a.out), a.only_disagreements))

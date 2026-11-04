"""Register a fine-tuned model from a Kaggle run so the Vision API can serve it.

  python -m livestock_vision.register --run-dir runs/kaggle/yolo11x --version yolo11x-livestock-weak-v1 [--models-dir models]

Reads the run's best.pt + run_summary.json and writes   models/<version>/{best.pt, model_card.json, labels.json}
and updates shared/model_manifest.json. Prints the environment variables that make the API serve it.

The labels file is derived from the classes the model was ACTUALLY trained on, with a status per class:
'active' (had training boxes) or 'untrained' (zero training instances: the model cannot detect it). Nothing here
upgrades what the model is: a weak-label run is registered as animal-level, with that scope written into the card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .config import ROOT

NOTIFIABLE = {"lumpy_skin_disease", "foot_and_mouth_disease"}      # screening hints for the flags; vets must confirm


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def labels_from_run(summary: dict) -> dict:
    inst = summary.get("dataset", {}).get("instances_per_class", {})
    classes = []
    for i, name in enumerate(summary["classes"]):
        classes.append({
            "id": i, "name": name, "status": "active" if inst.get(name, 0) > 0 else "untrained",
            "notifiable_suspect": name in NOTIFIABLE, "zoonotic_suspect": False,
            "suggests": name.replace("_", " "), "training_instances": inst.get(name, 0),
            "display": {"en": "Healthy animal" if name == "healthy" else name.replace("_", " ").capitalize(), "fr": None, "rw": None}})
    return {"version": 1, "scope": "animal-level (weak labels)", "reviewed_by_veterinarian": False,
            "note": "Classes of a weak-label model: one whole-animal box per photo labelled with the photo's condition. Not lesion "
                    "localisation. 'untrained' classes had zero boxes and cannot be detected. Generated from the training run; do not edit ids.",
            "species": ["cattle", "goat", "sheep", "pig", "poultry"], "classes": classes}


def register(run_dir: Path, version: str, models_dir: Path, manifest_path: Path | None = None) -> dict:
    run_dir, models_dir = Path(run_dir), Path(models_dir)
    best = next(iter(run_dir.rglob("best.pt")), None)
    sfile = next(iter(run_dir.rglob("run_summary.json")), None)
    if best is None or sfile is None:
        raise SystemExit(f"{run_dir} must contain best.pt and run_summary.json (pull them with kaggle_train/launch.py pull)")
    summary = json.loads(sfile.read_text(encoding="utf-8"))
    if "metrics" not in summary or "error" in summary["metrics"]:
        raise SystemExit(f"the run has no validation metrics: {summary.get('metrics')}")
    out = models_dir / version
    if out.exists():
        raise SystemExit(f"{out} already exists: versions are immutable, choose a new --version")
    out.mkdir(parents=True)
    shutil.copyfile(best, out / "best.pt")
    labels = labels_from_run(summary)
    (out / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    m = summary["metrics"]
    warnings = []
    if summary.get("warning"):
        warnings.append(summary["warning"])
    for c in labels["classes"]:
        if c["status"] == "untrained":
            warnings.append(f"class '{c['name']}' had no training boxes: the model cannot detect it")
    card = {"version": version, "model": summary.get("model"), "mode": summary.get("mode"), "scope": labels["scope"],
            "weights_sha256": _sha(out / "best.pt"), "trained_on": summary.get("gpu"), "train_minutes": summary.get("train_minutes"),
            "dataset": summary.get("dataset"), "config": summary.get("config"),
            "thresholds": {},                                    # none tuned yet: the service default applies to every class
            "thresholds_note": "Not tuned per class yet; tune on the validation set before relying on any threshold.",
            "metrics": m, "warnings": warnings}
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    mp = Path(manifest_path or ROOT / "shared" / "model_manifest.json")
    man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    key = "yolo11n" if (summary.get("model") == "yolo11n") else "yolo11x"
    entry = man.setdefault(key, {})
    if key == "yolo11x":
        entry.update(version=version, dataset_version=card["weights_sha256"][:12], weights=(out / "best.pt").as_posix())
    else:
        entry.update(latest=version, weights=(out / "best.pt").as_posix())
    mp.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return {"dir": str(out), "env": {
        "VISION_WEIGHTS": str(out / "best.pt"), "VISION_MODEL_VERSION": version, "VISION_MODEL_CARD": str(out / "model_card.json"),
        "LABELS_PATH": str(out / "labels.json")}, "warnings": warnings, "mAP50": m.get("mAP50"), "mAP50-95": m.get("mAP50-95")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    a = ap.parse_args()
    r = register(Path(a.run_dir), a.version, Path(a.models_dir))
    print(json.dumps({k: v for k, v in r.items() if k != "env"}, indent=2))
    print("\nTo serve it:\n" + "\n".join(f"  set {k}={v}" for k, v in r["env"].items()))

"""Kaggle notebook/script: fine-tune YOLO11x on a Kaggle GPU for livestock pathology.

Attach these (private) datasets to the notebook; kaggle_train/launch.py creates and attaches them for you:
  livestock-yolo-weights    yolo11x.pt  (COCO-pretrained, SHA-256 verified before upload)
  livestock-yolo-code       src/livestock_dx + configs/yolo_train.yaml
  livestock-yolo-data       class folders  <Species>___<Condition>/*.jpg  (image-level labels)
  livestock-yolo-annotated  OPTIONAL: a YOLO export  images/ labels/ classes.txt  (real boxes or polygons)

Mode 'annotated' (if livestock-yolo-annotated is attached): trains on your real lesion boxes/masks.  <- what you want
Mode 'weak' (otherwise): COCO YOLO11x boxes ONE whole animal per photo and labels it with the folder's condition.
  That is a diseased-ANIMAL detector, not a lesion detector, and photos with no animal found (e.g. FMD mouth/hoof
  close-ups) get no label at all. It exists so the pipeline can run before annotations do.
Everything is written to /kaggle/working (kept as the notebook's output), including run_summary.json.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = dict(
    model="yolo11x",         # yolo11x (server) | yolo11n (phone): same data, same classes, different weights
    mode="auto",             # auto | annotated | weak
    task="detect",           # detect | segment (segment needs polygon labels; starts from yolo11x-seg.pt)
    imgsz=640,               # 1024 for tiny lesions/ticks if memory allows (lower batch)
    epochs=120,
    time_hours="auto",       # wall-clock cap so the schedule finishes inside Kaggle's session limit (None = epochs only)
    batch=None,              # None = per-model default below; OOM -> batch is halved automatically
    multi_scale=0.25,        # +-25 % image size per batch (0.5 needs a bigger batch reduction)
    weak_conf=0.35,          # COCO detector confidence for the weak boxes
    weak_limit_per_class=None,
    export=["onnx", "litert"],   # litert = .tflite for the mobile app (INT8); best-effort; failures are logged, never fatal
    allow_cpu=False,         # refuse to silently train YOLO11x on a Kaggle CPU for hours
    dry_run=False,           # build the dataset and stop (no training)
)
PER_MODEL = {"yolo11x": {"batch": 6, "time_hours": 8.0},       # x at 640 + multi-scale on a 16 GB T4
             "yolo11n": {"batch": 32, "time_hours": 3.0}}      # n is tiny: big batches, short schedule
# Local testing / overrides:  KAGGLE_INPUT_DIR, KAGGLE_WORKING_DIR, LDX_CONFIG_JSON='{"epochs":1,"imgsz":320,...}'
INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
CONFIG.update(json.loads(os.environ.get("LDX_CONFIG_JSON", "{}")))
if CONFIG["model"] not in PER_MODEL:
    raise SystemExit(f"model must be one of {sorted(PER_MODEL)}")
for _k, _v in PER_MODEL[CONFIG["model"]].items():
    if CONFIG[_k] in (None, "auto") and not (_k == "time_hours" and CONFIG[_k] is None):
        CONFIG[_k] = _v
# Pinned Ultralytics/YOLO11 release a01aaa06...: SHA-256 as published by Hugging Face
SHA256 = {"yolo11x.pt": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
          "yolo11n.pt": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"}


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_one(pattern: str, what: str, required=True):
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    if not hits and required:
        raise SystemExit(f"Missing input: {what} (looked for {pattern} under {INPUT}). Attach the dataset to this notebook.")
    return Path(hits[0]) if hits else None


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ULTRALYTICS = "8.4.191"          # the version this pipeline was tested with; export/trainer arguments changed between releases
os.environ.setdefault("YOLO_AUTOINSTALL", "False")                # never let ultralytics pip-install things mid-run (hangs offline)
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / ".yolo_config"))
WORK.mkdir(parents=True, exist_ok=True)


def has_internet() -> bool:
    import socket
    try:
        socket.setdefaulttimeout(4)
        socket.gethostbyname("pypi.org")
        return True
    except OSError:
        return False


def importable(mod: str) -> bool:
    import importlib
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode


def ensure_ultralytics() -> bool:
    """Install ultralytics from the attached wheels dataset (no internet needed); fall back to PyPI. Returns has_internet."""
    online = has_internet()
    say("internet:", "yes" if online else "NO (offline install from the attached wheels dataset)")
    try:
        import ultralytics
        if ultralytics.__version__ == ULTRALYTICS:
            return online
        say(f"ultralytics {ultralytics.__version__} installed, pipeline tested with {ULTRALYTICS}")
    except ImportError:
        pass
    wheels = find_one("ultralytics-*.whl", "livestock-yolo-wheels", required=False)
    if wheels is not None:
        wd = wheels.parent
        ok = pip("--no-index", "--find-links", str(wd), "--no-deps", f"ultralytics=={ULTRALYTICS}", "ultralytics-thop") == 0
        for mod, pkgs in (("polars", ["polars", "polars-runtime-32"]), ("cv2", ["opencv-python-headless"])):
            if ok and not importable(mod):
                ok = pip("--no-index", "--find-links", str(wd), "--no-deps", *pkgs) == 0
        if ok:
            return online
        say("WARNING: offline install from wheels failed")
    if online and pip(f"ultralytics=={ULTRALYTICS}") == 0:
        return online
    if importable("ultralytics"):
        say("WARNING: could not install the tested version; continuing with the installed one")
        return online
    raise SystemExit("ultralytics could not be installed: attach the livestock-yolo-wheels dataset (kaggle_train/launch.py prepare) "
                     "or enable Internet in the notebook settings (needs a phone-verified Kaggle account)")


def locate_inputs() -> dict:
    code_src = find_one("src/livestock_dx/__init__.py", "livestock-yolo-code").parents[1]
    weights_name = f"{CONFIG['model']}-seg.pt" if CONFIG["task"] == "segment" else f"{CONFIG['model']}.pt"
    weights = find_one(weights_name, "livestock-yolo-weights", required=False)
    if weights is None:
        raise SystemExit(f"{weights_name} not found in the attached datasets. Add it to livestock-yolo-weights.")
    # the weak labels always come from the COCO YOLO11x, so yolo11x and yolo11n train on the SAME dataset
    label_weights = find_one("yolo11x.pt", "livestock-yolo-weights", required=False)
    for w in (weights, label_weights):
        if w is not None and w.name in SHA256 and sha256(w) != SHA256[w.name]:
            raise SystemExit(f"{w.name} checksum mismatch: refusing to load an unverified pickle checkpoint.")
    ann = None
    for marker in ("classes.txt", "dataset.yaml", "data.yaml"):
        for m in sorted(glob.glob(str(INPUT / "**" / marker), recursive=True)):
            root = Path(m).parent
            if (root / "images").is_dir() and (root / "labels").is_dir():
                ann = root
                break
        if ann:
            break
    data_dirs = sorted({Path(p).parent for p in glob.glob(str(INPUT / "**" / "*___*" / "*.jpg"), recursive=True)})
    data_root = sorted({d.parent for d in data_dirs})
    return dict(code=code_src.parent, weights=weights, label_weights=label_weights, annotated=ann,
                data=data_root[0] if data_root else None)


def read_classes(root: Path) -> list[str]:
    import yaml
    if (root / "classes.txt").exists():
        return [l.strip() for l in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    for y in ("dataset.yaml", "data.yaml"):
        if (root / y).exists():
            names = yaml.safe_load((root / y).read_text(encoding="utf-8"))["names"]
            return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    raise SystemExit(f"{root}: need classes.txt (one class per line) or a dataset.yaml with names")


def main():
    t0 = time.time()
    online = ensure_ultralytics()
    import torch
    inp = locate_inputs()
    sys.path.insert(0, str(inp["code"] / "src"))
    from livestock_dx.detect.dataset import build_dataset
    from livestock_dx.detect.pseudo import pseudo_label_animals

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    say("GPU:", gpu or "NONE", "| inputs:", {k: str(v) for k, v in inp.items()})
    if gpu is None and not CONFIG["allow_cpu"] and not CONFIG["dry_run"]:
        raise SystemExit("No GPU is attached. In the notebook settings choose Accelerator = GPU (T4/P100), or set allow_cpu.")

    ds_dir = WORK / "dataset"
    shutil.rmtree(ds_dir, ignore_errors=True)
    mode = CONFIG["mode"] if CONFIG["mode"] != "auto" else ("annotated" if inp["annotated"] else "weak")
    warning = None
    if mode == "annotated":
        if not inp["annotated"]:
            raise SystemExit("mode=annotated but no annotated dataset (images/ labels/ classes.txt) is attached")
        classes = read_classes(inp["annotated"])
        stats = build_dataset([inp["annotated"]], ds_dir, classes, CONFIG["task"])
    else:
        if not inp["data"]:
            raise SystemExit("mode=weak needs the livestock-yolo-data dataset (<Species>___<Condition> folders)")
        warning = ("WEAK LABELS: one whole-animal box per photo from the COCO detector, labelled with the folder's condition. "
                   "This is not lesion localisation; photos without a detectable animal (e.g. FMD mouth/hoof close-ups, pigs) are unlabelled.")
        say("WARNING:", warning)
        if CONFIG["task"] != "detect":
            raise SystemExit("weak mode produces boxes only; use task='detect', or attach real polygon annotations for 'segment'")
        if inp["label_weights"] is None:
            raise SystemExit("yolo11x.pt (the COCO labeller) must be in livestock-yolo-weights, even when training yolo11n")
        classes, pst = pseudo_label_animals(str(inp["label_weights"]), inp["data"], ds_dir / "_src", conf=CONFIG["weak_conf"],
                                            imgsz=CONFIG["imgsz"], limit_per_class=CONFIG["weak_limit_per_class"])
        stats = {"pseudo_label": pst, **build_dataset([ds_dir / "_src"], ds_dir, classes, "detect")}
        empty = [c for c in classes if stats.get("instances_per_class", {}).get(c, 0) == 0]
        if empty:
            say("WARNING: classes with ZERO labelled boxes (the model cannot learn them):", empty)
    say("dataset:", json.dumps(stats)[:1500])
    summary = {"model": CONFIG["model"], "mode": mode, "warning": warning, "gpu": gpu, "classes": classes, "dataset": stats, "config": CONFIG}
    (WORK / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if CONFIG["dry_run"]:
        say("dry_run: stopping before training")
        return

    import yaml
    from ultralytics import YOLO
    hyp = yaml.safe_load((inp["code"] / "configs" / "yolo_train.yaml").read_text(encoding="utf-8"))
    hyp.update(imgsz=CONFIG["imgsz"], epochs=CONFIG["epochs"], batch=CONFIG["batch"], multi_scale=CONFIG["multi_scale"],
               workers=2, amp=gpu is not None, plots=online)   # plots download a font: skip them offline
    if CONFIG["time_hours"]:
        hyp["time"] = CONFIG["time_hours"]
    if gpu is None:
        hyp.update(batch=2, workers=0, plots=False, freeze=10)
    out = WORK / "runs"
    while True:                                    # halve the batch on CUDA out-of-memory and start over
        try:
            res = YOLO(str(inp["weights"])).train(data=str(ds_dir / "dataset.yaml"), project=str(out), name=f"livestock_{CONFIG['model']}",
                                                  device=0 if gpu else "cpu", exist_ok=True, **hyp)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if hyp["batch"] <= 1:
                raise
            hyp["batch"] = max(1, hyp["batch"] // 2)
            say("CUDA out of memory: retrying with batch", hyp["batch"])
    wdir = out / f"livestock_{CONFIG['model']}" / "weights"
    best = wdir / "best.pt"
    final = WORK / "output"
    final.mkdir(exist_ok=True)
    for f in [best, wdir / "last.pt", out / f"livestock_{CONFIG['model']}" / "results.csv"]:       # the deliverable goes first, before anything can fail
        if f.exists():
            shutil.copyfile(f, final / f.name)
    summary["train_minutes"] = round((time.time() - t0) / 60, 1)
    summary["batch_used"] = hyp["batch"]
    (WORK / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    say("weights saved to", final, "(post-training steps below are best-effort)")

    try:
        m = YOLO(str(best)).val(data=str(ds_dir / "dataset.yaml"), imgsz=CONFIG["imgsz"], batch=min(hyp["batch"], 8),
                                device=0 if gpu else "cpu", plots=False)
        per = {}
        for i, c in enumerate(list(getattr(m.box, "ap_class_index", []))):
            per[str(m.names[int(c)])] = {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                         "ap50": float(m.box.ap50[i]), "ap": float(m.box.ap[i])}
        summary["metrics"] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp),
                              "recall": float(m.box.mr), "per_class": per}
    except Exception as e:
        summary["metrics"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
        say("validation summary failed (weights are safe):", summary["metrics"]["error"])
    exports = {}
    for fmt in CONFIG["export"]:
        try:
            fmt = "litert" if fmt == "tflite" else fmt          # 'tflite' is a deprecated alias; the output is a .tflite file
            kw = dict(format=fmt, imgsz=CONFIG["imgsz"], nms=False)
            if fmt == "litert":
                kw.update(quantize="int8", data=str(ds_dir / "dataset.yaml"))      # INT8 post-training quantisation, calibrated on the data
            try:
                pth = Path(str(YOLO(str(best)).export(**kw)))
            except Exception as e:
                if fmt != "litert":
                    raise
                say("INT8 export failed, retrying without quantisation:", f"{type(e).__name__}: {str(e)[:150]}")
                kw.pop("quantize"); kw.pop("data")
                pth = Path(str(YOLO(str(best)).export(**kw)))
            exports[fmt] = pth.name
            (shutil.copytree(pth, final / pth.name, dirs_exist_ok=True) if pth.is_dir() else shutil.copyfile(pth, final / pth.name))
        except Exception as e:                      # export is best-effort: the .pt is the real deliverable
            exports[fmt] = f"FAILED: {type(e).__name__}: {str(e)[:200]}"
    summary["exports"] = exports
    (WORK / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    # Everything under /kaggle/working becomes the notebook's downloadable output. Remove the bulky intermediates (two copies
    # of every labelled image, the settings dir, training batches) so `pull` fetches a few MB instead of thousands of files.
    # The dataset is reproducible from the inputs; its statistics are already in run_summary.json.
    for junk in (ds_dir, WORK / ".yolo_config", WORK / "runs"):
        shutil.rmtree(junk, ignore_errors=True)
    say("DONE. metrics:", json.dumps(summary["metrics"])[:300], "| exports:", exports)


if __name__ == "__main__":
    main()

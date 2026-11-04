"""Kaggle script: export a trained YOLO11 model to TFLite (LiteRT) FP32 + INT8 and measure what quantisation costs.

Inputs (attached datasets): livestock-yolo-code, livestock-yolo-data, livestock-yolo-weights (for the COCO server labeller),
livestock-yolo-trained (the fine-tuned best.pt to export).  Needs Internet (pip) and Linux x86: both true on a verified Kaggle notebook.

Why a subprocess + --target: litert-torch pulls its own torch pins. Installing them into an isolated folder and running the export
in a fresh interpreter keeps Kaggle's own environment untouched, and a failure here can never affect the trained weights.
Output: /kaggle/working/output/{*.tflite, export_summary.json} where the summary compares mAP of .pt vs FP32 vs INT8 on the SAME
validation split (the plan's release gate: a quantised model must stay within an agreed mAP gap of its parent).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIG = dict(model="yolo11n", imgsz=640, weak_conf=0.35, limit_per_class=None, dry_run=False)
INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
WORK = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
CONFIG.update(json.loads(os.environ.get("LDX_CONFIG_JSON", "{}")))
ULTRALYTICS = "8.4.191"
SHA256 = {"server-base.pt": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"}
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / ".yolo_config"))


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def find_one(pattern, what):
    hits = sorted(glob.glob(str(INPUT / "**" / pattern), recursive=True))
    if not hits:
        raise SystemExit(f"Missing input: {what} (looked for {pattern} under {INPUT})")
    return Path(hits[0])


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ensure_ultralytics():
    try:
        import ultralytics  # noqa: F401
        return
    except ImportError:
        pass
    say(f"pip install ultralytics=={ULTRALYTICS}")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRALYTICS}"])
    if r.returncode:
        raise SystemExit("could not install ultralytics: enable Internet in the notebook settings (needs a phone-verified account)")


WORKER = r'''
import json, os, shutil, sys
from pathlib import Path
from ultralytics import YOLO
cfg = json.loads(sys.argv[1]); out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)
res = {"exports": {}, "map": {}}
def val(path, tag):
    try:
        m = YOLO(str(path), task="detect").val(data=cfg["data"], imgsz=cfg["imgsz"], batch=1 if str(path).endswith(".tflite") else 16, plots=False, verbose=False)
        res["map"][tag] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    except Exception as e:
        res["map"][tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
val(cfg["weights"], "pt")
for tag, kw in (("fp32", dict()), ("int8", dict(quantize=8, data=cfg["data"])), ("int8_retry_name", dict(quantize="int8", data=cfg["data"])),
                ("w8a32_dynamic", dict(quantize="w8a32"))):
    if tag == "int8_retry_name" and "error" not in str(res["exports"].get("int8", "")):
        continue                                   # only retry the string spelling if the integer one failed
    try:
        p = Path(str(YOLO(cfg["weights"]).export(format="litert", imgsz=cfg["imgsz"], nms=False, **kw)))
        dest = out / f"{cfg['name']}_{tag}.tflite"
        shutil.copyfile(p if p.is_file() else next(p.rglob("*.tflite")), dest)
        res["exports"][tag] = {"file": dest.name, "mb": round(dest.stat().st_size / 1e6, 2)}
        val(dest, tag)
    except Exception as e:
        res["exports"][tag] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
json.dump(res, open(out / "worker_result.json", "w"), indent=2)
print(json.dumps(res, indent=1))
'''


def main():
    t0 = time.time()
    src_dir = find_one("src/livestock_dx/__init__.py", "livestock-yolo-code").parents[1]      # <code>/src
    sys.path.insert(0, str(src_dir))
    ensure_ultralytics()
    trained = find_one("best.pt", "livestock-yolo-trained (a dataset holding the fine-tuned best.pt)")
    labeller = find_one("server-base.pt", "livestock-yolo-weights (COCO server labeller)")
    if sha256(labeller) != SHA256["server-base.pt"]:
        raise SystemExit("server-base.pt checksum mismatch")
    data_dirs = sorted({Path(p).parent.parent for p in glob.glob(str(INPUT / "**" / "*___*" / "*.jpg"), recursive=True)})
    if not data_dirs:
        raise SystemExit("livestock-yolo-data not attached")
    say("trained weights:", trained, "| labeller:", labeller, "| data:", data_dirs[0])

    # 1) rebuild the SAME weak-label dataset (same labeller, same content-hash split) so the comparison is on the training split's val
    from livestock_dx.detect.dataset import build_dataset
    from livestock_dx.detect.pseudo import pseudo_label_animals
    ds = WORK / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    classes, pst = pseudo_label_animals(str(labeller), data_dirs[0], ds / "_src", conf=CONFIG["weak_conf"], imgsz=CONFIG["imgsz"],
                                      limit_per_class=CONFIG["limit_per_class"])
    stats = build_dataset([ds / "_src"], ds, classes, "detect")
    say("dataset:", json.dumps({"labelling": pst, "kept_train": stats.get("kept_train"), "kept_val": stats.get("kept_val")}))

    if CONFIG["dry_run"]:
        say("dry_run: stopping before the export stack install")
        return

    # 2) isolated install of the LiteRT export stack
    target = WORK / "litert_env"
    pkgs = ["litert-torch>=0.10.0", "ai-edge-litert>=2.3.0", "ai-edge-quantizer>=0.10.0", "onnx", "onnxslim", "onnxruntime", f"ultralytics=={ULTRALYTICS}"]
    say("installing export stack into", target, "(isolated)")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--target", str(target), *pkgs], capture_output=True, text=True)
    summary = {"model": CONFIG["model"], "trained_weights": str(trained), "classes": classes, "dataset": {"kept_val": stats.get("kept_val"), "kept_train": stats.get("kept_train")}}
    if r.returncode:
        summary["error"] = "pip install of the export stack failed: " + (r.stderr or "")[-600:]
        say(summary["error"])
    else:
        # 3) export + evaluate in a fresh interpreter that sees the isolated packages first
        env = dict(os.environ, PYTHONPATH=f"{target}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", YOLO_AUTOINSTALL="False")
        # ultralytics writes the exported file NEXT TO the weights, and /kaggle/input is read-only: export from a writable copy
        writable = WORK / "export_src" / "best.pt"
        writable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trained, writable)
        cfg = dict(weights=str(writable), data=str(ds / "dataset.yaml"), imgsz=CONFIG["imgsz"], out=str(WORK / "output"), name=CONFIG["model"])
        w = subprocess.run([sys.executable, "-c", WORKER, json.dumps(cfg)], env=env, capture_output=True, text=True)
        say((w.stdout or "")[-2500:])
        if w.returncode:
            summary["error"] = "export worker failed: " + (w.stderr or "")[-800:]
            say(summary["error"])
        wr = WORK / "output" / "worker_result.json"
        if wr.exists():
            summary.update(json.loads(wr.read_text()))
    (WORK / "output").mkdir(exist_ok=True)
    shutil.copyfile(trained, WORK / "output" / "best.pt")
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    (WORK / "output" / "export_summary.json").write_text(json.dumps(summary, indent=2))
    for junk in (ds, target, WORK / "export_src", WORK / ".yolo_config"):          # keep the downloadable output small
        shutil.rmtree(junk, ignore_errors=True)
    say("DONE", json.dumps({k: summary.get(k) for k in ("exports", "map", "error")})[:1200])


if __name__ == "__main__":
    main()

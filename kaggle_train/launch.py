"""Package, upload and run the YOLO11x fine-tune on Kaggle's GPUs from this machine.

  python kaggle_train/launch.py check                          are the Kaggle CLI + credentials in place?
  python kaggle_train/launch.py prepare [--annotated DIR]      build the upload folders (offline, nothing is sent)
  python kaggle_train/launch.py upload                         create/update the PRIVATE datasets on your account
  python kaggle_train/launch.py run [--epochs N --imgsz N ...] push + start the GPU notebook
  python kaggle_train/launch.py status                         is it running / finished?
  python kaggle_train/launch.py pull                           download weights + run_summary.json into runs/kaggle/
  python kaggle_train/launch.py all [--annotated DIR]          prepare + upload + run

Nothing is uploaded until you run `upload`/`run`/`all`. Datasets are created PRIVATE. Credentials are read from
the standard Kaggle locations only (see `check`); this script never asks for or stores a token.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "kaggle_train" / "_upload"
KERNEL_DIR = ROOT / "kaggle_train" / "_kernel"
OUT = ROOT / "runs" / "kaggle"
YOLO11X_SHA256 = "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
YOLO11N_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
SEP = "___"
SLUGS = {"weights": "livestock-yolo-weights", "code": "livestock-yolo-code", "data": "livestock-yolo-data",
         "wheels": "livestock-yolo-wheels", "trained": "livestock-yolo-trained", "annotated": "livestock-yolo-annotated"}
WHEELS = ROOT / "kaggle_train" / "_wheels"
ULTRALYTICS = "8.4.191"         # keep equal to ULTRALYTICS in train_yolo11x.py
KERNEL_TITLE = "LivestockDx YOLO11x fine-tune"
KERNEL_SLUG = "livestockdx-yolo11x-fine-tune"      # Kaggle derives the slug from the title: they must match
KERNELS = {"yolo11x": {"title": KERNEL_TITLE, "slug": KERNEL_SLUG, "hours": 8.0},
           "yolo11n": {"title": "LivestockDx YOLO11n fine-tune", "slug": "livestockdx-yolo11n-fine-tune", "hours": 3.0},
           # export-only job: INT8/FP32 TFLite of an already fine-tuned model + the mAP cost of quantisation
           "yolo11n-export": {"title": "LivestockDx YOLO11n export", "slug": "livestockdx-yolo11n-export", "hours": 1.0,
                              "script": "export_litert.py", "cfg_model": "yolo11n", "sources": ("weights", "code", "data", "trained")}}


# ---------------------------------------------------------------- credentials / CLI
def credential_sources() -> list[str]:
    home = Path.home() / ".kaggle"
    found = [str(p) for p in (home / "kaggle.json", home / "access_token") if p.exists()]
    found += [k for k in ("KAGGLE_API_TOKEN",) if os.environ.get(k)]
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        found.append("KAGGLE_USERNAME + KAGGLE_KEY")
    return found


def kaggle_cmd() -> list[str] | None:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    try:
        import kaggle  # noqa: F401
        return [sys.executable, "-m", "kaggle.cli"]
    except Exception:
        return None


def username() -> str:
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        return json.loads(kj.read_text())["username"]
    raise SystemExit("Cannot determine your Kaggle username. Set KAGGLE_USERNAME or use --user.")


def need_ready(user: str | None = None) -> tuple[list[str], str]:
    cmd = kaggle_cmd()
    problems = []
    if cmd is None:
        problems.append("The Kaggle CLI is not installed:   pip install kaggle")
    if not credential_sources():
        problems.append(
            "No Kaggle credentials found. One-time setup:\n"
            "    1. kaggle.com -> your profile -> Settings -> API -> 'Create New Token'\n"
            "    2. save the downloaded kaggle.json as  %USERPROFILE%\\.kaggle\\kaggle.json  (Windows)  or  ~/.kaggle/kaggle.json\n"
            "       (a new-style API token can instead go in ~/.kaggle/access_token or the KAGGLE_API_TOKEN variable)\n"
            "    3. keep that file private: it is a password for your account")
    if problems:
        raise SystemExit("Kaggle is not set up yet:\n  - " + "\n  - ".join(problems))
    return cmd, user or username()


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, **kw)


# ---------------------------------------------------------------- bundles
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def write_meta(folder: Path, user: str, slug: str, title: str, licence: str) -> None:
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}", "licenses": [{"name": licence}]}, indent=2))


def fetch_wheels(dest: Path) -> None:
    """Linux wheels for Kaggle's Python 3.13, so the notebook needs no internet (new accounts/unverified phones get none)."""
    dest.mkdir(parents=True, exist_ok=True)
    plat = [x for p in ("manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64", "manylinux_2_34_x86_64", "any")
            for x in ("--platform", p)]
    cmd = [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "--python-version", "3.13",
           "--implementation", "cp", "--abi", "cp313", *plat, "-d", str(dest),
           f"ultralytics=={ULTRALYTICS}", "ultralytics-thop", "polars==2.1.0", "polars-runtime-32==2.1.0", "opencv-python-headless"]
    if run(cmd).returncode:
        raise SystemExit("could not download the Linux wheels for Kaggle (see output above)")


def prepare(user: str, annotated: Path | None = None, trained: Path | None = None) -> dict:
    """Build the upload folders. Offline: nothing is sent anywhere."""
    shutil.rmtree(UP, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")

    w = UP / "weights"
    w.mkdir(parents=True)
    src = ROOT / "pretrained" / "yolo11x.pt"
    if not src.exists():
        raise SystemExit(f"{src} not found")
    if sha256(src) != YOLO11X_SHA256:
        raise SystemExit("pretrained/yolo11x.pt does not match the pinned SHA-256: refusing to package it")
    shutil.copyfile(src, w / "yolo11x.pt")
    n_src = ROOT / "pretrained" / "yolo11n.pt"
    if n_src.exists():
        if sha256(n_src) != YOLO11N_SHA256:
            raise SystemExit("pretrained/yolo11n.pt does not match the pinned SHA-256: refusing to package it")
        shutil.copyfile(n_src, w / "yolo11n.pt")
    write_meta(w, user, SLUGS["weights"], "LivestockDx YOLO11x weights", "AGPL-3.0")   # Ultralytics' licence

    c = UP / "code"
    shutil.copytree(ROOT / "src" / "livestock_dx" / "detect", c / "src" / "livestock_dx" / "detect", ignore=ignore)
    shutil.copytree(ROOT / "src" / "livestock_dx" / "clean", c / "src" / "livestock_dx" / "clean",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "harmonize.py"))   # harmonize needs the DB stack
    (c / "src" / "livestock_dx" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    (c / "configs").mkdir()
    shutil.copyfile(ROOT / "configs" / "yolo_train.yaml", c / "configs" / "yolo_train.yaml")
    write_meta(c, user, SLUGS["code"], "LivestockDx YOLO code", "other")

    wh = UP / "wheels"
    if not list(WHEELS.glob("ultralytics-*.whl")):
        fetch_wheels(WHEELS)
    shutil.copytree(WHEELS, wh)
    write_meta(wh, user, SLUGS["wheels"], "LivestockDx YOLO wheels", "AGPL-3.0")

    d = UP / "data"
    d.mkdir()
    n = 0
    for cls in sorted((ROOT / "vetdata").glob(f"*{SEP}*")):
        if cls.is_dir():
            shutil.copytree(cls, d / cls.name)
            n += sum(1 for _ in (d / cls.name).iterdir())
    if n == 0:
        raise SystemExit("vetdata/ has no class folders; run the dataset download first")
    write_meta(d, user, SLUGS["data"], "LivestockDx livestock images", "CC-BY-4.0")   # sources: Zenodo CC-BY-4.0 sets

    info = {"images": n, "weights_mb": round((w / "yolo11x.pt").stat().st_size / 1e6)}
    if trained:                                   # a fine-tuned best.pt to export (kept under its version folder name)
        t = UP / "trained" / Path(trained).parent.name
        t.mkdir(parents=True)
        shutil.copyfile(trained, t / "best.pt")
        write_meta(UP / "trained", user, SLUGS["trained"], "LivestockDx trained weights", "AGPL-3.0")
        info["trained"] = str(Path(trained).parent.name)
    if annotated:
        a = UP / "annotated"
        shutil.copytree(annotated, a)
        if not ((a / "images").is_dir() and (a / "labels").is_dir() and ((a / "classes.txt").exists() or (a / "dataset.yaml").exists())):
            raise SystemExit("--annotated must contain images/, labels/ and classes.txt (or dataset.yaml with names)")
        write_meta(a, user, SLUGS["annotated"], "LivestockDx annotated lesions", "other")
        info["annotated_images"] = sum(1 for _ in (a / "images").rglob("*") if _.is_file())
    print("prepared:", info, "->", UP)
    return info


# ---------------------------------------------------------------- Kaggle operations
def upload(user: str, only: list[str] | None = None) -> None:
    cmd, user = need_ready(user)
    for key in ("weights", "code", "wheels", "data", "trained", "annotated"):
        folder = UP / key
        if not folder.exists() or (only and key not in only):
            continue
        slug = f"{user}/{SLUGS[key]}"
        exists = run(cmd + ["datasets", "status", slug], capture_output=True).returncode == 0
        args = (["datasets", "version", "-p", str(folder), "-m", "update", "--dir-mode", "zip"] if exists
                else ["datasets", "create", "-p", str(folder), "--dir-mode", "zip"])        # private unless --public
        r = run(cmd + args)
        if r.returncode:
            raise SystemExit(f"Kaggle upload of {slug} failed (see output above)")


def wait_datasets_ready(cmd: list[str], slugs: list[str], timeout_s: int = 1800) -> None:
    """A freshly created dataset takes a while to process; a notebook pushed before that silently runs WITHOUT it."""
    t0 = time.time()
    pending = list(slugs)
    while pending:
        pending = [s for s in pending
                   if "ready" not in (run(cmd + ["datasets", "status", s], capture_output=True).stdout or "").lower()]
        if not pending:
            return
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"Datasets still not ready after {timeout_s // 60} min: {pending}. Check them on kaggle.com, then run `run` again.")
        print(f"waiting for Kaggle to finish processing: {pending}", flush=True)
        time.sleep(15)


def push_kernel(user: str, overrides: dict, accelerator: str = "NvidiaTeslaT4", model: str = "yolo11x") -> None:
    cmd, user = need_ready(user)
    k = KERNELS[model]
    overrides = {**overrides, "model": k.get("cfg_model", model)}
    kdir = KERNEL_DIR if model == "yolo11x" else KERNEL_DIR.with_name(KERNEL_DIR.name + "_" + model)
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    script_name = k.get("script", "train_yolo11x.py")
    script = (ROOT / "kaggle_train" / script_name).read_text(encoding="utf-8")
    if overrides:                                       # bake CLI overrides into the pushed copy of the script
        script = script.replace('CONFIG.update(json.loads(os.environ.get("LDX_CONFIG_JSON", "{}")))',
                                f'CONFIG.update({overrides!r})\nCONFIG.update(json.loads(os.environ.get("LDX_CONFIG_JSON", "{{}}")))')
    (kdir / script_name).write_text(script, encoding="utf-8")
    sources = [f"{user}/{SLUGS[x]}" for x in k.get("sources", ("weights", "code", "wheels", "data"))] + ([f"{user}/{SLUGS['annotated']}"] if (UP / "annotated").exists() else [])
    (kdir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{k['slug']}", "title": k["title"], "code_file": script_name,
        "language": "python", "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_internet": "true",
        "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}, indent=2))
    wait_datasets_ready(cmd, sources)
    hours = overrides.get("time_hours") or k["hours"]
    push = ["kernels", "push", "-p", str(kdir), "--accelerator", accelerator,
            "-t", str(int(hours * 3600 + 5400))]               # training budget + 1.5 h for validation/export; Kaggle caps it globally
    r = run(cmd + push, capture_output=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    if r.returncode:
        raise SystemExit("kernels push failed (see output above)")
    if "could not be added" in out or "not valid dataset sources" in out:
        raise SystemExit("Kaggle did not attach every dataset to the notebook (see above). The run would be missing its inputs: "
                         "do not rely on it; fix and push again.")
    print(f"Started {model}. Watch it at https://www.kaggle.com/code/{user}/{k['slug']}  or run:  python kaggle_train/launch.py status --model {model}")


def status(user: str, model: str = "yolo11x") -> None:
    cmd, user = need_ready(user)
    run(cmd + ["kernels", "status", f"{user}/{KERNELS[model]['slug']}"])


def pull(user: str, model: str = "yolo11x") -> None:
    cmd, user = need_ready(user)
    out = OUT / model
    out.mkdir(parents=True, exist_ok=True)
    run(cmd + ["kernels", "output", f"{user}/{KERNELS[model]['slug']}", "-p", str(out), "--page-size", "200",
                "--file-pattern", r"(best\.pt|last\.pt|results\.csv|run_summary\.json|\.onnx|\.tflite|\.log)$"])   # skip the bulky dataset copies
    s = out / "run_summary.json"
    if s.exists():
        d = json.loads(s.read_text())
        print("\nmode:", d.get("mode"), "| gpu:", d.get("gpu"), "| minutes:", d.get("train_minutes"))
        if d.get("warning"):
            print("WARNING:", d["warning"])
        print("metrics:", json.dumps(d.get("metrics", {}), indent=1)[:1500])
    print("files in", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "prepare", "upload", "run", "status", "pull", "all"])
    ap.add_argument("--user")
    ap.add_argument("--annotated", type=Path)
    ap.add_argument("--trained", type=Path, help="fine-tuned best.pt to package for the export job (e.g. models/yolo11n-weak-v1/best.pt)")
    ap.add_argument("--only", nargs="+", choices=list(SLUGS), help="upload only these datasets (e.g. --only wheels)")
    for k in ("epochs", "imgsz", "batch"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--time-hours", type=float)
    ap.add_argument("--task", choices=["detect", "segment"])
    ap.add_argument("--model", choices=sorted(KERNELS) + ["both"], default="yolo11x",
                    help="which model to train/watch/pull (default yolo11x; 'both' pushes yolo11x then yolo11n)")
    ap.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle accelerator id (default: GPU T4 x2; one GPU is used)")
    a = ap.parse_args(argv)
    over = {k: v for k, v in dict(epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, time_hours=a.time_hours, task=a.task).items() if v is not None}

    if a.cmd == "check":
        print("Kaggle CLI:", kaggle_cmd() or "NOT INSTALLED (pip install kaggle)")
        print("credentials:", credential_sources() or "NONE FOUND")
        try:
            need_ready(a.user)
            print("ready: yes")
        except SystemExit as e:
            print(e)
        return
    user = a.user or (username() if credential_sources() else "YOUR_KAGGLE_USERNAME")
    if a.cmd in ("prepare", "all"):
        prepare(user, a.annotated, a.trained)
    if a.cmd in ("upload", "all"):
        upload(user, a.only)
    models = ["yolo11x", "yolo11n"] if a.model == "both" else [a.model]              # x first, then n
    if a.cmd in ("run", "all"):
        for m in models:
            push_kernel(user, over, a.accelerator, m)
    if a.cmd == "status":
        for m in models:
            status(user, m)
    if a.cmd == "pull":
        for m in models:
            pull(user, m)


if __name__ == "__main__":
    main()

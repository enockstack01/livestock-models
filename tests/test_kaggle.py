"""Kaggle launcher: bundle layout, safety checks, and the exact CLI commands it would run (recorded, never executed)."""
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kaggle_train"))
import launch  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    for d in ("src/livestock_dx/detect", "src/livestock_dx/clean", "configs", "pretrained", "vetdata/Cattle___Healthy", "vetdata/_symptoms"):
        (r / d).mkdir(parents=True)
    (r / "src/livestock_dx/detect/dataset.py").write_text("x = 1\n")
    (r / "src/livestock_dx/clean/images.py").write_text("x = 1\n")
    (r / "src/livestock_dx/clean/harmonize.py").write_text("import sqlalchemy\n")
    (r / "configs/yolo_train.yaml").write_text("epochs: 1\n")
    (r / "pretrained/server-base.pt").write_bytes(b"fake weights")
    (r / "vetdata/Cattle___Healthy/a.jpg").write_bytes(b"img")
    (r / "vetdata/_symptoms/pmc.jsonl").write_text("{}\n")
    monkeypatch.setattr(launch, "ROOT", r)
    monkeypatch.setattr(launch, "UP", r / "up")
    monkeypatch.setattr(launch, "KERNEL_DIR", r / "kernel")
    monkeypatch.setattr(launch, "SERVER_BASE_SHA256", hashlib.sha256(b"fake weights").hexdigest())
    monkeypatch.setattr(launch, "WHEELS", r / "wheels_cache")
    monkeypatch.setattr(launch, "fetch_wheels", lambda d: (d.mkdir(parents=True, exist_ok=True), (d / "ultralytics-0.whl").write_bytes(b"w")))
    return r


def test_prepare_layout_and_metadata(fake_root):
    info = launch.prepare("alice")
    up = fake_root / "up"
    assert info["images"] == 1
    assert (up / "weights/server-base.pt").exists() and (up / "code/configs/yolo_train.yaml").exists()
    assert (up / "data/Cattle___Healthy/a.jpg").exists()
    assert not (up / "data/_symptoms").exists()                       # text notes are not uploaded
    assert not (up / "code/src/livestock_dx/clean/harmonize.py").exists()   # DB code is not shipped
    ids = {k: json.loads((up / k / "dataset-metadata.json").read_text())["id"] for k in ("weights", "code", "wheels", "data")}
    assert ids == {"weights": "alice/livestock-yolo-weights", "code": "alice/livestock-yolo-code",
                   "wheels": "alice/livestock-yolo-wheels", "data": "alice/livestock-yolo-data"}
    assert (up / "wheels" / "ultralytics-0.whl").exists()
    assert not (up / "annotated").exists()


def test_prepare_refuses_tampered_weights_and_empty_data(fake_root):
    (fake_root / "pretrained/server-base.pt").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="SHA-256"):
        launch.prepare("alice")
    (fake_root / "pretrained/server-base.pt").write_bytes(b"fake weights")
    import shutil
    shutil.rmtree(fake_root / "vetdata/Cattle___Healthy")
    with pytest.raises(SystemExit, match="no class folders"):
        launch.prepare("alice")


def test_annotated_dataset_is_validated_and_attached(fake_root, tmp_path):
    bad = tmp_path / "bad"
    (bad / "images").mkdir(parents=True)
    with pytest.raises(SystemExit, match="must contain"):
        launch.prepare("alice", bad)
    good = tmp_path / "good"
    (good / "images").mkdir(parents=True)
    (good / "labels").mkdir()
    (good / "classes.txt").write_text("lumpy_skin_nodules\n")
    (good / "images/a.jpg").write_bytes(b"i")
    assert launch.prepare("alice", good)["annotated_images"] == 1
    assert (fake_root / "up/annotated/dataset-metadata.json").exists()


def test_not_ready_without_credentials_or_cli(monkeypatch):
    monkeypatch.setattr(launch, "kaggle_cmd", lambda: None)
    monkeypatch.setattr(launch, "credential_sources", lambda: [])
    with pytest.raises(SystemExit) as e:
        launch.need_ready()
    msg = str(e.value)
    assert "pip install kaggle" in msg and "kaggle.json" in msg and "Create New Token" in msg


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    state = {"exists": False}

    def fake_run(cmd, **kw):
        calls.append(cmd)
        rc = 0 if (cmd[1:3] != ["datasets", "status"] or state["exists"]) else 1
        out = "ready" if cmd[1:3] == ["datasets", "status"] else state.get("push_output", "")   # rc says "exists?", stdout says "processed?"
        return subprocess.CompletedProcess(cmd, rc, out, "")

    monkeypatch.setattr(launch, "kaggle_cmd", lambda: ["kaggle"])
    monkeypatch.setattr(launch, "credential_sources", lambda: ["test"])
    monkeypatch.setattr(launch, "run", fake_run)
    monkeypatch.setattr(launch.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("unexpected wait")))
    return calls, state


def test_upload_creates_private_datasets_then_versions_them(fake_root, recorded):
    calls, state = recorded
    launch.prepare("alice")
    launch.upload("alice")
    creates = [c for c in calls if c[1:3] == ["datasets", "create"]]
    assert len(creates) == 4 and all("--public" not in c for c in creates)       # private by default
    assert not [c for c in calls if c[1:3] == ["datasets", "version"]]
    calls.clear()
    state["exists"] = True
    launch.upload("alice")
    assert len([c for c in calls if c[1:3] == ["datasets", "version"]]) == 4 and not [c for c in calls if c[1:3] == ["datasets", "create"]]


def test_kernel_push_metadata_and_overrides(fake_root, recorded):
    calls, _ = recorded
    (fake_root / "kaggle_train").mkdir()
    real = Path(__file__).resolve().parents[1] / "kaggle_train" / "train_server.py"
    (fake_root / "kaggle_train/train_server.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    launch.push_kernel("alice", {"epochs": 7, "imgsz": 800})
    meta = json.loads((fake_root / "kernel/kernel-metadata.json").read_text())
    assert meta["enable_gpu"] == "true" and meta["is_private"] == "true" and meta["kernel_type"] == "script"
    assert meta["id"] == f"alice/{launch.KERNEL_SLUG}" and meta["title"] == launch.KERNEL_TITLE
    assert meta["dataset_sources"] == ["alice/livestock-yolo-weights", "alice/livestock-yolo-code", "alice/livestock-yolo-wheels", "alice/livestock-yolo-data"]
    script = (fake_root / "kernel/train_server.py").read_text(encoding="utf-8")
    ast.parse(script)                                                            # still valid Python after baking overrides
    assert "'epochs': 7" in script and "'imgsz': 800" in script and "'model': 'server'" in script
    assert calls[-1][1:3] == ["kernels", "push"]
    assert calls[-1][calls[-1].index("--accelerator") + 1] == "NvidiaTeslaT4"
    assert int(calls[-1][calls[-1].index("-t") + 1]) == int(8.0 * 3600 + 5400)


def test_push_fails_loudly_if_kaggle_drops_a_dataset(fake_root, recorded):
    calls, state = recorded
    state["push_output"] = "The following are not valid dataset sources and could not be added to the kernel: ['alice/x']"
    (fake_root / "kaggle_train").mkdir()
    real = Path(__file__).resolve().parents[1] / "kaggle_train" / "train_server.py"
    (fake_root / "kaggle_train/train_server.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(SystemExit, match="did not attach every dataset"):
        launch.push_kernel("alice", {})


def test_slug_matches_title_for_every_model():
    import re
    for m, k in launch.KERNELS.items():
        assert re.sub(r"[^a-z0-9]+", "-", k["title"].lower()).strip("-") == k["slug"], m


def test_yolo11n_kernel_is_separate_and_gets_its_own_budget(fake_root, recorded):
    calls, _ = recorded
    (fake_root / "kaggle_train").mkdir()
    real = Path(__file__).resolve().parents[1] / "kaggle_train" / "train_server.py"
    (fake_root / "kaggle_train/train_server.py").write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    launch.push_kernel("alice", {}, model="server")
    launch.push_kernel("alice", {}, model="yolo11n")
    metas = {m: json.loads(((fake_root / ("kernel" if m == "server" else "kernel_yolo11n")) / "kernel-metadata.json").read_text())
             for m in ("server", "yolo11n")}
    assert metas["server"]["id"] != metas["yolo11n"]["id"] and metas["yolo11n"]["id"] == "alice/livestockdx-yolo11n-fine-tune"
    assert metas["yolo11n"]["dataset_sources"] == metas["server"]["dataset_sources"]      # same weights/data/code datasets
    pushes = [c for c in calls if c[1:3] == ["kernels", "push"]]
    assert int(pushes[0][pushes[0].index("-t") + 1]) == int(8.0 * 3600 + 5400)
    assert int(pushes[1][pushes[1].index("-t") + 1]) == int(3.0 * 3600 + 5400)              # the nano model gets a short budget
    assert "'model': 'yolo11n'" in (fake_root / "kernel_yolo11n/train_server.py").read_text(encoding="utf-8")


def test_notebook_labels_with_server_even_when_training_yolo11n():
    """Both models must train on the SAME pseudo-labelled data, so the labeller is always the COCO server."""
    src = (Path(__file__).resolve().parents[1] / "kaggle_train" / "train_server.py").read_text(encoding="utf-8")
    assert 'str(inp["label_weights"])' in src and 'find_one("server-base.pt"' in src
    assert 'PER_MODEL = {"server"' in src and '"yolo11n": {"batch": 32' in src
    assert "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1" in src       # yolo11n checksum pinned


def test_code_bundle_imports_without_side_effects(tmp_path):
    """The Kaggle inputs are read-only: importing the shipped modules must not load the web package (it creates folders)."""
    code = ("import sys; sys.path.insert(0, %r); import livestock_dx.detect.dataset, livestock_dx.detect.pseudo;"
            "print(any(m.startswith('livestock_dx.web') for m in sys.modules))" % str(SRC))
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
    assert not any(tmp_path.iterdir())


def test_export_job_packages_trained_weights_and_uses_its_own_script(fake_root, recorded, tmp_path):
    calls, _ = recorded
    (fake_root / "kaggle_train").mkdir()
    for name in ("train_server.py", "export_litert.py"):
        real = Path(__file__).resolve().parents[1] / "kaggle_train" / name
        (fake_root / "kaggle_train" / name).write_text(real.read_text(encoding="utf-8"), encoding="utf-8")
    w = tmp_path / "yolo11n-weak-v1"
    w.mkdir()
    (w / "best.pt").write_bytes(b"trained")
    info = launch.prepare("alice", trained=w / "best.pt")
    assert info["trained"] == "yolo11n-weak-v1"
    assert (fake_root / "up/trained/yolo11n-weak-v1/best.pt").read_bytes() == b"trained"
    assert json.loads((fake_root / "up/trained/dataset-metadata.json").read_text())["id"] == "alice/livestock-yolo-trained"
    launch.upload("alice", only=["trained"])
    creates = [c for c in calls if c[1:3] == ["datasets", "create"]]
    assert len(creates) == 1 and creates[0][creates[0].index("-p") + 1].endswith("trained")        # only that dataset was sent
    launch.push_kernel("alice", {}, model="yolo11n-export")
    kdir = fake_root / "kernel_yolo11n-export"
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    assert meta["code_file"] == "export_litert.py" and (kdir / "export_litert.py").exists()
    assert meta["id"] == "alice/livestockdx-yolo11n-export"
    assert meta["dataset_sources"] == ["alice/livestock-yolo-weights", "alice/livestock-yolo-code", "alice/livestock-yolo-data", "alice/livestock-yolo-trained"]
    baked = (kdir / "export_litert.py").read_text(encoding="utf-8")
    assert "'model': 'yolo11n'" in baked                                                           # the export job targets the nano model
    ast.parse(baked)

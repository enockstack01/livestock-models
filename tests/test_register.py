import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from livestock_vision.auth import make_token
from livestock_vision.config import Settings
from livestock_vision.main import create_app
from livestock_vision.register import register

SUMMARY = {
    "model": "server", "mode": "weak", "gpu": "Tesla T4", "train_minutes": 95.0,
    "warning": "WEAK LABELS: one whole-animal box per photo ...",
    "classes": ["foot_and_mouth_disease", "healthy", "lumpy_skin_disease"],
    "dataset": {"instances_per_class": {"healthy": 600, "lumpy_skin_disease": 200}},
    "config": {"epochs": 120, "imgsz": 640},
    "metrics": {"mAP50": 0.91, "mAP50-95": 0.78, "precision": 0.9, "recall": 0.85, "per_class": {}}}


def make_run(tmp_path, summary=None):
    d = tmp_path / "run"
    d.mkdir()
    (d / "best.pt").write_bytes(b"fake weights")
    (d / "run_summary.json").write_text(json.dumps(summary or SUMMARY))
    return d


def test_register_writes_card_labels_and_manifest(tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"server": {"version": None}, "yolo11n": {"latest": None}}))
    r = register(make_run(tmp_path), "livestock-v1", tmp_path / "models", man)
    out = Path(r["dir"])
    labels = json.loads((out / "labels.json").read_text())
    status = {c["name"]: c["status"] for c in labels["classes"]}
    assert status == {"foot_and_mouth_disease": "untrained", "healthy": "active", "lumpy_skin_disease": "active"}
    assert [c["id"] for c in labels["classes"]] == [0, 1, 2] and labels["scope"].startswith("animal-level")
    flags = {c["name"]: c["notifiable_suspect"] for c in labels["classes"]}
    assert flags == {"foot_and_mouth_disease": True, "healthy": False, "lumpy_skin_disease": True}
    card = json.loads((out / "model_card.json").read_text())
    assert card["metrics"]["mAP50"] == 0.91 and card["thresholds"] == {} and "not tuned" in card["thresholds_note"].lower()
    assert any("foot_and_mouth_disease" in w and "cannot detect" in w for w in card["warnings"])      # untrained class is called out
    assert len(card["weights_sha256"]) == 64
    assert json.loads(man.read_text())["server"]["version"] == "livestock-v1"
    assert set(r["env"]) == {"VISION_WEIGHTS", "VISION_MODEL_VERSION", "VISION_MODEL_CARD", "LABELS_PATH"}


def test_register_registers_yolo11n_under_its_own_manifest_key(tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"server": {"version": None}, "yolo11n": {"latest": None}}))
    register(make_run(tmp_path, {**SUMMARY, "model": "yolo11n"}), "yolo11n-weak-v1", tmp_path / "models", man)
    m = json.loads(man.read_text())
    assert m["yolo11n"]["latest"] == "yolo11n-weak-v1" and m["server"]["version"] is None


def test_register_refuses_bad_inputs(tmp_path):
    man = tmp_path / "m.json"
    with pytest.raises(SystemExit, match="best.pt"):
        register(tmp_path, "v", tmp_path / "models", man)                              # nothing to register
    bad = make_run(tmp_path, {**SUMMARY, "metrics": {"error": "ValueError: boom"}})
    with pytest.raises(SystemExit, match="no validation metrics"):
        register(bad, "v", tmp_path / "models", man)
    good = tmp_path / "good"
    good.mkdir()
    (good / "best.pt").write_bytes(b"w")
    (good / "run_summary.json").write_text(json.dumps(SUMMARY))
    register(good, "v1", tmp_path / "models", man)
    with pytest.raises(SystemExit, match="immutable"):
        register(good, "v1", tmp_path / "models", man)                                 # a released version is never overwritten


class Fake:
    def __init__(self, names):
        self.names = dict(enumerate(names))

    def predict(self, im, conf):
        return [{"class": "lumpy_skin_disease", "confidence": 0.9, "box": [10.0, 10.0, 90.0, 90.0]},
                {"class": "healthy", "confidence": 0.5, "box": [5.0, 5.0, 100.0, 100.0]}]


def test_api_serves_a_registered_model_and_flags_notifiable(tmp_path):
    r = register(make_run(tmp_path), "livestock-v1", tmp_path / "models", tmp_path / "m.json")
    env = r["env"]
    s = Settings(data_dir=tmp_path / "d", jwt_secret="s", labels_path=Path(env["LABELS_PATH"]), model_card_path=env["VISION_MODEL_CARD"],
                 model_version=env["VISION_MODEL_VERSION"])
    with TestClient(create_app(s, Fake(SUMMARY["classes"]))) as c:
        assert c.get("/ready").status_code == 200
        tok = make_token({"sub": "u", "exp": time.time() + 99}, "s")
        img = io.BytesIO()
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)).save(img, "JPEG")
        j = c.post("/v1/detect", files={"image": ("a.jpg", img.getvalue(), "image/jpeg")}, headers={"Authorization": f"Bearer {tok}"}).json()
        assert j["flags"] == ["possible_notifiable_disease"] and j["model_version"] == "livestock-v1"
        info = c.get("/v1/model", headers={"Authorization": f"Bearer {tok}"}).json()
        assert info["metrics"]["mAP50"] == 0.91
        assert {x["name"]: x["status"] for x in info["classes"]}["foot_and_mouth_disease"] == "untrained"
    # the standard livestock model's class list is NOT accepted for this model, and vice versa
    other = Settings(data_dir=tmp_path / "d2", jwt_secret="s", model_version="x")
    with TestClient(create_app(other, Fake(SUMMARY["classes"]))) as c:
        assert c.get("/ready").status_code == 503

"""The Space's plain-JSON "detect" endpoint (space/app.py), without loading the real weights."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

SPACE = Path(__file__).resolve().parents[1] / "space" / "app.py"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    # Stand-ins for the YOLO weights and the ZeroGPU package, so importing app.py loads nothing heavy.
    (tmp_path / "best.pt").write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda weights: object()
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.delitem(sys.modules, "spaces", raising=False)
    spec = importlib.util.spec_from_file_location("space_app", SPACE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_detect_returns_plain_json(app, monkeypatch):
    seen = {}

    def fake_infer(image, conf):
        seen["size"], seen["conf"] = image.size, conf
        return [((1, 2, 60, 40), "lumpy_skin_disease", 0.87654), ((0, 0, 64, 48), "healthy", 0.3)]

    monkeypatch.setattr(app, "_infer", fake_infer)
    out = app.detect_json(Image.new("RGB", (64, 48)), 0.3)
    assert out == {
        "model_version": app.MODEL_VERSION,
        "image": {"width": 64, "height": 48},
        "detections": [
            {"label": "lumpy_skin_disease", "confidence": 0.8765, "box": {"x1": 1, "y1": 2, "x2": 60, "y2": 40}, "reportable": True},
            {"label": "healthy", "confidence": 0.3, "box": {"x1": 0, "y1": 0, "x2": 64, "y2": 48}, "reportable": False},
        ],
    }
    assert seen == {"size": (64, 48), "conf": 0.3}


def test_detect_defaults_and_clamps_confidence(app, monkeypatch):
    seen = []
    monkeypatch.setattr(app, "_infer", lambda image, conf: seen.append(conf) or [])
    assert app.detect_json(Image.new("RGB", (8, 8)), None)["detections"] == []
    app.detect_json(Image.new("RGB", (8, 8)), 5)
    assert seen == [0.25, 0.9]


def test_detect_without_image_is_an_error(app):
    import gradio as gr
    with pytest.raises(gr.Error):
        app.detect_json(None, 0.25)


def test_only_one_detect_endpoint(app):
    names = [d.api_name for d in app.demo.fns.values() if getattr(d, "api_name", None)]
    assert names.count("detect") == 1, names

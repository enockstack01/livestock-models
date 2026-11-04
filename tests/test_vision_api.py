import asyncio
import io
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from livestock_dx.detect.dataset import build_dataset
from livestock_vision.auth import make_token, verify_token, AuthError
from livestock_vision.config import ROOT, Settings
from livestock_vision.engine import Labels
from livestock_vision.export import export_corrections
from livestock_vision.main import create_app

SECRET = "test-secret-123"
LABELS = Labels(ROOT / "shared" / "labels.json")
NAMES = {c["id"]: c["name"] for c in LABELS.classes}


class FakeDetector:
    def __init__(self, script=None, names=None, delay=0.0):
        self.names = names or NAMES
        self.script = script if script is not None else [
            {"class": "skin_nodule", "confidence": 0.88, "box": [40.0, 30.0, 120.0, 100.0]},
            {"class": "tick", "confidence": 0.74, "box": [200.0, 150.0, 230.0, 175.0]}]
        self.delay, self.active, self.max_active, self.lock = delay, 0, 0, threading.Lock()
        self.threads = set()

    def predict(self, im, conf):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return [dict(d) for d in self.script if d["confidence"] >= conf]


def token(sub="farmer-1", role="farmer", exp_in=3600, secret=SECRET):
    return make_token({"sub": sub, "role": role, "exp": time.time() + exp_in}, secret)


def auth(**kw):
    return {"Authorization": f"Bearer {token(**kw)}"}


def photo(fmt="JPEG", size=(320, 240), seed=1, exif=False):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    b = io.BytesIO()
    kw = {}
    if exif and fmt == "JPEG":
        ex = Image.Exif()
        ex[0x010F] = "SecretPhoneMaker"
        kw["exif"] = ex
    im.save(b, fmt, **kw)
    return b.getvalue()


def make_settings(tmp_path, **kw):
    base = dict(data_dir=tmp_path / "data", jwt_secret=SECRET, public_base_url="http://test", model_version="yolo11x-test-v1")
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def det():
    return FakeDetector()


@pytest.fixture
def client(tmp_path, det):
    with TestClient(create_app(make_settings(tmp_path), det)) as c:
        c.settings, c.det = c.app.state.settings, det
        yield c


def post(c, headers=None, img=None, **data):
    img = img if img is not None else photo()
    return c.post("/v1/detect", files={"image": ("a.jpg", img, "image/jpeg")}, data=data, headers=headers or auth())


# ------------------------------------------------------------------ shared class files
def test_labels_and_livestock_yaml_agree():
    y = yaml.safe_load((ROOT / "shared" / "livestock.yaml").read_text())
    assert y["names"] == NAMES
    assert len(NAMES) == len(set(NAMES.values())) == len(LABELS.classes)
    assert all(c["display"]["en"] and "fr" in c["display"] and "rw" in c["display"] for c in LABELS.classes)
    assert [c["id"] for c in LABELS.classes] == list(range(len(LABELS.classes)))


# ------------------------------------------------------------------ auth
def test_auth_rejects_bad_tokens(client):
    assert client.get("/v1/model").status_code == 401
    assert client.get("/v1/model", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert client.get("/v1/model", headers=auth(secret="other-secret")).status_code == 401
    assert client.get("/v1/model", headers=auth(exp_in=-10)).status_code == 401
    import base64
    none_alg = (base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode() + "." +
                base64.urlsafe_b64encode(json.dumps({"sub": "x", "exp": time.time() + 99}).encode()).rstrip(b"=").decode() + ".")
    assert client.get("/v1/model", headers={"Authorization": f"Bearer {none_alg}"}).status_code == 401   # alg=none attack
    ok = client.get("/v1/model", headers=auth())
    assert ok.status_code == 200 and ok.json()["model_version"] == "yolo11x-test-v1"
    assert client.get("/health").json() == {"status": "ok"} and client.get("/ready").status_code == 200   # probes need no token


def test_auth_fails_closed_without_secret(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, jwt_secret=""), FakeDetector())) as c:
        assert c.get("/v1/model", headers={"Authorization": "Bearer x"}).status_code == 503


def test_token_unit():
    t = make_token({"sub": "a", "exp": 100}, "s")
    assert verify_token(t, "s", now=50)["sub"] == "a"
    with pytest.raises(AuthError):
        verify_token(t, "s", now=200)
    with pytest.raises(AuthError):
        verify_token(make_token({"sub": "a"}, "s"), "s")             # exp is mandatory


# ------------------------------------------------------------------ detection
def test_detect_happy_path(client):
    r = post(client, species="Cattle", district="Huye", animal_id="cow-7")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["diagnosis_id"].startswith("lv_") and len(j["diagnosis_id"]) == 15
    assert j["model_version"] == "yolo11x-test-v1" and j["species"] == "cattle"
    assert j["summary"] == {"skin_nodule": 1, "tick": 1}
    assert j["flags"] == ["possible_notifiable_disease"]                   # skin nodules may mean LSD
    assert set(j["detections"][0]) == {"class", "confidence", "box"} and j["latency_ms"] >= 0
    img = client.get(j["annotated_image_url"].replace("http://test", ""), headers=auth())
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    Image.open(io.BytesIO(img.content)).verify()
    rec = client.get(f"/v1/diagnosis/{j['diagnosis_id']}", headers=auth()).json()
    assert rec["animal_id"] == "cow-7" and rec["detections"] == j["detections"]
    assert r.headers["x-request-id"]


def test_zoonotic_flag_and_no_findings(tmp_path):
    zoo = FakeDetector([{"class": "ringworm_patch", "confidence": 0.9, "box": [1.0, 1.0, 50.0, 50.0]}])
    with TestClient(create_app(make_settings(tmp_path), zoo)) as c:
        assert post(c).json()["flags"] == ["possible_zoonotic_risk"]
    with TestClient(create_app(make_settings(tmp_path / "b"), FakeDetector([]))) as c:
        j = post(c).json()
        assert j["detections"] == [] and j["summary"] == {} and j["flags"] == []


def test_per_class_thresholds_from_model_card(tmp_path):
    card = tmp_path / "card.json"
    card.write_text(json.dumps({"thresholds": {"tick": 0.8}, "metrics": {"mAP50-95": 0.5}}))
    with TestClient(create_app(make_settings(tmp_path, model_card_path=str(card)), FakeDetector())) as c:
        assert post(c).json()["summary"] == {"skin_nodule": 1}              # tick 0.74 < its 0.8 threshold
        assert c.get("/v1/model", headers=auth()).json()["thresholds"] == {"tick": 0.8}


def test_input_validation(client):
    assert post(client, img=b"not an image").status_code == 422
    assert post(client, img=photo("GIF")).status_code == 422                # format not allowed
    assert post(client, img=photo(size=(30, 30))).status_code == 422        # too small
    assert post(client, species="dragon").status_code == 422
    assert post(client, offline_result="{bad json").status_code == 422
    assert post(client, offline_result=json.dumps({"model_version": "n", "detections": [{"class": "tick", "box": [5, 5, 1, 1]}]})).status_code == 422
    assert post(client, img=photo("WEBP")).status_code == 200 and post(client, img=photo("PNG")).status_code == 200
    r = client.post("/v1/detect", headers=auth())
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error" and r.json()["error"]["request_id"]


def test_upload_limit_and_exif_stripped(tmp_path, det):
    with TestClient(create_app(make_settings(tmp_path, max_upload_bytes=5000), det)) as c:
        assert post(c, img=photo(size=(600, 600))).status_code == 413
    with TestClient(create_app(make_settings(tmp_path / "x"), det)) as c:
        did = post(c, img=photo(exif=True)).json()["diagnosis_id"]
        stored = Image.open(c.app.state.settings.data_dir / "images" / f"{did}.jpg")
        assert len(stored.getexif()) == 0                                   # camera metadata (incl. GPS) is not kept


def test_large_images_are_downscaled_before_inference(tmp_path):
    seen = {}

    class Spy(FakeDetector):
        def predict(self, im, conf):
            seen["size"] = im.size
            return super().predict(im, conf)
    with TestClient(create_app(make_settings(tmp_path, max_side=640), Spy())) as c:
        assert post(c, img=photo(size=(1600, 1200))).status_code == 200
    assert max(seen["size"]) == 640


# ------------------------------------------------------------------ readiness: never serve the wrong model
def test_refuses_generic_coco_model(tmp_path):
    coco = FakeDetector(names={0: "person", 17: "horse", 19: "cow"})
    with TestClient(create_app(make_settings(tmp_path), coco)) as c:
        r = c.get("/ready")
        assert r.status_code == 503 and "labels.json" in r.json()["reason"]
        assert c.get("/health").status_code == 200                          # process is alive, model is not usable
        d = post(c)
        assert d.status_code == 503 and d.json()["error"]["code"] == "model_not_ready"
    with TestClient(create_app(make_settings(tmp_path / "dev", allow_unmatched_classes=True), coco)) as c:
        assert c.get("/ready").status_code == 200                           # explicit dev override


def test_starts_without_weights(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, weights=""))) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and "VISION_WEIGHTS" in r.json()["reason"]


# ------------------------------------------------------------------ design claims
def test_inference_never_runs_concurrently(tmp_path):
    slow = FakeDetector(delay=0.05)
    app = create_app(make_settings(tmp_path, rate_limit_per_min=100), slow)

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as ac:
                files = lambda: {"image": ("a.jpg", photo(), "image/jpeg")}
                rs = await asyncio.gather(*[ac.post("/v1/detect", files=files(), headers=auth()) for _ in range(6)])
                return [r.status_code for r in rs]
    assert asyncio.run(go()) == [200] * 6
    assert slow.max_active == 1 and len(slow.threads) == 1                 # one model instance, one worker thread


def test_rate_limit_and_request_id(tmp_path, det):
    with TestClient(create_app(make_settings(tmp_path, rate_limit_per_min=3), det)) as c:
        codes = [post(c).status_code for _ in range(4)]
        assert codes == [200, 200, 200, 429]
        assert post(c, headers=auth(sub="someone-else")).status_code == 200  # limits are per user
        r = c.get("/v1/model", headers={**auth(), "X-Request-ID": "trace-abc_1"})
        assert r.headers["x-request-id"] == "trace-abc_1"
        r = c.get("/v1/model", headers={**auth(), "X-Request-ID": "bad id\r\n"})
        assert r.headers["x-request-id"] != "bad id\r\n" and "\n" not in r.headers["x-request-id"]


# ------------------------------------------------------------------ privacy / ownership
def test_ownership(client):
    did = post(client, headers=auth(sub="alice")).json()["diagnosis_id"]
    assert client.get(f"/v1/diagnosis/{did}", headers=auth(sub="alice")).status_code == 200
    assert client.get(f"/v1/diagnosis/{did}", headers=auth(sub="mallory")).status_code == 404
    assert client.get(f"/v1/annotated/{did}.jpg", headers=auth(sub="mallory")).status_code == 404
    assert client.get(f"/v1/diagnosis/{did}", headers=auth(sub="dr-vet", role="vet")).status_code == 200
    assert client.get("/v1/diagnosis/../../etc/passwd", headers=auth()).status_code == 404
    assert client.get("/v1/annotated/..%2f..%2fsecret.jpg", headers=auth()).status_code == 404


# ------------------------------------------------------------------ offline comparison + feedback + retraining loop
def test_offline_comparison_is_recorded(client):
    agree = json.dumps({"model_version": "yolo11n-v1", "detections": [
        {"class": "skin_nodule", "confidence": 0.6, "box": [1, 1, 50, 50]}, {"class": "tick", "confidence": 0.5, "box": [2, 2, 20, 20]}]})
    assert post(client, offline_result=agree).json()["offline_comparison"] == {"agrees": True, "only_online": [], "only_offline": []}
    differ = json.dumps({"model_version": "yolo11n-v1", "detections": [{"class": "wound_abscess", "confidence": 0.6, "box": [1, 1, 50, 50]}]})
    cmp = post(client, offline_result=differ).json()["offline_comparison"]
    assert cmp == {"agrees": False, "only_online": ["skin_nodule", "tick"], "only_offline": ["wound_abscess"]}


def test_feedback_and_retraining_export(tmp_path, det):
    st = make_settings(tmp_path)
    with TestClient(create_app(st, det)) as c:
        differ = json.dumps({"model_version": "n", "detections": [{"class": "wound_abscess", "confidence": 0.6, "box": [1, 1, 50, 50]}]})
        d1 = post(c, headers=auth(sub="alice"), img=photo(size=(400, 200), seed=11), offline_result=differ).json()["diagnosis_id"]
        d2 = post(c, headers=auth(sub="alice"), img=photo(size=(400, 200), seed=12)).json()["diagnosis_id"]

        fb = lambda body, **h: c.post("/v1/feedback", json=body, headers=h or auth(sub="alice"))
        assert fb({"diagnosis_id": d1, "status": "corrected"}).status_code == 422                   # corrected needs findings
        assert fb({"diagnosis_id": d1, "status": "corrected", "findings": [{"class": "nonsense", "box": [1, 1, 5, 5]}]}).status_code == 422
        assert fb({"diagnosis_id": d1, "status": "corrected", "findings": [{"class": "tick", "box": [1, 1, 5, 5]}]}, **auth(sub="mallory")).status_code == 404
        ok = fb({"diagnosis_id": d1, "status": "corrected", "findings": [{"class": "wound_abscess", "box": [100, 50, 300, 150]}], "notes": "vet check"})
        assert ok.status_code == 200 and ok.json()["findings_recorded"] == 1
        assert fb({"diagnosis_id": d2, "status": "confirmed"}).json()["findings_recorded"] == 2    # confirmed keeps the model's findings
        assert fb({"diagnosis_id": "lv_000000000000", "status": "confirmed"}).status_code == 404

    out = tmp_path / "yolo_fb"
    stats = export_corrections(st, out)
    assert stats["exported"] == 2 and stats["boxes"] == 3
    lab = (out / "labels" / f"{d1}.txt").read_text().split()
    assert lab[0] == "5"                                                   # wound_abscess keeps its permanent id
    cx, cy, w, h = map(float, lab[1:])
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.5, 0.5, 0.5, 0.5)       # [100,50,300,150] on 400x200
    assert (out / "classes.txt").read_text().split()[5] == "wound_abscess"
    only = export_corrections(st, tmp_path / "yolo_dis", only_disagreements=True)
    assert only["exported"] == 1 and only["skipped_disagreement_filter"] == 1                     # the YOLO11n/x disagreement
    # the export is directly usable by the training pipeline
    ds = build_dataset([out], tmp_path / "ds", [c["name"] for c in LABELS.classes], "detect")
    assert ds["kept_train"] + ds["kept_val"] == 2 and ds["dropped_bad_label_lines"] == 0


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LABELS_PATH", str(tmp_path / "l.json"))
    monkeypatch.setenv("VISION_CONF", "0.5")
    monkeypatch.setenv("VISION_AUTH_DISABLED", "1")
    monkeypatch.setenv("AGRICODERS_JWT_SECRET", "s")
    s = Settings.from_env()
    assert s.labels_path == tmp_path / "l.json" and s.default_threshold == 0.5 and s.auth_disabled and s.jwt_secret == "s"
    monkeypatch.delenv("VISION_AUTH_DISABLED")
    assert Settings.from_env().auth_disabled is False                     # auth is ON unless explicitly disabled


def test_any_decoder_failure_is_a_422_not_a_500(client, monkeypatch):
    """Live bug: ultralytics patches Image.open and raises ModuleNotFoundError (pi_heif) on unrecognised files."""
    import PIL.Image as PI
    real = PI.open

    def patched(*a, **k):
        try:
            return real(*a, **k)
        except Exception:
            raise ModuleNotFoundError("No module named 'pi_heif'")
    monkeypatch.setattr(PI, "open", patched)
    for junk in (b"not an image", b""):
        r = post(client, img=junk)
        assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_image", (junk, r.status_code, r.text)
    assert post(client).status_code == 200                                     # real photos still work


def test_unexpected_errors_return_the_standard_json_error(tmp_path, det, monkeypatch):
    app = create_app(make_settings(tmp_path), det)
    with TestClient(app, raise_server_exceptions=False) as c:
        async def boom(self, im):
            raise RuntimeError("secret internal detail")
        monkeypatch.setattr("livestock_vision.engine.Engine.detect", boom)
        r = post(c)
        assert r.status_code == 500 and r.headers["content-type"].startswith("application/json")
        e = r.json()["error"]
        assert e["code"] == "internal_error" and e["request_id"] and "secret" not in r.text     # no internals leak to the client


def test_openapi_declares_bearer_auth_so_docs_can_authorize(client):
    spec = client.get("/openapi.json").json()
    assert any(v.get("scheme") == "bearer" for v in spec["components"]["securitySchemes"].values())
    assert client.get("/v1/model", headers=auth()).status_code == 200            # behaviour unchanged


def test_serve_builds_env_from_a_registered_model_and_refuses_unregistered(tmp_path, monkeypatch):
    from livestock_vision.serve import env_for
    with pytest.raises(SystemExit, match="register the model first"):
        env_for("nope", tmp_path)
    d = tmp_path / "v1"
    d.mkdir()
    for f in ("best.pt", "labels.json", "model_card.json"):
        (d / f).write_text("x")
    e = env_for("v1", tmp_path, tmp_path / "data")
    assert e["VISION_WEIGHTS"].endswith("best.pt") and e["LABELS_PATH"].endswith("labels.json") and e["VISION_MODEL_VERSION"] == "v1"
    monkeypatch.delenv("AGRICODERS_JWT_SECRET", raising=False)
    from livestock_vision import serve
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="AGRICODERS_JWT_SECRET"):
        serve.main(["--version", "v1", "--models-dir", str(tmp_path)])             # no secret and no --dev: refuses to start open


def test_response_reports_the_processed_image_size_so_clients_can_draw_boxes(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, max_side=640), FakeDetector())) as c:
        j = post(c, img=photo(size=(1600, 1200))).json()                       # server downscales to 640 on the long side
        assert j["image"] == {"width": 640, "height": 480}
        assert post(c, img=photo(size=(320, 240))).json()["image"] == {"width": 320, "height": 240}   # small images untouched

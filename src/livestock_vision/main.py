"""Livestock Vision API.   uvicorn livestock_vision.main:app --host 0.0.0.0 --port 8000 --workers 1

One worker process on purpose: the YOLO model lives in this process and is called from a single dedicated thread.
Scale out with more containers behind the load balancer, not with --workers.
"""
from __future__ import annotations

import json
import logging
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from .auth import AuthError, verify_token
from .config import Settings
from .db import Detection, Feedback, make_session_factory
from .engine import Engine, ImageError, annotate, compare_offline, flags_for, prepare_image, summarise

log = logging.getLogger("livestock_vision")
ID_RE = re.compile(r"^lv_[0-9a-f]{12}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


class Box(BaseModel):
    cls: str = Field(alias="class", max_length=64)
    confidence: float = Field(default=1.0, ge=0, le=1)
    box: list[float] = Field(min_length=4, max_length=4)

    model_config = {"populate_by_name": True}

    @field_validator("box")
    @classmethod
    def sane_box(cls, v):
        if not all(abs(x) < 1e6 for x in v) or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("box must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1")
        return v


class OfflineResult(BaseModel):
    model_version: str = Field(max_length=64)
    detections: list[Box] = Field(max_length=100)


class FeedbackIn(BaseModel):
    diagnosis_id: str
    status: Literal["confirmed", "corrected"]
    findings: list[Box] | None = Field(default=None, max_length=100)    # the corrected truth (required for 'corrected')
    notes: str = Field(default="", max_length=1000)


def clean_text(v: str, n: int = 128) -> str:
    return _CTRL.sub("", v or "").strip()[:n]


def create_app(settings: Settings | None = None, detector=None) -> FastAPI:
    s = settings or Settings.from_env()
    state: dict = {}
    counters = defaultdict(float)
    hits: dict[str, deque] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app):
        s.data_dir.mkdir(parents=True, exist_ok=True)
        (s.data_dir / "images").mkdir(exist_ok=True)
        (s.data_dir / "annotated").mkdir(exist_ok=True)
        state["engine"] = Engine(s, detector)
        state["db"] = make_session_factory(s.db_url)
        log.info("startup ready=%s reason=%s", state["engine"].ready, state["engine"].reason)
        yield
        state["engine"].exec.shutdown(wait=False)

    app = FastAPI(title="Agricoders Livestock Vision API", version="0.1.0", lifespan=lifespan)
    app.state.settings = s
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    # ---------------------------------------------------------------- plumbing
    @app.middleware("http")
    async def request_id_and_metrics(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        request.state.rid = rid if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid) else uuid.uuid4().hex[:16]
        t0 = time.time()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = request.state.rid
        counters[f'http_requests_total{{path="{request.url.path if request.url.path in ("/v1/detect", "/v1/feedback", "/v1/model", "/health", "/ready") else "other"}",status="{resp.status_code}"}}'] += 1
        log.info(json.dumps({"request_id": request.state.rid, "method": request.method, "path": request.url.path,
                             "status": resp.status_code, "ms": round((time.time() - t0) * 1000, 1)}))
        return resp

    def err(request: Request, status: int, code: str, message: str):
        return JSONResponse({"error": {"code": code, "message": message, "request_id": getattr(request.state, "rid", None)}}, status)

    @app.exception_handler(ApiError)
    async def _api(request, e: ApiError):
        return err(request, e.status, e.code, e.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request, e):
        return err(request, e.status_code, "http_error", str(e.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request, e: Exception):
        log.exception("unhandled error request_id=%s", getattr(request.state, "rid", None))
        return err(request, 500, "internal_error", "Internal server error. Quote the request_id when reporting it.")

    @app.exception_handler(RequestValidationError)
    async def _val(request, e):
        return err(request, 422, "validation_error", "; ".join(f"{'.'.join(map(str, x['loc'][1:]))}: {x['msg']}" for x in e.errors())[:400])

    bearer = HTTPBearer(auto_error=False, description="Agricoders JWT (HS256)")   # declares the scheme so /docs shows 'Authorize'

    def user(request: Request, _creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
        if s.auth_disabled:
            return {"sub": "dev", "role": "admin"}
        h = request.headers.get("authorization", "")
        if not h.lower().startswith("bearer "):
            raise ApiError(401, "unauthorized", "Missing bearer token.")
        if not s.jwt_secret:
            raise ApiError(503, "auth_not_configured", "Authentication is not configured on this server.")
        try:
            c = verify_token(h[7:].strip(), s.jwt_secret)
        except AuthError as e:
            raise ApiError(401, "unauthorized", f"Invalid token: {e}.")
        c.setdefault("role", "farmer")
        return c

    def limit(u: dict):
        q, now = hits[u["sub"]], time.time()
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= s.rate_limit_per_min:
            raise ApiError(429, "rate_limited", "Too many requests. Please wait a minute.")
        q.append(now)

    def privileged(u: dict) -> bool:
        return u["role"] in s.privileged_roles

    def need_ready() -> Engine:
        e: Engine = state["engine"]
        if not e.ready:
            raise ApiError(503, "model_not_ready", f"The detection model is not available: {e.reason}")
        return e

    def own_or_priv(rec: Detection, u: dict):
        if rec.user_id != u["sub"] and not privileged(u):
            raise ApiError(404, "not_found", "No such diagnosis.")        # 404, not 403: do not reveal that it exists

    # ---------------------------------------------------------------- endpoints
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        e: Engine = state["engine"]
        return JSONResponse({"ready": e.ready, "reason": e.reason, "model_version": s.model_version}, 200 if e.ready else 503)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return "".join(f"{k} {v}\n" for k, v in sorted(counters.items()))

    @app.get("/v1/model")
    async def model_info(u: dict = Depends(user)):
        e: Engine = state["engine"]
        man = json.loads(Path(s.manifest_path).read_text(encoding="utf-8")) if Path(s.manifest_path).exists() else {}
        return {"model_version": s.model_version, "dataset_version": s.dataset_version, "ready": e.ready,
                "labels_version": e.labels.raw.get("version"), "reviewed_by_veterinarian": e.labels.raw.get("reviewed_by_veterinarian"),
                "classes": e.labels.classes, "species": sorted(e.labels.species), "default_threshold": s.default_threshold,
                "thresholds": e.thresholds, "metrics": e.metrics, "yolo11n": man.get("yolo11n")}

    @app.post("/v1/detect")
    async def detect(request: Request, image: UploadFile = File(...), species: str = Form(""), district: str = Form(""),
                     animal_id: str = Form(""), offline_result: str = Form(""), u: dict = Depends(user)):
        limit(u)
        eng = need_ready()
        species = clean_text(species, 32).lower()
        if species and species not in eng.labels.species:
            raise ApiError(422, "validation_error", f"species must be one of {sorted(eng.labels.species)}")
        data = await image.read(s.max_upload_bytes + 1)
        if len(data) > s.max_upload_bytes:
            raise ApiError(413, "image_too_large", f"Image larger than {s.max_upload_bytes // 1_000_000} MB.")
        try:
            im = prepare_image(data, s.max_side)
        except ImageError as e:
            raise ApiError(422, "invalid_image", str(e))
        offline = None
        if offline_result.strip():
            try:
                offline = OfflineResult.model_validate_json(offline_result)
            except ValidationError:
                raise ApiError(422, "validation_error", "offline_result must be {model_version, detections:[{class,confidence,box}]}")

        t0 = time.time()
        dets = await eng.detect(im)
        ms = round((time.time() - t0) * 1000, 1)
        counters["detections_total"] += len(dets)
        counters["detect_latency_ms_sum"] += ms
        counters["detect_latency_ms_count"] += 1

        did = "lv_" + secrets.token_hex(6)
        summary, flags = summarise(dets), flags_for(dets, eng.labels)
        cmp = compare_offline(dets, [d.model_dump(by_alias=True) for d in offline.detections]) if offline else None
        img_p, ann_p = s.data_dir / "images" / f"{did}.jpg", s.data_dir / "annotated" / f"{did}.jpg"

        def persist():
            im.save(img_p, "JPEG", quality=90)                      # re-encoded: EXIF/GPS metadata is not kept
            annotate(im, dets).save(ann_p, "JPEG", quality=88)
            with state["db"]() as db:
                db.add(Detection(id=did, user_id=u["sub"], species=species, district=clean_text(district),
                                 animal_id=clean_text(animal_id), model_version=s.model_version, image_path=str(img_p),
                                 annotated_path=str(ann_p), detections=json.dumps(dets), summary=json.dumps(summary),
                                 flags=json.dumps(flags), latency_ms=ms,
                                 offline_model_version=offline.model_version if offline else None,
                                 offline_detections=json.dumps([d.model_dump(by_alias=True) for d in offline.detections]) if offline else None,
                                 disagreement=bool(cmp and not cmp["agrees"])))
                db.commit()

        await run_in_threadpool(persist)
        out = {"diagnosis_id": did, "model_version": s.model_version, "species": species, "detections": dets,
               "summary": summary, "flags": flags, "annotated_image_url": f"{s.public_base_url}/v1/annotated/{did}.jpg",
               "image": {"width": im.width, "height": im.height},      # box coordinates are in THIS image's pixels (after any downscale)
               "latency_ms": ms}
        if cmp:
            out["offline_comparison"] = cmp
        return out

    @app.get("/v1/diagnosis/{diagnosis_id}")
    async def get_diagnosis(diagnosis_id: str, u: dict = Depends(user)):
        if not ID_RE.match(diagnosis_id):
            raise ApiError(404, "not_found", "No such diagnosis.")

        def load():
            with state["db"]() as db:
                return db.get(Detection, diagnosis_id)
        rec = await run_in_threadpool(load)
        if rec is None:
            raise ApiError(404, "not_found", "No such diagnosis.")
        own_or_priv(rec, u)
        return {"diagnosis_id": rec.id, "model_version": rec.model_version, "species": rec.species, "district": rec.district,
                "animal_id": rec.animal_id, "detections": json.loads(rec.detections), "summary": json.loads(rec.summary),
                "flags": json.loads(rec.flags), "created_at": rec.created_at.isoformat() if rec.created_at else None}

    @app.get("/v1/annotated/{name}")
    async def annotated(name: str, u: dict = Depends(user)):
        m = re.fullmatch(r"(lv_[0-9a-f]{12})\.jpg", name)
        if not m:
            raise ApiError(404, "not_found", "No such image.")

        def load():
            with state["db"]() as db:
                return db.get(Detection, m.group(1))
        rec = await run_in_threadpool(load)
        if rec is None or not Path(rec.annotated_path).exists():
            raise ApiError(404, "not_found", "No such image.")
        own_or_priv(rec, u)
        return FileResponse(rec.annotated_path, media_type="image/jpeg")

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackIn, u: dict = Depends(user)):
        """A vet or the user confirms or corrects the findings; corrections become labelled training data."""
        limit(u)
        if not ID_RE.match(body.diagnosis_id):
            raise ApiError(404, "not_found", "No such diagnosis.")
        eng: Engine = state["engine"]
        if body.status == "corrected" and body.findings is None:
            raise ApiError(422, "validation_error", "findings are required when status is 'corrected' (use [] for 'nothing there').")
        for f in body.findings or []:
            if f.cls not in eng.labels.by_name:
                raise ApiError(422, "validation_error", f"unknown class '{f.cls}'")

        def save():
            with state["db"]() as db:
                rec = db.get(Detection, body.diagnosis_id)
                if rec is None:
                    return None
                own_or_priv(rec, u)
                truth = [f.model_dump(by_alias=True) for f in body.findings] if body.findings is not None else json.loads(rec.detections)
                db.add(Feedback(diagnosis_id=rec.id, user_id=u["sub"], status=body.status, findings=json.dumps(truth),
                                notes=clean_text(body.notes, 1000)))
                db.commit()
                return len(truth)
        n = await run_in_threadpool(save)
        if n is None:
            raise ApiError(404, "not_found", "No such diagnosis.")
        counters[f'feedback_total{{status="{body.status}"}}'] += 1
        return {"ok": True, "diagnosis_id": body.diagnosis_id, "status": body.status, "findings_recorded": n}

    return app


app = create_app()          # configuration comes from environment variables (see config.py); nothing heavy happens until startup

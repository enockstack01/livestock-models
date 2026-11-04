"""LLM API.   uvicorn livestock_llm.main:app --host 0.0.0.0 --port 8002

POST /v1/report  {diagnosis_id, language, animal{weight_kg,...}, stream}   structured veterinary advisory report
POST /v1/chat    {messages, diagnosis_id?, language, stream}               advisory chat (SSE streaming)
GET  /v1/report/{diagnosis_id}                                              latest stored report
GET  /v1/model, /health, /ready, /metrics

Design: the API takes a diagnosis_id and READS the detections from the database the Vision API wrote; the two services never call
each other, and every report is grounded in a real stored detection. If the language model is down, slow, overloaded or returns an
invalid report, the user still gets a complete report from the approved advice pack (source: "advice_pack").
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from livestock_vision.auth import AuthError, verify_token
from livestock_vision.db import Detection, make_session_factory
from livestock_vision.main import ID_RE, ApiError, clean_text

from .advice import AdvicePack
from .config import Settings
from .llm import LLMClient, LLMUnavailable, ReportInvalid, extract_json
from .prompts import LANG_NAMES, chat_messages, context_block, repair_messages, report_messages
from .schema import AnimalContext, Report, report_json_schema, sanitize_report
from .store import LlmReport

log = logging.getLogger("livestock_llm")
Lang = Literal["en", "fr", "rw"]


class ReportIn(BaseModel):
    diagnosis_id: str
    language: Lang = "en"
    animal: AnimalContext | None = None
    stream: bool = False
    refresh: bool = False


class Msg(BaseModel):
    role: Literal["user", "assistant"]              # 'system' is not accepted: clients can never inject instructions
    content: str = Field(min_length=1, max_length=2000)


class ChatIn(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=20)
    diagnosis_id: str | None = None
    language: Lang = "en"
    think: bool = False                              # thinking mode only for hard questions
    stream: bool = True


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(settings: Settings | None = None, transport=None) -> FastAPI:
    s = settings or Settings.from_env()
    st: dict = {}
    counters = defaultdict(float)
    hits: dict[str, deque] = defaultdict(deque)

    @asynccontextmanager
    async def lifespan(app):
        s.data_dir.mkdir(parents=True, exist_ok=True)
        st["db"] = make_session_factory(s.db_url)
        st["advice"] = AdvicePack(s.advice_path)
        st["llm"] = LLMClient(s, transport)
        st["sem"] = asyncio.Semaphore(s.max_concurrency)
        yield
        await st["llm"].aclose()

    app = FastAPI(title="Agricoders LLM API (livestock)", version="0.1.0", lifespan=lifespan)
    app.state.settings = s
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["Authorization", "Content-Type", "X-Request-ID"], expose_headers=["X-Request-ID"])

    # ---------------------------------------------------------------- plumbing
    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")
        request.state.rid = rid if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid) else uuid.uuid4().hex[:16]
        t0 = time.time()
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = request.state.rid
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

    bearer = HTTPBearer(auto_error=False, description="Agricoders JWT (HS256), the same token as the Vision API")

    def user(request: Request, _c: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
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

    async def load_detection(did: str, u: dict) -> dict:
        if not ID_RE.match(did):
            raise ApiError(404, "not_found", "No such diagnosis.")

        def load():
            with st["db"]() as db:
                r = db.get(Detection, did)
                return None if r is None else {"id": r.id, "user_id": r.user_id, "species": r.species, "district": r.district,
                                               "model_version": r.model_version, "detections": json.loads(r.detections),
                                               "summary": json.loads(r.summary), "flags": json.loads(r.flags)}
        rec = await run_in_threadpool(load)
        if rec is None or (rec["user_id"] != u["sub"] and u["role"] not in s.privileged_roles):
            raise ApiError(404, "not_found", "No such diagnosis.")           # same answer for "missing" and "not yours"
        return rec

    # ---------------------------------------------------------------- generation
    def parse(text: str) -> Report:
        try:
            return Report.model_validate(extract_json(text))
        except ValidationError as e:
            raise ReportInvalid(f"schema: {str(e.errors()[0]['loc'])} {e.errors()[0]['msg']}"[:200])

    async def take_slot():
        try:
            await asyncio.wait_for(st["sem"].acquire(), s.queue_wait_s)
        except asyncio.TimeoutError:
            raise LLMUnavailable("overloaded: no free generation slot")

    async def llm_report(msgs: list[dict]) -> Report:
        text = await st["llm"].complete(msgs, max_tokens=s.report_max_tokens, schema=report_json_schema())
        try:
            return parse(text)
        except ReportInvalid as first:                                       # one repair attempt, then the advice pack
            counters["report_repairs_total"] += 1
            return parse(await st["llm"].complete(repair_messages(msgs, text, str(first)), max_tokens=s.report_max_tokens, schema=report_json_schema()))

    def finish(rec: dict, report: Report, source: str, notes: list[str], language: str, animal: AnimalContext | None, t0: float) -> dict:
        safe, adj = sanitize_report(report, rec["flags"], animal)
        if source == "advice_pack" and language != "en":
            notes = notes + [f"the advice pack is available in English only (requested: {LANG_NAMES[language]})"]
        counters[f'reports_total{{source="{source}"}}'] += 1
        return {"diagnosis_id": rec["id"], "model_version": s.llm_model if source == "llm" else f"advice-pack-v{st['advice'].version}",
                "language": language, "source": source, "degraded": source != "llm", "report": safe.model_dump(),
                "safety_adjustments": notes + adj, "advice_pack_reviewed_by_vet": st["advice"].reviewed,
                "latency_ms": round((time.time() - t0) * 1000, 1)}

    def report_context(rec: dict, animal: AnimalContext | None):
        adv = st["advice"].for_detections(rec["detections"])
        ctx = context_block(rec["species"], rec["district"], rec["detections"], rec["summary"], rec["flags"], adv,
                            animal.model_dump() if animal else None, rec["model_version"])
        return ctx

    def cache_key(did: str, language: str, animal: AnimalContext | None) -> str:
        return hashlib.sha1(json.dumps([did, language, animal.model_dump() if animal else None], sort_keys=True).encode()).hexdigest()

    def persist(rec: dict, out: dict, key: str, u: dict, animal: AnimalContext | None):
        with st["db"]() as db:
            db.add(LlmReport(diagnosis_id=rec["id"], key=key, user_id=u["sub"], language=out["language"], source=out["source"],
                             model_version=out["model_version"], animal=json.dumps(animal.model_dump() if animal else {}),
                             report=json.dumps(out["report"], ensure_ascii=False), adjustments=json.dumps(out["safety_adjustments"])))
            db.commit()

    def stored(did: str, key: str | None = None) -> dict | None:
        with st["db"]() as db:
            q = db.query(LlmReport).filter(LlmReport.diagnosis_id == did)
            if key:
                q = q.filter(LlmReport.key == key)
            r = q.order_by(LlmReport.id.desc()).first()
            return None if r is None else {"diagnosis_id": r.diagnosis_id, "model_version": r.model_version, "language": r.language,
                                           "source": r.source, "degraded": r.source != "llm", "report": json.loads(r.report),
                                           "safety_adjustments": json.loads(r.adjustments), "cached": True,
                                           "created_at": r.created_at.isoformat() if r.created_at else None}

    # ---------------------------------------------------------------- endpoints
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        """Ready as long as reports can be served: the advice-pack fallback means an LLM outage does not make the API unready."""
        llm_ok = await st["llm"].ready()
        return {"ready": True, "llm_reachable": llm_ok, "fallback_available": True, "advice_pack_version": st["advice"].version}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        return "".join(f"{k} {v}\n" for k, v in sorted(counters.items()))

    @app.get("/v1/model")
    async def model_info(u: dict = Depends(user)):
        return {"llm_model": s.llm_model, "llm_reachable": await st["llm"].ready(), "languages": LANG_NAMES, "thinking": "off for reports, optional for chat",
                "max_concurrency": s.max_concurrency, "report_timeout_s": s.llm_timeout_s,
                "advice_pack": {"version": st["advice"].version, "reviewed_by_veterinarian": st["advice"].reviewed, "entries": len(st["advice"].entries)},
                "limits": {"chat_messages": 20, "message_chars": 2000}}

    @app.post("/v1/report")
    async def report(body: ReportIn, u: dict = Depends(user)):
        limit(u)
        rec = await load_detection(body.diagnosis_id, u)
        key = cache_key(rec["id"], body.language, body.animal)
        if not body.refresh:
            hit = await run_in_threadpool(stored, rec["id"], key)
            if hit:
                if body.stream:                                              # streaming clients get the same final event
                    async def one():
                        yield sse("final", hit)
                    return StreamingResponse(one(), media_type="text/event-stream")
                return hit
        msgs = report_messages(report_context(rec, body.animal), body.language)
        t0 = time.time()

        if not body.stream:
            notes, source = [], "llm"
            try:
                await take_slot()
                try:
                    rep = await llm_report(msgs)
                finally:
                    st["sem"].release()
            except (LLMUnavailable, ReportInvalid) as e:
                counters["llm_failures_total"] += 1
                rep, source, notes = st["advice"].fallback_report(rec["detections"], rec["flags"]), "advice_pack", [f"fallback used: {e}"]
            out = finish(rec, rep, source, notes, body.language, body.animal, t0)
            await run_in_threadpool(persist, rec, out, key, u, body.animal)
            return out

        async def events():
            yield sse("start", {"diagnosis_id": rec["id"], "language": body.language})
            notes, source, buf = [], "llm", []
            try:
                await take_slot()
                try:
                    async for delta in st["llm"].stream(msgs, max_tokens=s.report_max_tokens):
                        buf.append(delta)
                        yield sse("delta", {"chars": sum(map(len, buf))})   # progress only: raw JSON fragments are not useful to a UI
                finally:
                    st["sem"].release()
                rep = parse("".join(buf))
            except (LLMUnavailable, ReportInvalid) as e:
                counters["llm_failures_total"] += 1
                rep, source, notes = st["advice"].fallback_report(rec["detections"], rec["flags"]), "advice_pack", [f"fallback used: {e}"]
            out = finish(rec, rep, source, notes, body.language, body.animal, t0)
            await run_in_threadpool(persist, rec, out, key, u, body.animal)
            yield sse("final", out)                                          # a final event ALWAYS arrives, from the LLM or the pack
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/report/{diagnosis_id}")
    async def get_report(diagnosis_id: str, u: dict = Depends(user)):
        await load_detection(diagnosis_id, u)                               # ownership check
        out = await run_in_threadpool(stored, diagnosis_id)
        if out is None:
            raise ApiError(404, "not_found", "No report has been generated for this diagnosis yet.")
        return out

    @app.post("/v1/chat")
    async def chat(body: ChatIn, u: dict = Depends(user)):
        limit(u)
        if body.messages[-1].role != "user":
            raise ApiError(422, "validation_error", "the last message must be from the user")
        if sum(len(m.content) for m in body.messages) > 8000:
            raise ApiError(422, "validation_error", "conversation too long (8000 characters maximum)")
        ctx = None
        if body.diagnosis_id:
            rec = await load_detection(body.diagnosis_id, u)
            ctx = report_context(rec, None)
        history = [{"role": m.role, "content": re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", m.content)} for m in body.messages]
        msgs = chat_messages(ctx, body.language, history)
        counters["chat_total"] += 1
        try:
            await take_slot()
        except LLMUnavailable as e:
            raise ApiError(503, "llm_unavailable", f"The assistant is busy ({e}). The standard advice in the report still applies.")

        unavailable = ApiError(503, "llm_unavailable", "The assistant is not available right now. The standard advice in the report still applies.")
        if not body.stream:
            try:
                text = await st["llm"].complete(msgs, max_tokens=s.chat_max_tokens, think=body.think)
            except LLMUnavailable:
                counters["llm_failures_total"] += 1
                raise unavailable
            finally:
                st["sem"].release()
            return {"answer": re.sub(r"(?s)<think>.*?</think>", "", text).strip(), "language": body.language}

        it = st["llm"].stream(msgs, max_tokens=s.chat_max_tokens, think=body.think).__aiter__()
        try:
            first = await it.__anext__()                       # fail with a proper 503 BEFORE the stream starts
        except StopAsyncIteration:
            first = ""
        except LLMUnavailable:
            st["sem"].release()
            counters["llm_failures_total"] += 1
            raise unavailable

        async def events():
            try:
                if first:
                    yield sse("delta", {"text": first})
                async for delta in it:
                    yield sse("delta", {"text": delta})
                yield sse("done", {"language": body.language})
            except LLMUnavailable:
                yield sse("error", {"code": "llm_unavailable", "message": "The assistant stopped responding."})
            finally:
                st["sem"].release()                            # also runs if the client disconnects mid-stream
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()

"""Client for any OpenAI-compatible endpoint serving Qwen: a hosted provider or your own vLLM. Raises only LLMUnavailable to callers."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator

import httpx

from .config import Settings

log = logging.getLogger("livestock_llm")
MODES = ["schema", "json_object", "none"]


class LLMUnavailable(Exception):
    """The language model could not answer (down, slow, overloaded or returned an error)."""


class ReportInvalid(Exception):
    """The model answered, but not with a valid report."""


_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)


def extract_json(text: str) -> dict:
    t = _FENCE.sub("", _THINK.sub("", text or "").strip()).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ReportInvalid("no JSON object in the model output")
    try:
        obj = json.loads(t[a:b + 1])
    except json.JSONDecodeError as e:
        raise ReportInvalid(f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise ReportInvalid("the JSON is not an object")
    return obj


class LLMClient:
    def __init__(self, s: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = s
        self.mode = s.llm_json_mode if s.llm_json_mode in MODES else "schema"     # lowered when the provider rejects it
        key = (s.llm_api_key or "").strip()
        headers = {"Authorization": f"Bearer {key}"} if key and key != "EMPTY" else {}   # the key is never logged or put in an error
        self.http = httpx.AsyncClient(base_url=s.llm_base_url.rstrip("/"), timeout=httpx.Timeout(s.llm_timeout_s, connect=5.0),
                                      headers=headers, transport=transport)

    def _payload(self, messages, *, max_tokens, stream, think, schema=None) -> dict:
        p = {"model": self.s.llm_model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens, "stream": stream}
        if self.s.llm_template_kwargs:
            # Qwen3-family chat template switch (vLLM): reports run with thinking OFF (fast, grounded); hard chat questions may turn it on
            p["chat_template_kwargs"] = {"enable_thinking": think}
        if schema is not None and self.mode == "schema":
            p["response_format"] = {"type": "json_schema", "json_schema": {"name": "report", "schema": schema, "strict": True}}
        elif schema is not None and self.mode == "json_object":
            p["response_format"] = {"type": "json_object"}
        return p

    async def _post(self, make_payload, *, schema):
        """POST with: retries on 429/503 (short Retry-After honoured) and automatic step-down of the JSON mode on HTTP 400."""
        r = None
        for attempt in range(4):
            r = await self.http.post("/chat/completions", json=make_payload())
            if r.status_code == 400 and schema is not None and self.mode != "none":
                self.mode = MODES[MODES.index(self.mode) + 1]
                log.warning("provider rejected the structured-output mode; falling back to %r", self.mode)
                continue
            if r.status_code in (429, 503) and attempt < 3:
                try:
                    wait = min(3.0, float(r.headers.get("retry-after", "1")))
                except ValueError:
                    wait = 1.0
                await asyncio.sleep(wait)
                continue
            break
        if r.status_code in (401, 403):
            log.error("LLM provider refused the credentials (HTTP %s): check LLM_API_KEY", r.status_code)
        r.raise_for_status()
        return r

    @staticmethod
    def _why(e: Exception) -> str:
        if isinstance(e, httpx.HTTPStatusError):
            return f"provider answered HTTP {e.response.status_code}"
        return f"{type(e).__name__}: {str(e)[:120]}"

    async def complete(self, messages: list[dict], *, max_tokens: int, think: bool = False, schema: dict | None = None) -> str:
        try:
            r = await self._post(lambda: self._payload(messages, max_tokens=max_tokens, stream=False, think=think, schema=schema), schema=schema)
            return r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMUnavailable(self._why(e))

    async def stream(self, messages: list[dict], *, max_tokens: int, think: bool = False) -> AsyncIterator[str]:
        try:
            async with self.http.stream("POST", "/chat/completions", json=self._payload(messages, max_tokens=max_tokens, stream=True, think=think)) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    except (ValueError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as e:
            raise LLMUnavailable(self._why(e))

    async def ready(self) -> bool:
        try:
            r = await self.http.get("/models", timeout=3.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def aclose(self):
        await self.http.aclose()

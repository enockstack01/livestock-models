"""The LLM client against a hosted, OpenAI-compatible provider (httpx.MockTransport stand-ins; no real provider was contacted)."""
import asyncio
import json

import httpx
import pytest

from livestock_llm.config import Settings
from livestock_llm.llm import LLMClient, LLMUnavailable
from livestock_llm.serve import main as serve_main

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}
MSG = [{"role": "user", "content": "hi"}]


def run(coro):
    return asyncio.run(coro)


def client(handler, **kw):
    return LLMClient(Settings(**kw), httpx.MockTransport(handler))


def ok(content="{}"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def test_api_key_is_sent_only_when_configured():
    seen = []

    def h(req):
        seen.append(req.headers.get("authorization"))
        return ok()

    run(client(h, llm_api_key="sk-secret").complete(MSG, max_tokens=5))
    run(client(h).complete(MSG, max_tokens=5))                       # default "EMPTY": self-hosted vLLM, no header
    assert seen == ["Bearer sk-secret", None]


def test_vllm_only_field_can_be_turned_off_for_hosted_providers():
    bodies = []

    def h(req):
        bodies.append(json.loads(req.content))
        return ok()

    run(client(h).complete(MSG, max_tokens=5))
    run(client(h, llm_template_kwargs=False).complete(MSG, max_tokens=5))
    assert "chat_template_kwargs" in bodies[0] and "chat_template_kwargs" not in bodies[1]


def test_structured_output_steps_down_when_provider_answers_400_and_remembers():
    modes = []

    def h(req):
        rf = json.loads(req.content).get("response_format")
        modes.append(rf["type"] if rf else None)
        return httpx.Response(400, json={"error": "unsupported"}) if rf and rf["type"] == "json_schema" else ok('{"a": "x"}')

    c = client(h)
    assert run(c.complete(MSG, max_tokens=5, schema=SCHEMA)) == '{"a": "x"}'
    assert modes == ["json_schema", "json_object"]
    run(c.complete(MSG, max_tokens=5, schema=SCHEMA))                # remembered: no second wasted 400
    assert modes[2:] == ["json_object"]


def test_400_without_a_structured_mode_to_drop_is_an_error_not_a_loop():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(400, json={"error": "bad"})

    with pytest.raises(LLMUnavailable, match="HTTP 400"):
        run(client(h).complete(MSG, max_tokens=5, schema=SCHEMA))
    assert len(calls) == 3                                            # schema, json_object, none: then stop


def test_rate_limit_is_retried_then_succeeds(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    n = []

    def h(req):
        n.append(1)
        return httpx.Response(429, headers={"retry-after": "1"}) if len(n) < 3 else ok("done")

    assert run(client(h).complete(MSG, max_tokens=5)) == "done" and len(n) == 3


def test_persistent_rate_limit_becomes_llm_unavailable(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    with pytest.raises(LLMUnavailable, match="HTTP 429"):
        run(client(lambda r: httpx.Response(429)).complete(MSG, max_tokens=5))


def test_bad_key_error_never_contains_the_key():
    with pytest.raises(LLMUnavailable) as e:
        run(client(lambda r: httpx.Response(401, json={"error": "bad key"}), llm_api_key="sk-secret").complete(MSG, max_tokens=5))
    assert "sk-secret" not in str(e.value) and "401" in str(e.value)


def test_serve_refuses_to_send_a_key_over_plain_http_to_a_remote_host(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    with pytest.raises(SystemExit, match="https"):
        serve_main(["--llm-url", "http://llm.example.com/v1", "--dev"])

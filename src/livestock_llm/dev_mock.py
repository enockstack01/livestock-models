"""A MOCK OpenAI-compatible LLM server for development without a GPU:  python -m livestock_llm.dev_mock --port 8003

It is NOT a language model. It builds a report from the findings in the request (using the approved advice pack) and answers chat with
a fixed, honest message, so the Vision API, the LLM API, the web app and the mobile app can be developed and tested end to end on a
laptop. Every text it produces says "MOCK". Quality of real Qwen output is NOT represented by this server.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .advice import AdvicePack
from .config import ROOT

app = FastAPI(title="MOCK vLLM (development only)")
PACK = AdvicePack(ROOT / "shared" / "advice_pack.json")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "mock-qwen", "object": "model"}]}


def mock_report(user_text: str) -> str:
    data = json.loads(user_text.split("DATA:\n", 1)[1])
    dets = [{"class": f["class"], "confidence": f["best_confidence"]} for f in data["findings"]]
    base = PACK.fallback_report(dets, data.get("flags", []))
    r = base.model_dump()
    r["summary"] = ("(MOCK LLM, not a real model) " + base.summary.split("This is standard")[0]).strip()
    r["treatment"]["medicines"] = []                      # the mock never recommends medicines
    return json.dumps(r, ensure_ascii=False)


def mock_chat(messages: list[dict]) -> str:
    last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return (f"(MOCK assistant, not a real language model) You asked: \"{last[:120]}\". A real model would answer here. "
            "For now: isolate any animal you are worried about, keep it comfortable with shade and clean water, and contact a veterinarian. "
            "Suspected foot-and-mouth disease or lumpy skin disease must be reported to the sector veterinary officer / RAB.")


@app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    msgs = body["messages"]
    await asyncio.sleep(0.05)
    is_report = "Output ONLY one JSON object" in msgs[0]["content"]
    text = mock_report(msgs[-1]["content"]) if is_report and "DATA:\n" in msgs[-1]["content"] else mock_chat(msgs)
    if not body.get("stream"):
        return JSONResponse({"id": "mock", "object": "chat.completion", "created": int(time.time()), "model": "mock-qwen",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})

    async def gen():
        for i in range(0, len(text), 24):
            yield "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": text[i:i + 24]}}]}) + "\n\n"
            await asyncio.sleep(0.01)
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)

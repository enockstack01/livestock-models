"""Start the LLM API.   python -m livestock_llm.serve --llm-url http://vllm-host:8000/v1 [--dev]

The model is reached through any OpenAI-compatible endpoint, so it does NOT have to run on your own GPU:
    set LLM_API_KEY=<key from your provider>        (environment only, never on the command line)
    python -m livestock_llm.serve --llm-url https://<provider>/v1 --model <the provider's Qwen model id> --port 8002
Hosted providers may reject vLLM-only fields: if requests fail with HTTP 400, set LLM_TEMPLATE_KWARGS=0.
A self-hosted vLLM server works the same way (--llm-url http://host:8000/v1). The wrapper needs POST /chat/completions (and GET /models for /ready).
Authentication stays ON: set AGRICODERS_JWT_SECRET (shared with the other Agricoders APIs), or --dev for a local secret shared
with `python -m livestock_vision.serve --dev` (one printed token then works on both services).
"""
from __future__ import annotations

import argparse
import os
import time

from livestock_vision.auth import dev_secret, make_token


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-url", default=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "Qwen/Qwen3.6-27B-FP8"))
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 only behind TLS and real authentication")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--data-dir", help="must be the Vision API's data dir (shared database)")
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    from urllib.parse import urlparse
    u = urlparse(a.llm_url)
    if os.environ.get("LLM_API_KEY", "EMPTY") != "EMPTY" and u.scheme != "https" and u.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise SystemExit("Refusing to send LLM_API_KEY over plain http to a remote host: use an https:// URL.")
    os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"] = a.llm_url, a.model
    if a.data_dir:
        os.environ["VISION_DATA_DIR"] = a.data_dir
    if a.dev:
        os.environ["AGRICODERS_JWT_SECRET"] = dev_secret()
        tok = make_token({"sub": "dev-user", "role": "vet", "exp": time.time() + 8 * 3600}, os.environ["AGRICODERS_JWT_SECRET"])
        print(f"\nDEV TOKEN (valid 8 h, role=vet; works on the Vision API too):\n{tok}\n", flush=True)
    elif not os.environ.get("AGRICODERS_JWT_SECRET"):
        raise SystemExit("Set AGRICODERS_JWT_SECRET (shared with the other Agricoders APIs), or pass --dev for local testing.")
    import uvicorn
    uvicorn.run("livestock_llm.main:app", host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()

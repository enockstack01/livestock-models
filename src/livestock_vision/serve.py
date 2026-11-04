"""Start the Vision API on a registered model:  python -m livestock_vision.serve --version server-weak-v1 [--dev]

Reads models/<version>/{best.pt, labels.json, model_card.json} (written by livestock_vision.register) and sets the environment the
service expects, so you do not have to juggle variables. Authentication stays ON:
  * production: set AGRICODERS_JWT_SECRET to the secret shared with the Agricoders login service
  * --dev: uses a random local secret (./.dev_jwt_secret, shared with the LLM API's --dev) and prints a ready-to-use token
One worker process only: the model lives in the process and is called from a single thread.
"""
from __future__ import annotations

import argparse
import os
import secrets
import time
from pathlib import Path

from .auth import dev_secret, make_token
from .config import ROOT


def env_for(version: str, models_dir: Path, data_dir: Path | None = None) -> dict:
    d = Path(models_dir) / version
    for f in ("best.pt", "labels.json", "model_card.json"):
        if not (d / f).exists():
            raise SystemExit(f"{d / f} not found: register the model first (python -m livestock_vision.register ...)")
    env = {"VISION_WEIGHTS": str(d / "best.pt"), "VISION_MODEL_VERSION": version, "VISION_MODEL_CARD": str(d / "model_card.json"),
           "LABELS_PATH": str(d / "labels.json")}
    if data_dir:
        env["VISION_DATA_DIR"] = str(data_dir)
    return env


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only behind TLS and real authentication")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir")
    ap.add_argument("--dev", action="store_true", help="random secret + printed dev token (never use for a public service)")
    a = ap.parse_args(argv)
    os.environ.update(env_for(a.version, Path(a.models_dir), Path(a.data_dir) if a.data_dir else None))
    os.environ.setdefault("PUBLIC_BASE_URL", f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}")
    if a.dev:
        os.environ["AGRICODERS_JWT_SECRET"] = dev_secret()          # shared with livestock_llm.serve --dev
        tok = make_token({"sub": "dev-user", "role": "vet", "exp": time.time() + 8 * 3600}, os.environ["AGRICODERS_JWT_SECRET"])
        print(f"\nDEV TOKEN (valid 8 h, role=vet):\n{tok}\nDocs: {os.environ['PUBLIC_BASE_URL']}/docs  -> Authorize -> paste the token\n", flush=True)
    elif not os.environ.get("AGRICODERS_JWT_SECRET"):
        raise SystemExit("Set AGRICODERS_JWT_SECRET (the secret shared with the Agricoders login service), or pass --dev for local testing.")
    import uvicorn
    uvicorn.run("livestock_vision.main:app", host=a.host, port=a.port, workers=1)


if __name__ == "__main__":
    main()

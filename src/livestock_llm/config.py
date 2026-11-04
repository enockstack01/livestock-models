from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _b(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    # --- the language model: any OpenAI-compatible server (vLLM serving Qwen3.6-27B-FP8)
    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen3.6-27B-FP8"
    llm_api_key: str = "EMPTY"             # hosted provider: set LLM_API_KEY in the environment (never on a command line); "EMPTY" = send no key
    llm_template_kwargs: bool = True       # vLLM-only `chat_template_kwargs`; hosted providers may reject it -> LLM_TEMPLATE_KWARGS=0
    llm_json_mode: str = "schema"          # schema -> json_object -> none; steps down automatically when the server answers HTTP 400
    llm_timeout_s: float = 60.0            # one generation; on timeout the report falls back to the advice pack
    report_max_tokens: int = 1500
    chat_max_tokens: int = 1000
    max_concurrency: int = 8               # simultaneous generations sent to vLLM
    queue_wait_s: float = 5.0              # waiting longer than this for a slot -> advice-pack fallback instead of a long queue
    # --- shared platform pieces
    data_dir: Path = ROOT / "vision_data"
    database_url: str = ""                 # SAME database as the Vision API: it reads the stored detections by diagnosis_id
    jwt_secret: str = ""                   # same shared HS256 secret as every Agricoders API
    auth_disabled: bool = False            # development only
    advice_path: Path = ROOT / "shared" / "advice_pack.json"
    cors_origins: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8080", "http://localhost:8080"])
    rate_limit_per_min: int = 20
    privileged_roles: tuple[str, ...] = ("vet", "admin")

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        s = cls(llm_base_url=e("LLM_BASE_URL", cls.llm_base_url), llm_model=e("LLM_MODEL", cls.llm_model),
                llm_api_key=e("LLM_API_KEY", "EMPTY"), llm_timeout_s=float(e("LLM_TIMEOUT_S", "60")),
                max_concurrency=int(e("LLM_MAX_CONCURRENCY", "8")), data_dir=Path(e("VISION_DATA_DIR", str(ROOT / "vision_data"))),
                database_url=e("DATABASE_URL", ""), jwt_secret=e("AGRICODERS_JWT_SECRET", ""),
                auth_disabled=_b("LLM_AUTH_DISABLED"), llm_template_kwargs=_b("LLM_TEMPLATE_KWARGS", True),
                llm_json_mode=e("LLM_JSON_MODE", "schema"), rate_limit_per_min=int(e("LLM_RATE_LIMIT", "20")))
        if e("ADVICE_PACK"):
            s.advice_path = Path(e("ADVICE_PACK"))
        if e("CORS_ORIGINS"):
            s.cors_origins = [o.strip() for o in e("CORS_ORIGINS").split(",") if o.strip()]
        return s

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'vision.db').as_posix()}"

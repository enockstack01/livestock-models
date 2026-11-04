from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _b(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).lower() in {"1", "true", "yes"}


@dataclass
class Settings:
    weights: str = ""                       # path to the fine-tuned yolo11x .pt (empty -> service starts but is not ready)
    model_version: str = "unversioned"
    dataset_version: str | None = None
    labels_path: Path = ROOT / "shared" / "labels.json"
    manifest_path: Path = ROOT / "shared" / "model_manifest.json"
    model_card_path: str = ""               # optional JSON: {"thresholds": {"tick": 0.45, ...}, "metrics": {...}}
    default_threshold: float = 0.35
    imgsz: int = 640
    device: str = "auto"                    # auto | cpu | 0
    data_dir: Path = ROOT / "vision_data"
    database_url: str = ""                  # default: sqlite in data_dir; use postgresql+psycopg2://... in production
    public_base_url: str = "http://localhost:8000"
    jwt_secret: str = ""                    # shared HS256 secret used by all Agricoders APIs
    auth_disabled: bool = False             # development only; never set in production
    max_upload_bytes: int = 10_000_000
    max_side: int = 1280
    rate_limit_per_min: int = 30
    allow_unmatched_classes: bool = False   # dev only: serve a model whose class names are not in labels.json
    privileged_roles: tuple[str, ...] = ("vet", "admin")
    cors_origins: list[str] = field(default_factory=lambda: ["http://127.0.0.1:8080", "http://localhost:8080"])

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        s = cls(weights=e("VISION_WEIGHTS", ""), model_version=e("VISION_MODEL_VERSION", "unversioned"),
                dataset_version=e("VISION_DATASET_VERSION"), model_card_path=e("VISION_MODEL_CARD", ""),
                default_threshold=float(e("VISION_CONF", "0.35")), imgsz=int(e("VISION_IMGSZ", "640")),
                device=e("VISION_DEVICE", "auto"), data_dir=Path(e("VISION_DATA_DIR", str(ROOT / "vision_data"))),
                database_url=e("DATABASE_URL", ""), public_base_url=e("PUBLIC_BASE_URL", "http://localhost:8000"),
                jwt_secret=e("AGRICODERS_JWT_SECRET", ""), auth_disabled=_b("VISION_AUTH_DISABLED"),
                rate_limit_per_min=int(e("VISION_RATE_LIMIT", "30")),
                allow_unmatched_classes=_b("VISION_ALLOW_UNMATCHED_CLASSES"))
        if e("CORS_ORIGINS"):
            s.cors_origins = [o.strip() for o in e("CORS_ORIGINS").split(",") if o.strip()]
        if e("LABELS_PATH"):
            s.labels_path = Path(e("LABELS_PATH"))
        if e("MANIFEST_PATH"):
            s.manifest_path = Path(e("MANIFEST_PATH"))
        return s

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'vision.db').as_posix()}"

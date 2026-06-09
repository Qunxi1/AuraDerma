from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class AppConfig:
    model_api_base: str
    model_api_key: str
    default_model: str
    context_window: int               # 模型上下文窗口（token 数）
    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection_products: str
    qdrant_collection_memory: str
    qdrant_collection_docs: str
    web_search_enabled: bool
    data_dir: Path
    skills_dir: Path
    local_env_path: Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is not None and raw.strip().isdigit():
        return int(raw.strip())
    return default

def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    data_dir = Path(os.getenv("AURADERMA_DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = Path(os.getenv("AURADERMA_SKILLS_DIR", "./skills")).resolve()
    skills_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        model_api_base=os.getenv("AURADERMA_MODEL_BASE", "https://api.deepseek.com"),
        model_api_key=os.getenv("AURADERMA_MODEL_API_KEY", ""),
        default_model=os.getenv("AURADERMA_DEFAULT_MODEL", "deepseek-v4-flash"),
        context_window=_env_int("AURADERMA_CONTEXT_WINDOW", 1000000),
        qdrant_url=os.getenv("AURADERMA_QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("AURADERMA_QDRANT_API_KEY") or None,
        qdrant_collection_products=os.getenv("AURADERMA_QDRANT_PRODUCTS", "AuraDerma_products"),
        qdrant_collection_memory=os.getenv("AURADERMA_QDRANT_MEMORY", "AuraDerma_memory"),
        qdrant_collection_docs=os.getenv("AURADERMA_QDRANT_DOCS", "AuraDerma_docs"),
        web_search_enabled=_env_bool("AURADERMA_WEB_SEARCH_ENABLED", "true"),
        data_dir=data_dir,
        skills_dir=skills_dir,
        local_env_path=Path(".env").resolve(),
    )

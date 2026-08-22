"""
Central configuration system for Phase 4 RAG.

Loads from rag/config/default.yaml, with support for:
- RAG_CONFIG_PATH environment variable to override the config file path
- RAG_<SECTION>__<KEY> environment variables for individual overrides
  e.g. RAG_EMBEDDING__MODEL_NAME="openai/text-embedding-3-small"

Usage:
    from rag.config.settings import get_settings
    cfg = get_settings()
    print(cfg.chunking.max_tokens)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeneralConfig:
    project_name: str = "rag-docs-assistant"
    log_level: str = "INFO"
    data_dir: str = "./rag/data"


@dataclass
class ScraperOutputConfig:
    directory: str = "./scraped_output"


@dataclass
class ChunkingConfig:
    max_tokens: int = 512
    min_tokens: int = 100
    overlap_tokens: int = 75
    encoding_name: str = "cl100k_base"


@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"
    dimensions: Optional[int] = None


@dataclass
class VectorStoreConfig:
    provider: str = "chroma"
    persist_dir: str = "./rag/data/indexes/chroma"
    collection_name: str = "docs"


@dataclass
class BM25Config:
    index_path: str = "./rag/data/indexes/bm25.pkl"


@dataclass
class RetrievalConfig:
    top_k: int = 5
    candidate_k: int = 20
    vector_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60
    use_query_transform: bool = False


@dataclass
class RerankerConfig:
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-large"


@dataclass
class GenerationConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    max_context_tokens: int = 3000


@dataclass
class ServingConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cache_ttl_seconds: int = 3600


@dataclass
class LLMSettings:
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 300


# ---------------------------------------------------------------------------
# Root Settings dataclass
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    """Root configuration object. Access sections as attributes."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    scraper_output: ScraperOutputConfig = field(default_factory=ScraperOutputConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    llm: LLMSettings = field(default_factory=LLMSettings)

    def __repr__(self) -> str:  # pragma: no cover
        lines = ["Settings("]
        for section_name in [
            "general", "scraper_output", "chunking", "embedding",
            "vector_store", "bm25", "retrieval", "reranker",
            "generation", "serving", "llm",
        ]:
            section = getattr(self, section_name)
            lines.append(f"  {section_name}={section!r},")
        lines.append(")")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section loaders
# ---------------------------------------------------------------------------

_SECTION_MAP: dict[str, type] = {
    "general": GeneralConfig,
    "scraper_output": ScraperOutputConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "vector_store": VectorStoreConfig,
    "bm25": BM25Config,
    "retrieval": RetrievalConfig,
    "reranker": RerankerConfig,
    "generation": GenerationConfig,
    "serving": ServingConfig,
    "llm": LLMSettings,
}


def _coerce(value: str, target_type: type) -> object:
    """Coerce a string value to the given Python type."""
    if target_type is bool:
        return value.lower() in ("1", "true", "yes", "on")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is Optional[int]:
        return None if value.lower() in ("null", "none", "") else int(value)
    return value  # str passthrough


def _load_yaml(path: Path) -> dict:
    """Load YAML file and return raw dict. Returns {} on failure."""
    if not path.exists():
        logger.warning("Config file not found: %s — using defaults.", path)
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _build_section(section_cls: type, yaml_data: dict, env_prefix: str) -> object:
    """
    Construct a section dataclass from YAML values, then apply any
    RAG_<SECTION>__<KEY> environment variable overrides.
    """
    import dataclasses

    fields = {f.name: f for f in dataclasses.fields(section_cls)}
    kwargs: dict = {}

    # Pull from YAML first
    for name, dc_field in fields.items():
        if name in yaml_data:
            kwargs[name] = yaml_data[name]

    # Apply env-var overrides: RAG_CHUNKING__MAX_TOKENS=256
    for name, dc_field in fields.items():
        env_key = f"{env_prefix}__{name.upper()}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                kwargs[name] = _coerce(env_val, dc_field.type)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    "Could not coerce env var %s=%r: %s", env_key, env_val, exc
                )

    return section_cls(**kwargs)


def _load_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Build a Settings instance.

    Priority (highest to lowest):
    1. RAG_<SECTION>__<KEY> environment variables
    2. YAML config file specified by RAG_CONFIG_PATH (or the default path)
    3. Dataclass defaults
    """
    # Resolve config file path
    if config_path is None:
        env_path = os.environ.get("RAG_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        else:
            config_path = Path(__file__).parent / "default.yaml"

    raw = _load_yaml(config_path)

    kwargs: dict = {}
    for section_name, section_cls in _SECTION_MAP.items():
        section_data = raw.get(section_name, {}) or {}
        env_prefix = f"RAG_{section_name.upper()}"
        kwargs[section_name] = _build_section(section_cls, section_data, env_prefix)

    settings = Settings(**kwargs)

    # Configure logging based on loaded settings
    logging.basicConfig(level=getattr(logging, settings.general.log_level, logging.INFO))

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the singleton Settings instance.

    Cached after first call. To reload (e.g. in tests), call:
        get_settings.cache_clear()
    """
    return _load_settings()

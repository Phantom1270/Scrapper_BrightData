"""
Settings and Ollama model list endpoints.
"""
from fastapi import APIRouter
import os
import yaml
from pydantic import BaseModel, Field
from typing import Optional
from rag.config.settings import get_settings


# ── Request model for partial settings update ─────────────────────

class SettingsPatch(BaseModel):
    llm_model: Optional[str] = Field(default=None, description="Ollama model name")
    llm_base_url: Optional[str] = Field(default=None, description="Ollama base URL")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model name")
    reranker_model: Optional[str] = Field(default=None, description="Reranker model name")
    reranker_enabled: Optional[bool] = Field(default=None, description="Enable/disable reranker")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Default top_k")
    use_query_transform: Optional[bool] = Field(default=None, description="Enable/disable query transform")
    use_reranking: Optional[bool] = Field(default=None, description="Enable/disable reranking")
router = APIRouter()


@router.get("/settings")
def get_current_settings():
    """
    Returns the current runtime configuration.
    Frontend uses this to populate the settings panel.
    """
    settings = get_settings()

    return {
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "base_url": settings.llm.base_url,
        },
        "embedding": {
            "model_name": settings.embedding.model_name,
        },
        "reranker": {
            "enabled": settings.reranker.enabled,
            "model_name": settings.reranker.model_name,
        },
        "retrieval": {
            "top_k": settings.retrieval.top_k,
            "use_query_transform": settings.retrieval.use_query_transform,
            "use_reranking": settings.reranker.enabled,
        },
    }

@router.get("/ollama/models")
def list_ollama_models():
    """
    Queries the Ollama API to list installed models.
    Returns model names for the frontend dropdown.
    """
    settings = get_settings()
    base_url = settings.llm.base_url  # e.g. http://localhost:11434

    try:
        import requests as http_requests
        resp = http_requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = [m["name"] for m in data.get("models", [])]
        return {"models": sorted(models)}
    except ImportError:
        # Fallback to urllib if requests is not installed
        try:
            import urllib.request
            import json

            req = urllib.request.Request(f"{base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                models = [m["name"] for m in data.get("models", [])]
                return {"models": sorted(models)}
        except Exception as e:
            return {"models": [], "error": f"Ollama unreachable: {str(e)}"}
    except Exception as e:
        return {"models": [], "error": f"Ollama unreachable: {str(e)}"}

@router.put("/settings")
def update_settings(patch: SettingsPatch):
    """
    Updates settings in-memory AND writes changes to default.yaml.
    Engine singletons are reset so changes take effect on the next query.
    """
    from rag.config.settings import get_settings
    from rag.serving import dependencies

    settings = get_settings()
    changes = {}

    # ── Apply in-memory changes ──────────────────────────────────

    if patch.llm_model is not None:
        settings.llm.model = patch.llm_model
        changes["llm.model"] = patch.llm_model

    if patch.llm_base_url is not None:
        settings.llm.base_url = patch.llm_base_url
        changes["llm.base_url"] = patch.llm_base_url

    if patch.embedding_model is not None:
        settings.embedding.model_name = patch.embedding_model
        changes["embedding.model_name"] = patch.embedding_model

    if patch.reranker_model is not None:
        settings.reranker.model_name = patch.reranker_model
        changes["reranker.model_name"] = patch.reranker_model

    if patch.reranker_enabled is not None:
        settings.reranker.enabled = patch.reranker_enabled
        changes["reranker.enabled"] = patch.reranker_enabled

    if patch.top_k is not None:
        settings.retrieval.top_k = patch.top_k
        changes["retrieval.top_k"] = patch.top_k

    if patch.use_query_transform is not None:
        settings.retrieval.use_query_transform = patch.use_query_transform
        changes["retrieval.use_query_transform"] = patch.use_query_transform

    if patch.use_reranking is not None:
        settings.retrieval.use_reranking = patch.use_reranking
        changes["retrieval.use_reranking"] = patch.use_reranking

    if not changes:
        return {"status": "no_changes", "message": "No settings were updated."}

    # ── Write to default.yaml ────────────────────────────────────

    yaml_path = _find_default_yaml()
    yaml_write_error = None

    if yaml_path:
        try:
            _write_yaml(yaml_path, settings)
        except Exception as e:
            yaml_write_error = str(e)
    else:
        yaml_write_error = "Could not locate default.yaml"

    # ── Reset engine singletons ──────────────────────────────────

    dependencies._generation_engine = None
    dependencies._retrieval_engine = None

    if not yaml_write_error:
        try:
            get_settings.cache_clear()
        except Exception:
            pass

    # ── Build response ───────────────────────────────────────────

    response = {
        "status": "updated",
        "changes": changes,
        "note": "Changes take effect on the next query. First query may be slow while models reload.",
    }

    if "embedding.model_name" in changes:
        response["warning"] = (
            "Embedding model changed. All documents must be re-indexed "
            "for this change to take effect. Use POST /api/v1/index to re-index."
        )

    if yaml_write_error:
        response["yaml_status"] = f"Config updated in memory only. YAML write failed: {yaml_write_error}"

    return response


# ── YAML helpers ──────────────────────────────────────────────────

def _find_default_yaml() -> Optional[str]:
    """
    Locates default.yaml by walking up from this file's directory.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        candidate = os.path.join(current, "rag", "config", "default.yaml")
        if os.path.isfile(candidate):
            return candidate
        candidate = os.path.join(current, "config", "default.yaml")
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for fallback in [
        "rag/config/default.yaml",
        "../config/default.yaml",
        "../../config/default.yaml",
        "config/default.yaml",
    ]:
        if os.path.isfile(fallback):
            return os.path.abspath(fallback)

    return None


def _write_yaml(yaml_path: str, settings) -> None:
    """
    Reads the existing YAML, updates all settings fields,
    and writes it back.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("llm", {})
    data.setdefault("embedding", {})
    data.setdefault("reranker", {})
    data.setdefault("retrieval", {})

    data["llm"]["provider"] = settings.llm.provider
    data["llm"]["model"] = settings.llm.model
    data["llm"]["base_url"] = settings.llm.base_url

    data["embedding"]["model_name"] = settings.embedding.model_name

    data["reranker"]["enabled"] = settings.reranker.enabled
    data["reranker"]["model_name"] = settings.reranker.model_name

    data["retrieval"]["top_k"] = settings.retrieval.top_k
    data["retrieval"]["use_query_transform"] = settings.retrieval.use_query_transform
    data["retrieval"]["use_reranking"] = settings.retrieval.use_reranking

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

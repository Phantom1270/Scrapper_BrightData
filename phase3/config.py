"""
Configuration settings for Phase 3 (CLI Orchestration).
"""
import os

# ─── CLI Settings ────────────────────────────────────────────
# The prompt used when initially creating a scraper for a template
DEFAULT_CREATE_PROMPT: str = (
    "Extract the page title, main content sections, code snippets, "
    "and any parameter tables into structured JSON."
)

# The prompt used when a scraper fails and needs to be healed
DEFAULT_HEAL_PROMPT: str = (
    "The extraction failed. The HTML structure likely changed. "
    "Please fix the extraction selectors to grab the title, content, "
    "code snippets, and parameters."
)

# ─── Orchestration Settings ──────────────────────────────────
MAX_RETRIES: int = 3
MAX_HEAL_ATTEMPTS: int = 2

# ─── Validation Settings ─────────────────────────────────────
GARBAGE_STRINGS: list[str] = ["undefined", "n/a", "page not found", ""]

SCHEMA_FIELDS: dict = {
    "page_title": {"min_length": 3},
    "function_signature": {"min_length": 5},
    "description": {"min_length": 10},
    "parameters": {"min_length": 0}  # Optional field
}

# ─── LLM Validation Settings ─────────────────────────────────
LLM_VALIDATION_ENABLED: bool = os.getenv("LLM_VALIDATION_ENABLED", "false").lower() == "true"
LLM_API_KEY: str | None = os.getenv("GEMINI_API_KEY", None)
LLM_VALIDATION_TIMEOUT: int = 5  # seconds


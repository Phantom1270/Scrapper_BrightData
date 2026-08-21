"""
Data models for Phase 3.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Any
from datetime import datetime


# ─── Phase 2 Models (Input to Phase 3) ─────────────────────────

class TemplatePattern(BaseModel):
    template_id: str
    name: Optional[str] = None
    pattern: str
    fingerprint: str
    member_count: int
    versions_covered: list[str] = Field(default_factory=list)
    scope: str = "internal"
    example_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0

class CoverageReport(BaseModel):
    total_internal_urls: int
    covered_urls: int
    uncovered_urls: int
    coverage_percent: float
    template_count: int
    avg_urls_per_template: float

class UncoveredURL(BaseModel):
    url: str
    reason: str
    recommendation: str = "self_heal"

class Phase2Output(BaseModel):
    crawl_id: str
    root_domain: str
    generator_detected: Optional[str] = None
    generator_confidence: float = 0.0
    summary: CoverageReport = Field(default_factory=dict)
    templates: list[TemplatePattern] = Field(default_factory=list)
    uncovered_urls: list[UncoveredURL] = Field(default_factory=list)
    external_domains: list[dict] = Field(default_factory=list)
    template_map: dict[str, list[str]] = Field(default_factory=dict)


# ─── Enums ────────────────────────────────────────────────────

class FieldImportance(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class FieldType(str, Enum):
    TEXT = "text"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    URL = "url"
    IMAGE = "image"
    NESTED = "nested"


class RecordStatus(str, Enum):
    PENDING = "pending"
    FETCHED = "fetched"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    HEALED = "healed"
    FAILED = "failed"


class PageType(str, Enum):
    API_REFERENCE = "api_reference"
    EXAMPLES = "examples"
    PROSE_DOCS = "prose_docs"
    CHANGELOG = "changelog"
    INDEX_PAGE = "index_page"
    LANDING_PAGE = "landing_page"
    DOWNLOAD_PAGE = "download_page"
    TEAM_PAGE = "team_page"
    EVENT_PAGE = "event_page"
    OTHER = "other"


# ─── Schema models ────────────────────────────────────────────

class FieldSchema(BaseModel):
    """Definition of one field to extract from a page."""
    name: str
    description: str = ""
    field_type: FieldType = FieldType.TEXT
    importance: FieldImportance = FieldImportance.OPTIONAL
    css_selector: Optional[str] = None
    fallback_selectors: list[str] = Field(default_factory=list)
    extraction_hint: str = ""
    example_value: Optional[str] = None


class ValidationSchema(BaseModel):
    """What we expect to find on pages matching a template."""
    template_id: str
    template_pattern: str
    page_type: PageType = PageType.OTHER
    fields: list[FieldSchema] = Field(default_factory=list)
    sample_urls: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ─── Scraping models ─────────────────────────────────────────

class FetchedPage(BaseModel):
    """Raw HTML fetched from Bright Data for one URL."""
    url: str
    template_id: str
    html: Optional[str] = None
    http_status: Optional[int] = None
    fetch_success: bool = False
    fetch_error: Optional[str] = None
    bytes_size: int = 0


class ExtractedRecord(BaseModel):
    """Structured data extracted from one page."""
    url: str
    template_id: str
    status: RecordStatus = RecordStatus.EXTRACTED
    data: dict[str, Any] = Field(default_factory=dict)
    fields_found: int = 0
    fields_missing: int = 0
    fields_empty: int = 0


class ValidationResult(BaseModel):
    """Result of validating one extracted record."""
    url: str
    template_id: str
    passed: bool = False
    missing_required: list[str] = Field(default_factory=list)
    empty_required: list[str] = Field(default_factory=list)
    missing_optional: list[str] = Field(default_factory=list)
    score: float = 0.0  # fraction of required fields present and non-empty


class HealedRecord(BaseModel):
    """A record that was fixed by LLM self-heal."""
    url: str
    template_id: str
    original_data: dict[str, Any] = Field(default_factory=dict)
    healed_data: dict[str, Any] = Field(default_factory=dict)
    fields_healed: list[str] = Field(default_factory=list)
    heal_success: bool = False


# ─── Output models ────────────────────────────────────────────

class TemplateStats(BaseModel):
    """Stats for one template's scraping results."""
    template_id: str
    template_pattern: str
    page_type: str = ""
    total_urls: int = 0
    fetched: int = 0
    extracted: int = 0
    validated_passed: int = 0
    validated_failed: int = 0
    self_healed: int = 0
    permanently_failed: int = 0


class Phase3Output(BaseModel):
    """Complete Phase 3 output."""
    crawl_id: str
    root_domain: str
    generator_detected: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    # Credits and cost tracking
    bright_data_credits_used: int = 0
    llm_calls_made: int = 0
    llm_tokens_used: int = 0

    # Summary stats
    total_urls: int = 0
    total_fetched: int = 0
    total_extracted: int = 0
    total_validated_passed: int = 0
    total_self_healed: int = 0
    total_permanently_failed: int = 0
    final_success_rate: float = 0.0

    # Per-template stats
    template_stats: list[TemplateStats] = Field(default_factory=list)

    # The actual scraped data: template_id → list of records
    results: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    # Schemas used
    schemas: dict[str, ValidationSchema] = Field(default_factory=dict)

    # URLs that could not be scraped
    failed_urls: list[dict[str, str]] = Field(default_factory=list)

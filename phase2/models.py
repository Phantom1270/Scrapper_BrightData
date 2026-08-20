"""
Data models for Phase 2.
All data flowing through the pipeline uses these types.
Every field has a purpose. No unused fields.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


# ─── Enums ────────────────────────────────────────────────────

class URLClassification(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class SegmentType(str, Enum):
    """Lexical type of a single URL path segment."""
    LITERAL = "literal"              # "modules", "generated", "user_guide"
    SLUG = "slug"                    # "linear-model", "random-forest-classifier"
    CAMEL_CASE = "camel_case"        # "LogisticRegression", "RandomForestClassifier"
    DOTTED_PATH = "dotted_path"      # "sklearn.linear_model", "numpy.core"
    INTEGER = "integer"              # "123", "42"
    FLOAT = "float"                  # "3.14"
    VERSION = "version"              # "v2.1", "0.24", "3.1.2"
    DATE = "date"                    # "2024-01-15"
    UUID = "uuid"                    # "550e8400-e29b-41d4-a716-446655440000"
    HASH = "hash"                    # "a1b2c3d4" (hex string, 8+ chars)
    FILENAME = "filename"            # "LogisticRegression.html", "plot_example.png"
    EXTENSION = "extension"          # ".html", ".rst", ".md"
    UNKNOWN = "unknown"


class ExternalDomainType(str, Enum):
    """Classification of external domains."""
    DOCUMENTATION_CROSSREF = "documentation_crossref"
    PROJECT_HOMEPAGE = "project_homepage"
    PACKAGE_INDEX = "package_index"
    CODE_HOSTING = "code_hosting"
    UNKNOWN = "unknown"


# ─── Input models (Phase 1 → Phase 2) ────────────────────────

class DiscoveredURL(BaseModel):
    """A single URL as discovered by Phase 1."""
    url: str
    source_url: str = Field(description="The page where this link was found")
    depth: int = Field(description="Crawl depth from the root that discovered it")
    link_text: str = Field(default="", description="Visible text of the anchor tag")
    http_status: Optional[int] = Field(default=None, description="HTTP status if fetched")


class Phase1Output(BaseModel):
    """The complete input that Phase 2 receives from Phase 1."""
    crawl_id: str = Field(description="Unique identifier for this crawl run")
    root_domain: str = Field(description="The domain being crawled, e.g. scikit-learn.org")
    root_urls: list[str] = Field(description="Entry point URLs that Phase 1 started from")

    internal_urls: list[DiscoveredURL] = Field(
        default_factory=list,
        description="URLs belonging to root_domain"
    )
    external_urls: list[DiscoveredURL] = Field(
        default_factory=list,
        description="URLs pointing to other domains"
    )

    signals: dict = Field(
        default_factory=dict,
        description="Hints from Phase 1, e.g. has_objects_inv=True"
    )


# ─── Internal processing models ──────────────────────────────

class ParsedSegment(BaseModel):
    """One segment of a parsed URL path."""
    raw: str                           # Original text: "LogisticRegression.html"
    position: int                      # 0-indexed position in path
    lexical_type: SegmentType          # Determined by segment classifier
    is_static: Optional[bool] = None   # Filled in during grouping (Step 2.5)
    frequency: Optional[float] = None  # How often this exact value appears at this position


class ParsedURL(BaseModel):
    """A fully parsed and classified URL."""
    original_url: str
    canonical_url: str
    domain: str
    scheme: str
    path: str                          # Canonical path
    segments: list[ParsedSegment]
    query: Optional[str] = None
    fragment: Optional[str] = None
    classification: URLClassification
    is_asset: bool
    version: Optional[str] = None      # Extracted version prefix, e.g. "stable"
    version_free_path: Optional[str] = None  # Path without version prefix
    source_url: str = ""
    depth: int = 0
    link_text: str = ""


# ─── Output models (Phase 2 → Phase 3) ───────────────────────

class TemplatePattern(BaseModel):
    """A discovered URL pattern that maps to one scraper template."""
    template_id: str                   # e.g. "tpl_001"
    name: Optional[str] = None         # Human-readable name, e.g. "api_reference"
    pattern: str                       # e.g. "/modules/generated/<dotted_path>.<filename>"
    fingerprint: str                   # e.g. "LIT/LIT/LIT/DOTTED_PATH.FILENAME"
    member_count: int                  # How many URLs match this pattern
    versions_covered: list[str] = Field(default_factory=list)
    scope: str = "internal"            # "internal" or "external"
    example_urls: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExternalDomain(BaseModel):
    """Summary of an external domain encountered during crawling."""
    domain: str
    url_count: int
    classification: ExternalDomainType
    observed_paths: list[str] = Field(default_factory=list)
    note: str = ""


class CoverageReport(BaseModel):
    """How well the templates cover the discovered URLs."""
    total_internal_urls: int
    covered_urls: int
    uncovered_urls: int
    coverage_percent: float
    template_count: int
    avg_urls_per_template: float


class UncoveredURL(BaseModel):
    """A URL that no template matches. Sent to Phase 3 self-heal."""
    url: str
    reason: str
    recommendation: str = "self_heal"


class Phase2Output(BaseModel):
    """The complete output that Phase 2 sends to Phase 3."""
    crawl_id: str
    root_domain: str
    generator_detected: Optional[str] = None
    generator_confidence: float = 0.0

    summary: CoverageReport
    templates: list[TemplatePattern] = Field(default_factory=list)
    uncovered_urls: list[UncoveredURL] = Field(default_factory=list)
    external_domains: list[ExternalDomain] = Field(default_factory=list)

    # Phase 3 reads this directly: template_id → list of URLs to scrape
    template_map: dict[str, list[str]] = Field(default_factory=dict)

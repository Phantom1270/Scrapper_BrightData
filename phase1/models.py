"""
Data models for Phase 1.
Mirrors what Phase 2 expects to receive.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

from config import CRAWL_ID_PREFIX


class URLClassification(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class CrawlStatus(str, Enum):
    """Status of a URL fetch attempt."""
    SUCCESS = "success"
    REDIRECT = "redirect"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    SKIPPED_EXTENSION = "skipped_extension"
    SKIPPED_CONTENT_TYPE = "skipped_content_type"
    SKIPPED_DEPTH = "skipped_depth"
    SKIPPED_LIMIT = "skipped_limit"
    ERROR = "error"


class DiscoveredURL(BaseModel):
    """
    A single discovered URL.
    This is the exact format Phase 2 expects.
    """
    url: str = Field(description="The discovered URL")
    source_url: str = Field(description="Page where this link was found")
    depth: int = Field(description="Crawl depth from root")
    link_text: str = Field(default="", description="Anchor tag text")
    http_status: Optional[int] = Field(default=None, description="HTTP response code if fetched")


class CrawlQueueItem(BaseModel):
    """
    An item in the crawl frontier/queue.
    Internal tracking structure — not part of output.
    """
    url: str
    source_url: str
    depth: int
    link_text: str = ""


class FetchResult(BaseModel):
    """
    Result of fetching a single URL.
    Internal tracking structure — not part of output.
    """
    url: str
    final_url: str                  # After redirects
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    html: Optional[str] = None
    error: Optional[str] = None
    crawl_status: CrawlStatus = CrawlStatus.SUCCESS
    redirect_chain: list[str] = Field(default_factory=list)


class CrawlStats(BaseModel):
    """Statistics about the crawl run."""
    pages_fetched: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    internal_urls_discovered: int = 0
    external_urls_discovered: int = 0
    unique_internal_domains: int = 0
    unique_external_domains: int = 0
    max_depth_reached: int = 0
    redirects_followed: int = 0
    total_bytes_downloaded: int = 0


class CrawlResult(BaseModel):
    """
    Complete result of one crawl run.
    This gets serialized to the output JSON.
    """
    crawl_id: str
    root_domain: str
    root_urls: list[str]
    started_at: str
    finished_at: str
    duration_seconds: float

    signals: dict = Field(default_factory=dict)

    internal_urls: list[DiscoveredURL] = Field(default_factory=list)
    external_urls: list[DiscoveredURL] = Field(default_factory=list)

    stats: CrawlStats = Field(default_factory=CrawlStats)


def make_crawl_id() -> str:
    """
    Generate a unique crawl ID.
    Format: crawl_YYYY_MM_DD_HHMMSS

    Example: crawl_2026_08_20_143022
    """
    now = datetime.now()
    return f"{CRAWL_ID_PREFIX}_{now.strftime('%Y_%m_%d_%H%M%S')}"

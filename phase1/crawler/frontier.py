"""
URL frontier — the BFS queue.
Manages which URLs to visit next, tracks what's been visited,
enforces depth and count limits.
"""

from collections import deque, defaultdict
from typing import Optional
from urllib.parse import urlparse
from models import CrawlQueueItem, DiscoveredURL, URLClassification
from config import (
    MAX_INTERNAL_DEPTH, MAX_INTERNAL_URLS,
    MAX_EXTERNAL_URLS_PER_DOMAIN, MAX_EXTERNAL_URLS_TOTAL,
)


import posixpath

def _normalize_url(url: str) -> str:
    """
    Canonical URL normalization for deduplication.
    - Lowercase scheme + host
    - Resolve '.' and '..' dot segments in path (fixes /stable/./install.html)
    - Collapse double slashes in path
    - Strip trailing slash (unless root path)
    - Strip fragment (never relevant for dedup)
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path

    # Resolve dot segments: posixpath.normpath collapses . and ..
    # It also strips trailing slash (so root "/" stays as "/")
    if path:
        path = posixpath.normpath(path)
        # normpath strips trailing /, but we need to preserve root
        # and we never want a trailing slash on non-root paths (already handled)

    # Rebuild without fragment
    query = parsed.query
    normalized = f"{scheme}://{netloc}{path}"
    if query:
        normalized += f"?{query}"
    return normalized



class Frontier:
    """
    BFS URL frontier for web crawling.

    Maintains:
    - A queue of URLs to visit (BFS order)
    - A set of visited URLs (no duplicates)
    - A set of discovered internal URLs
    - A set of discovered external URLs
    - Counters for limits enforcement
    """

    def __init__(self, root_domain: str):
        self.root_domain: str = root_domain
        self.queue: deque[CrawlQueueItem] = deque()
        self.visited: set[str] = set()          # normalized canonical URLs already fetched or queued
        self.internal_urls: list[DiscoveredURL] = []
        self.external_urls: list[DiscoveredURL] = []
        self.internal_domain_urls: set[str] = set()   # all internal URLs seen (dedup)
        self.external_domain_counts: dict[str, int] = defaultdict(int)
        self.external_total: int = 0
        self.max_depth_reached: int = 0

    def add_seed(self, url: str) -> None:
        """
        Add the initial seed URL to the queue.
        Depth 0, source is itself, link_text is empty.
        """
        item = CrawlQueueItem(
            url=url,
            source_url=url,
            depth=0,
            link_text="",
        )
        normalized = _normalize_url(url)
        self.visited.add(normalized)
        self.queue.append(item)

    def enqueue(self, item: CrawlQueueItem) -> bool:
        """
        Add a URL to the queue if it should be visited.

        Returns True if added, False if rejected.

        Rejection reasons:
        1. URL already in self.visited (exact string match after normalization)
        2. URL is empty or malformed
        """
        if not item.url:
            return False

        normalized = _normalize_url(item.url)

        if not normalized or normalized == "://":
            return False

        if normalized in self.visited:
            return False

        self.visited.add(normalized)
        self.queue.append(item)
        return True

    def should_crawl(self, item: CrawlQueueItem) -> tuple[bool, str]:
        """
        Determine if a URL should actually be fetched.

        Returns: (should_fetch: bool, reason: str)
        """
        parsed = urlparse(item.url)
        host = parsed.netloc.lower().removeprefix("www.")
        root = self.root_domain.lower().removeprefix("www.")

        # Check if external
        is_internal = (host == root) or host.endswith("." + root)

        if not is_internal:
            return False, "external_not_crawled"

        # Depth limit
        if item.depth > MAX_INTERNAL_DEPTH:
            return False, "max_depth"

        # Internal URL count limit
        if len(self.internal_urls) >= MAX_INTERNAL_URLS:
            return False, "max_internal_urls"

        return True, "ok"

    def record_internal(self, item: CrawlQueueItem, http_status: Optional[int] = None) -> None:
        """
        Record an internal URL as discovered.
        Add to self.internal_urls as a DiscoveredURL.
        Add to self.visited set.
        Update max_depth_reached if this URL's depth is higher.
        """
        normalized = _normalize_url(item.url)
        if normalized in self.internal_domain_urls:
            return

        self.internal_domain_urls.add(normalized)
        self.internal_urls.append(DiscoveredURL(
            url=item.url,
            source_url=item.source_url,
            depth=item.depth,
            link_text=item.link_text,
            http_status=http_status,
        ))

        if item.depth > self.max_depth_reached:
            self.max_depth_reached = item.depth

    def record_external(self, item: CrawlQueueItem) -> None:
        """
        Record an external URL as discovered.

        Checks:
        1. If external_total >= MAX_EXTERNAL_URLS_TOTAL → skip
        2. If this domain's count >= MAX_EXTERNAL_URLS_PER_DOMAIN → skip
        3. Otherwise → add to self.external_urls as DiscoveredURL
        """
        if self.external_total >= MAX_EXTERNAL_URLS_TOTAL:
            return

        parsed = urlparse(item.url)
        domain = parsed.netloc.lower()

        if self.external_domain_counts[domain] >= MAX_EXTERNAL_URLS_PER_DOMAIN:
            return

        self.external_urls.append(DiscoveredURL(
            url=item.url,
            source_url=item.source_url,
            depth=item.depth,
            link_text=item.link_text,
        ))
        self.external_domain_counts[domain] += 1
        self.external_total += 1

    def get_next(self) -> Optional[CrawlQueueItem]:
        """
        Pop the next URL from the queue (BFS order).
        Returns None if queue is empty.
        """
        if self.queue:
            return self.queue.popleft()
        return None

    def get_stats(self) -> dict:
        """
        Return current frontier statistics.
        """
        return {
            "queue_size": len(self.queue),
            "visited_count": len(self.visited),
            "internal_discovered": len(self.internal_urls),
            "external_discovered": len(self.external_urls),
            "max_depth_reached": self.max_depth_reached,
            "external_domains": dict(self.external_domain_counts),
        }

    @property
    def is_empty(self) -> bool:
        """True if the queue has no more items to process."""
        return len(self.queue) == 0

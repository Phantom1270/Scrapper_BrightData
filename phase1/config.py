"""
All constants for Phase 1 crawler.
No magic numbers anywhere else.
"""

# ─── Hardcoded input (for now) ──────────────────────────────
# Change this URL to test different sites.
INPUT_URL: str = "https://scikit-learn.org/stable/"

# ─── Crawl limits ───────────────────────────────────────────
# Maximum depth to follow internal links.
# Depth 0 = seed URL only.
# Depth 1 = links found on the seed page.
# Depth N = links found on depth-(N-1) pages.
MAX_INTERNAL_DEPTH: int = 3

# Maximum total internal URLs to discover.
# Safety net — stops the crawl if something generates an unexpectedly
# large number of pages (e.g. infinite pagination or query-string explosion).
MAX_INTERNAL_URLS: int = 10000

# Maximum external URLs to record per domain.
# We don't want 10,000 github.com links.
MAX_EXTERNAL_URLS_PER_DOMAIN: int = 50

# Maximum total external URLs to record.
MAX_EXTERNAL_URLS_TOTAL: int = 500

# Number of concurrent fetch workers (threads).
# Each worker fetches one page at a time, each respecting POLITENESS_DELAY
# independently.  With 5 workers + 0.1s delay ≈ 50 pages/second aggregate.
MAX_WORKERS: int = 5

# ─── HTTP settings ──────────────────────────────────────────
# Request timeout in seconds.
REQUEST_TIMEOUT: int = 30

# Delay between requests to the SAME domain (be polite).
# In seconds. Set to 0 for speed during development.
POLITENESS_DELAY: float = 0.1

# Maximum retries on failure.
MAX_RETRIES: int = 2

# Retry delay multiplier (exponential backoff).
RETRY_BACKOFF: float = 2.0

# User-Agent string.
USER_AGENT: str = (
    "Phase1Crawler/1.0 "
    "(+https://github.com/yourproject; "
    "documentation scraper; collecting URL structure)"
)

# HTTP headers to send with every request.
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
}

# ─── Content filtering ─────────────────────────────────────
# Only crawl pages with these content types.
CRAWLABLE_CONTENT_TYPES: set[str] = {
    "text/html",
    "application/xhtml+xml",
}

# Do not follow links with these file extensions.
# These are assets, downloads, etc.
SKIP_EXTENSIONS: set[str] = {
    ".css", ".js", ".mjs",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
    ".woff", ".woff2", ".ttf", ".eot",
    ".mp4", ".mp3", ".wav", ".ogg",
    ".zip", ".tar", ".gz", ".bz2", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".xml",  # sitemap.xml is handled separately
    ".map",  # source maps
    ".exe", ".dmg", ".whl", ".egg",
    ".py", ".pyc", ".pyd",
    ".rst", ".md",  # raw source files — we want the rendered HTML
}

# ─── URLs to always check (for signal detection) ───────────
# Phase 1 proactively checks these URLs to detect the doc generator.
# These are fetched once, not crawled further.
SIGNAL_CHECK_URLS: dict[str, str] = {
    "objects_inv": "/objects.inv",
    "sitemap_xml": "/sitemap.xml",
    "robots_txt": "/robots.txt",
    "search_index": "/search/search_index.json",
}

# ─── Output ─────────────────────────────────────────────────
# Default output file name.
DEFAULT_OUTPUT_FILE: str = "phase1_output.json"

# Crawl ID prefix.
CRAWL_ID_PREFIX: str = "crawl"

"""
HTTP fetcher.
Makes the actual HTTP requests.
Handles retries, timeouts, redirects, content-type checking.
Returns raw HTML + metadata.
No URL parsing. No HTML parsing. Just HTTP.

Thread-safety: fetch() is safe to call from multiple threads concurrently.
Each thread gets its own requests.Session (via threading.local) so connections
are not shared across threads.  The shared counters (redirects_followed,
total_bytes_downloaded) are updated with a lock.
"""

import logging
import threading
import time
import requests
from typing import Optional
from models import FetchResult, CrawlStatus
from config import (
    REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF,
    DEFAULT_HEADERS, POLITENESS_DELAY, CRAWLABLE_CONTENT_TYPES,
)

logger = logging.getLogger(__name__)


class Fetcher:
    """
    Thread-safe HTTP fetcher.

    One Fetcher instance per crawl.  Internally each worker thread gets its
    own requests.Session via threading.local, so connections are reused within
    a thread but never shared across threads.

    Rate-limiting (POLITENESS_DELAY) is also per-thread: each thread waits the
    required gap between its own consecutive requests.  With N workers the
    aggregate throughput is roughly N / POLITENESS_DELAY pages/second.
    """

    def __init__(self):
        self._local = threading.local()   # thread-local: session + last_request_time
        self._stats_lock = threading.Lock()
        self.redirects_followed: int = 0
        self.total_bytes_downloaded: int = 0

    # ── Thread-local helpers ───────────────────────────────────

    def _session(self) -> requests.Session:
        """Return this thread's Session, creating it on first call."""
        if not hasattr(self._local, "session"):
            s = requests.Session()
            s.headers.update(DEFAULT_HEADERS)
            self._local.session = s
            self._local.last_request_time = 0.0
        return self._local.session

    def _rate_limit(self) -> None:
        """Per-thread politeness delay."""
        elapsed = time.monotonic() - getattr(self._local, "last_request_time", 0.0)
        if elapsed < POLITENESS_DELAY:
            sleep_for = POLITENESS_DELAY - elapsed
            logger.debug(
                "Politeness delay: sleeping %.3fs (elapsed=%.3fs, limit=%.3fs)",
                sleep_for, elapsed, POLITENESS_DELAY,
            )
            time.sleep(sleep_for)

    def _extract_content_type(self, response: requests.Response) -> str:
        ct = response.headers.get("Content-Type", "")
        return ct.split(";")[0].strip().lower()

    # ── Public API ────────────────────────────────────────────

    def fetch(self, url: str) -> FetchResult:
        """Fetch a single URL. Thread-safe."""
        session = self._session()
        last_error: Optional[str] = None

        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                time.sleep(RETRY_BACKOFF ** attempt)

            self._rate_limit()
            self._local.last_request_time = time.monotonic()

            try:
                response = session.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=True,
                    stream=False,
                )

                redirect_chain = [r.url for r in response.history]
                final_url = response.url
                content_bytes = response.content
                content_type = self._extract_content_type(response)
                status_code = response.status_code

                with self._stats_lock:
                    self.redirects_followed += len(response.history)
                    self.total_bytes_downloaded += len(content_bytes)

                if 200 <= status_code < 300:
                    crawl_status = CrawlStatus.REDIRECT if redirect_chain else CrawlStatus.SUCCESS
                    is_crawlable = any(
                        content_type.startswith(ct) for ct in CRAWLABLE_CONTENT_TYPES
                    )
                    if is_crawlable:
                        html = response.text
                    else:
                        html = None
                        crawl_status = CrawlStatus.SKIPPED_CONTENT_TYPE
                    return FetchResult(
                        url=url, final_url=final_url, status_code=status_code,
                        content_type=content_type, html=html,
                        crawl_status=crawl_status, redirect_chain=redirect_chain,
                    )

                elif 400 <= status_code < 500:
                    return FetchResult(
                        url=url, final_url=final_url, status_code=status_code,
                        content_type=content_type,
                        crawl_status=CrawlStatus.HTTP_ERROR, redirect_chain=redirect_chain,
                    )

                elif status_code >= 500:
                    last_error = f"HTTP {status_code}"
                    if attempt < MAX_RETRIES:
                        continue
                    return FetchResult(
                        url=url, final_url=final_url, status_code=status_code,
                        content_type=content_type,
                        crawl_status=CrawlStatus.HTTP_ERROR,
                        error=last_error, redirect_chain=redirect_chain,
                    )

                else:
                    return FetchResult(
                        url=url, final_url=final_url, status_code=status_code,
                        content_type=content_type,
                        crawl_status=CrawlStatus.HTTP_ERROR, redirect_chain=redirect_chain,
                    )

            except requests.Timeout:
                last_error = "timeout"
                if attempt < MAX_RETRIES:
                    continue
                return FetchResult(url=url, final_url=url,
                                   crawl_status=CrawlStatus.TIMEOUT, error=last_error)

            except requests.ConnectionError as e:
                last_error = f"connection_error: {e}"
                if attempt < MAX_RETRIES:
                    continue
                return FetchResult(url=url, final_url=url,
                                   crawl_status=CrawlStatus.ERROR, error=last_error)

            except Exception as e:
                last_error = f"error: {e}"
                if attempt < MAX_RETRIES:
                    continue
                return FetchResult(url=url, final_url=url,
                                   crawl_status=CrawlStatus.ERROR, error=last_error)

        return FetchResult(url=url, final_url=url,
                           crawl_status=CrawlStatus.ERROR, error=last_error or "unknown")

    def check_url_exists(self, url: str) -> tuple[bool, int]:
        """HEAD check. Thread-safe."""
        session = self._session()
        self._rate_limit()
        self._local.last_request_time = time.monotonic()
        try:
            response = session.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            if response.status_code == 405:
                self._rate_limit()
                self._local.last_request_time = time.monotonic()
                response = session.get(url, timeout=REQUEST_TIMEOUT,
                                       allow_redirects=True, stream=True)
                response.close()
            return 200 <= response.status_code < 400, response.status_code
        except Exception:
            return False, 0

    def close(self):
        """No-op — thread-local sessions are cleaned up by GC."""
        pass


"""
Main crawl engine.
Orchestrates: fetch → parse → classify → enqueue → repeat.
"""

import concurrent.futures
import time
from datetime import datetime, timezone
from typing import Optional, Callable
from urllib.parse import urlparse, urljoin

from models import (
    CrawlResult, CrawlQueueItem, CrawlStats, DiscoveredURL, make_crawl_id
)
from crawler.fetcher import Fetcher
from crawler.parser import extract_links, should_skip_url, classify_link
from crawler.frontier import Frontier, _normalize_url
from crawler.signals import detect_signals
from models import URLClassification, CrawlStatus
from config import MAX_WORKERS


class Crawler:
    """
    BFS web crawler for documentation sites.

    Usage:
        crawler = Crawler("https://scikit-learn.org/stable/")
        result = crawler.crawl()
        # result is a CrawlResult with all discovered URLs
    """

    def __init__(
        self,
        seed_url: str,
        on_progress: Optional[Callable[[str, dict], None]] = None,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the crawler.

        Parameters:
        - seed_url: The starting URL
        - on_progress: Optional callback for progress logging.
        """
        parsed = urlparse(seed_url)
        host = parsed.netloc.lower()
        # Strip port and www
        if ":" in host:
            host = host.rsplit(":", 1)[0]
        self.root_domain = host.removeprefix("www.")

        self.seed_url = seed_url
        self.fetcher = Fetcher()
        self.frontier = Frontier(self.root_domain)
        self.on_progress = on_progress or (lambda e, d: None)
        self.max_workers: int = max_workers if max_workers is not None else MAX_WORKERS

        self._crawl_id = make_crawl_id()
        self._started_at: Optional[str] = None
        self._finished_at: Optional[str] = None
        self._start_mono: float = 0.0
        self._signals: dict = {}
        self._pages_fetched: int = 0
        self._pages_failed: int = 0
        self._pages_skipped: int = 0

    def crawl(self) -> CrawlResult:
        """Run the complete crawl. This is the main entry point."""
        # ── 1. SETUP ──
        self._start_mono = time.monotonic()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self.on_progress("start", {"url": self.seed_url})

        # ── 2. SIGNAL DETECTION — fetch seed page first ──
        seed_result = self.fetcher.fetch(self.seed_url)

        if seed_result.crawl_status in (CrawlStatus.SUCCESS, CrawlStatus.REDIRECT):
            self._signals = detect_signals(
                root_url=self.seed_url,
                fetcher=self.fetcher,
                homepage_html=seed_result.html,
            )
        else:
            self._signals = detect_signals(
                root_url=self.seed_url,
                fetcher=self.fetcher,
                homepage_html=None,
            )

        # ── 3. PROCESS SEED PAGE ──
        self.frontier.add_seed(self.seed_url)
        self.frontier.get_next()  # pop seed so we don't re-fetch it

        seed_item = CrawlQueueItem(
            url=seed_result.final_url,
            source_url=self.seed_url,
            depth=0,
            link_text="",
        )
        self.frontier.record_internal(seed_item, http_status=seed_result.status_code)
        self._pages_fetched += 1

        if seed_result.html:
            self._enqueue_links_from(
                html=seed_result.html,
                page_url=seed_result.final_url,
                parent_depth=0,
            )

        # ── 4. CONCURRENT BFS LOOP ──
        #
        # Strategy:
        #   - Main thread owns the frontier (no locking needed there).
        #   - Worker threads ONLY call fetcher.fetch(url) — pure I/O.
        #   - Main thread collects completed futures, processes results
        #     (link extraction + enqueue) one at a time.
        #
        # This gives us N parallel HTTP round-trips while keeping all
        # mutable state (visited set, queues, counters) on one thread.

        in_flight: dict = {}   # future → CrawlQueueItem

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="crawler",
        ) as executor:

            while not self.frontier.is_empty or in_flight:

                # ── Fill worker slots from the frontier ──
                while not self.frontier.is_empty and len(in_flight) < self.max_workers:
                    item = self.frontier.get_next()
                    if item is None:
                        break

                    should_fetch, reason = self.frontier.should_crawl(item)

                    if not should_fetch:
                        self._pages_skipped += 1
                        self.on_progress("skipped", {
                            "url": item.url, "reason": reason, "depth": item.depth
                        })
                        if reason == "external_not_crawled":
                            self.frontier.record_external(item)
                        continue

                    self.on_progress("fetching", {
                        "url": item.url,
                        "depth": item.depth,
                        "queue_size": len(self.frontier.queue),
                        "in_flight": len(in_flight),
                    })

                    future = executor.submit(self.fetcher.fetch, item.url)
                    in_flight[future] = item

                if not in_flight:
                    break

                # ── Wait for at least one fetch to finish ──
                done, _ = concurrent.futures.wait(
                    in_flight,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for future in done:
                    item = in_flight.pop(future)
                    try:
                        fetch_result = future.result()
                    except Exception as exc:
                        self._pages_failed += 1
                        self.on_progress("error", {
                            "url": item.url, "error": str(exc), "depth": item.depth
                        })
                        continue

                    # Mark both requested and final URL as visited
                    self.frontier.visited.add(_normalize_url(item.url))
                    if fetch_result.final_url != item.url:
                        self.frontier.visited.add(_normalize_url(fetch_result.final_url))

                    if fetch_result.crawl_status in (
                        CrawlStatus.TIMEOUT, CrawlStatus.ERROR, CrawlStatus.HTTP_ERROR
                    ):
                        self._pages_failed += 1
                        self.on_progress("error", {
                            "url": item.url,
                            "error": fetch_result.error or str(fetch_result.crawl_status),
                            "depth": item.depth,
                        })
                        continue

                    # Check if a redirect took us off-domain
                    canonical_item = CrawlQueueItem(
                        url=fetch_result.final_url,
                        source_url=item.source_url,
                        depth=item.depth,
                        link_text=item.link_text,
                    )
                    final_parsed = urlparse(fetch_result.final_url)
                    final_host = (
                        final_parsed.netloc.lower().removeprefix("www.").rsplit(":", 1)[0]
                    )
                    root = self.root_domain.lower()
                    is_still_internal = (
                        final_host == root or final_host.endswith("." + root)
                    )

                    if not is_still_internal:
                        self.frontier.record_external(canonical_item)
                        self._pages_skipped += 1
                        continue

                    self.frontier.record_internal(
                        canonical_item, http_status=fetch_result.status_code
                    )
                    self._pages_fetched += 1

                    links_found = 0
                    if fetch_result.html:
                        links_found = self._enqueue_links_from(
                            html=fetch_result.html,
                            page_url=fetch_result.final_url,
                            parent_depth=item.depth,
                        )

                    self.on_progress("fetched", {
                        "url": fetch_result.final_url,
                        "status": fetch_result.status_code,
                        "links_found": links_found,
                        "depth": item.depth,
                    })

        # ── 5. FINALIZE ──
        self._finished_at = datetime.now(timezone.utc).isoformat()
        result = self._build_result(self._signals)
        stats = self.frontier.get_stats()
        self.on_progress("done", {
            "stats": stats,
            "duration": time.monotonic() - self._start_mono,
            "pages_fetched": self._pages_fetched,
            "pages_failed": self._pages_failed,
        })
        self.fetcher.close()
        return result


    def _enqueue_links_from(self, html: str, page_url: str, parent_depth: int) -> int:
        """
        Parse HTML and enqueue all valid links.
        Returns the count of links found.
        """
        try:
            metadata = extract_links(html, page_url)
        except Exception:
            return 0

        count = 0
        for link in metadata.anchor_tags:
            resolved = link.href  # already absolute from extract_links
            if should_skip_url(resolved):
                continue

            classification = classify_link(resolved, self.root_domain)
            new_item = CrawlQueueItem(
                url=resolved,
                source_url=page_url,
                depth=parent_depth + 1,
                link_text=link.text,
            )
            count += 1

            if classification == URLClassification.INTERNAL:
                self.frontier.enqueue(new_item)
            elif classification == URLClassification.EXTERNAL:
                # Record external immediately (not crawled)
                self.frontier.record_external(new_item)

        return count

    def _build_result(self, signals: dict) -> CrawlResult:
        """
        Assemble the final CrawlResult from frontier data.
        """
        stats_data = self.frontier.get_stats()
        duration = time.monotonic() - self._start_mono

        crawl_stats = CrawlStats(
            pages_fetched=self._pages_fetched,
            pages_skipped=self._pages_skipped,
            pages_failed=self._pages_failed,
            internal_urls_discovered=stats_data["internal_discovered"],
            external_urls_discovered=stats_data["external_discovered"],
            unique_external_domains=len(stats_data["external_domains"]),
            max_depth_reached=stats_data["max_depth_reached"],
            redirects_followed=self.fetcher.redirects_followed,
            total_bytes_downloaded=self.fetcher.total_bytes_downloaded,
        )

        return CrawlResult(
            crawl_id=self._crawl_id,
            root_domain=self.root_domain,
            root_urls=[self.seed_url],
            started_at=self._started_at or "",
            finished_at=self._finished_at or "",
            duration_seconds=round(duration, 2),
            signals=signals,
            internal_urls=self.frontier.internal_urls,
            external_urls=self.frontier.external_urls,
            stats=crawl_stats,
        )

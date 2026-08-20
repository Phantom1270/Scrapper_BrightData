"""
Phase 1 — Entry Point

Usage:
    python main.py
    python main.py --url https://scikit-learn.org/stable/
    python main.py --url https://scikit-learn.org/stable/ --output output.json
    python main.py --url https://scikit-learn.org/stable/ --depth 2 --verbose
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from crawler.crawler import Crawler
from config import INPUT_URL, MAX_INTERNAL_DEPTH, DEFAULT_OUTPUT_FILE, MAX_WORKERS


# ── Progress state shared across callbacks ──
_progress_state = {
    "last_event": None,
    "start_time": 0.0,
}


def progress_printer(event_type: str, data: dict) -> None:
    """
    Print crawl progress to stdout.
    """
    sep = "=" * 45

    if event_type == "start":
        print(sep)
        print("CRAWL STARTING")
        print(f"Seed: {data.get('url', '')}")
        print(sep)
        _progress_state["start_time"] = time.monotonic()

    elif event_type == "fetching":
        depth = data.get("depth", 0)
        queue = data.get("queue_size", 0)
        url = data.get("url", "")
        print(f"[depth={depth} queue={queue}] Fetching {url}")

    elif event_type == "fetched":
        depth = data.get("depth", 0)
        queue = data.get("queue_size", 0)
        status = data.get("status", "?")
        links = data.get("links_found", 0)
        print(f"  [OK] {status} ({links} links found)")

    elif event_type == "skipped":
        depth = data.get("depth", 0)
        url = data.get("url", "")
        reason = data.get("reason", "")
        print(f"[depth={depth}] [SKIP] {url} ({reason})")

    elif event_type == "external":
        depth = data.get("depth", 0)
        url = data.get("url", "")
        domain = data.get("domain", "")
        print(f"[depth={depth}] -> External: {domain}")

    elif event_type == "error":
        depth = data.get("depth", 0)
        url = data.get("url", "")
        error = data.get("error", "unknown")
        print(f"[depth={depth}] [ERR] {url} ({error})")

    elif event_type == "done":
        elapsed = time.monotonic() - _progress_state.get("start_time", time.monotonic())
        stats = data.get("stats", {})
        fetched = data.get("pages_fetched", 0)
        failed = data.get("pages_failed", 0)
        print()
        print(sep)
        print("CRAWL COMPLETE")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Pages fetched: {fetched}")
        print(f"Pages failed: {failed}")
        print(f"Internal URLs: {stats.get('internal_discovered', 0)}")
        print(f"External URLs: {stats.get('external_discovered', 0)}")
        print(f"Max depth: {stats.get('max_depth_reached', 0)}")
        print(sep)


def save_output(crawler_result, output_path: str) -> None:
    """
    Serialize CrawlResult to JSON.

    Only include fields that Phase 2 needs:
    - crawl_id, root_domain, root_urls, signals, internal_urls, external_urls

    Do NOT include: started_at, finished_at, duration_seconds, stats.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def url_to_dict(u) -> dict:
        d = {
            "url": u.url,
            "source_url": u.source_url,
            "depth": u.depth,
            "link_text": u.link_text,
        }
        if u.http_status is not None:
            d["http_status"] = u.http_status
        return d

    output = {
        "crawl_id": crawler_result.crawl_id,
        "root_domain": crawler_result.root_domain,
        "root_urls": crawler_result.root_urls,
        "signals": crawler_result.signals,
        "internal_urls": [url_to_dict(u) for u in crawler_result.internal_urls],
        "external_urls": [url_to_dict(u) for u in crawler_result.external_urls],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Web Crawler — discovers URLs from a documentation site"
    )
    parser.add_argument(
        "--url", "-u",
        default=None,
        help=f"Seed URL to crawl (default: hardcoded in config.py → {INPUT_URL})"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=None,
        help=f"Max internal crawl depth (default: {MAX_INTERNAL_DEPTH})"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help=f"Concurrent fetch threads (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all output except errors"
    )

    args = parser.parse_args()

    # Resolve settings
    seed_url = args.url or INPUT_URL
    output_path = args.output or DEFAULT_OUTPUT_FILE
    max_depth = args.depth if args.depth is not None else MAX_INTERNAL_DEPTH

    # Override config if --depth provided
    import config
    config.MAX_INTERNAL_DEPTH = max_depth

    # Progress callback
    progress = None
    if not args.quiet:
        progress = progress_printer

    # Enable debug logging for fetcher when verbose
    if args.verbose:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(name)s %(levelname)s: %(message)s",
        )
        # Suppress overly chatty urllib3 debug output
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    # Run crawl
    if not args.quiet:
        print(f"Crawling: {seed_url}")
        print(f"Max depth: {max_depth}")
        print(f"Output: {output_path}")
        print()

    crawler = Crawler(seed_url, on_progress=progress, max_workers=args.workers)
    result = crawler.crawl()

    # Save output
    save_output(result, output_path)

    # Print summary
    if not args.quiet:
        print(f"\nResults saved to: {output_path}")
        print(f"Internal URLs: {len(result.internal_urls)}")
        print(f"External URLs: {len(result.external_urls)}")


if __name__ == "__main__":
    main()

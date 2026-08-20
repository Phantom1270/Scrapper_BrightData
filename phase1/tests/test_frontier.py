"""
Test URL frontier (BFS queue + limits).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from crawler.frontier import Frontier
from models import CrawlQueueItem
from config import MAX_INTERNAL_DEPTH, MAX_INTERNAL_URLS


class TestFrontier:
    def test_add_seed(self):
        frontier = Frontier("scikit-learn.org")
        frontier.add_seed("https://scikit-learn.org/stable/")
        assert not frontier.is_empty
        item = frontier.get_next()
        assert item.url == "https://scikit-learn.org/stable/"
        assert item.depth == 0

    def test_enqueue_internal(self):
        frontier = Frontier("scikit-learn.org")
        item = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="https://scikit-learn.org/stable/",
            depth=1,
            link_text="Install"
        )
        assert frontier.enqueue(item) is True
        assert not frontier.is_empty

    def test_deduplication(self):
        frontier = Frontier("scikit-learn.org")
        item1 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="https://scikit-learn.org/stable/",
            depth=1
        )
        item2 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="https://scikit-learn.org/stable/modules.html",
            depth=2
        )
        frontier.enqueue(item1)
        # Second enqueue of same URL should be rejected
        result = frontier.enqueue(item2)
        assert result is False
        # Only one item in queue
        assert frontier.get_next() is not None
        assert frontier.is_empty

    def test_should_crawl_respects_depth(self):
        frontier = Frontier("scikit-learn.org")
        item = CrawlQueueItem(
            url="https://scikit-learn.org/stable/deep/page.html",
            source_url="https://scikit-learn.org/stable/",
            depth=MAX_INTERNAL_DEPTH + 1
        )
        should, reason = frontier.should_crawl(item)
        assert should is False
        assert "depth" in reason.lower()

    def test_should_not_crawl_external(self):
        frontier = Frontier("scikit-learn.org")
        item = CrawlQueueItem(
            url="https://numpy.org/doc/",
            source_url="https://scikit-learn.org/stable/",
            depth=1
        )
        should, reason = frontier.should_crawl(item)
        assert should is False
        assert "external" in reason.lower()

    def test_record_internal(self):
        frontier = Frontier("scikit-learn.org")
        item = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="https://scikit-learn.org/stable/",
            depth=1,
            link_text="Install"
        )
        frontier.record_internal(item)
        assert len(frontier.internal_urls) == 1
        assert frontier.internal_urls[0].url == "https://scikit-learn.org/stable/install.html"
        assert frontier.internal_urls[0].link_text == "Install"

    def test_record_external(self):
        frontier = Frontier("scikit-learn.org")
        item = CrawlQueueItem(
            url="https://numpy.org/doc/",
            source_url="https://scikit-learn.org/stable/",
            depth=1,
            link_text="NumPy"
        )
        frontier.record_external(item)
        assert len(frontier.external_urls) == 1
        assert frontier.external_urls[0].url == "https://numpy.org/doc/"

    def test_bfs_order(self):
        """Items should come out in FIFO order."""
        frontier = Frontier("example.com")
        frontier.enqueue(CrawlQueueItem(url="https://example.com/a", source_url="", depth=1))
        frontier.enqueue(CrawlQueueItem(url="https://example.com/b", source_url="", depth=1))
        frontier.enqueue(CrawlQueueItem(url="https://example.com/c", source_url="", depth=1))

        assert frontier.get_next().url == "https://example.com/a"
        assert frontier.get_next().url == "https://example.com/b"
        assert frontier.get_next().url == "https://example.com/c"
        assert frontier.is_empty

    def test_stats(self):
        frontier = Frontier("example.com")
        frontier.add_seed("https://example.com/")
        frontier.get_next()  # consume seed
        frontier.record_internal(CrawlQueueItem(url="https://example.com/a", source_url="", depth=1))
        frontier.record_external(CrawlQueueItem(url="https://other.com/", source_url="", depth=1))

        stats = frontier.get_stats()
        assert stats["internal_discovered"] == 1
        assert stats["external_discovered"] == 1

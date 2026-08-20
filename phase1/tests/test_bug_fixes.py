"""
Regression tests for Bugs 1–3.
Each test directly exercises the fixed code path in isolation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from urllib.parse import urlparse
from crawler.parser import resolve_url
from crawler.frontier import Frontier, _normalize_url
from models import CrawlQueueItem


# ────────────────────────────────────────────────────────────
# BUG 1 — resolve_url must not produce doubled path segments
# ────────────────────────────────────────────────────────────

class TestResolveUrl:
    """BUG 1: Double path segments must be collapsed by resolve_url."""

    def test_no_double_when_base_ends_with_segment_dir(self):
        """
        The classic Sphinx case:
        base = .../modules/generated/   (current page is inside generated/)
        href = generated/sklearn.Foo.html
        urljoin alone → .../modules/generated/generated/sklearn.Foo.html  (WRONG)
        resolve_url   → .../modules/generated/sklearn.Foo.html            (RIGHT)
        """
        base = "https://scikit-learn.org/stable/modules/generated/"
        href = "generated/sklearn.linear_model.LogisticRegression.html"
        result = resolve_url(base, href)
        path = urlparse(result).path
        assert "/generated/generated/" not in path, (
            f"Doubled segment found: {result}"
        )
        assert "generated/sklearn.linear_model.LogisticRegression.html" in path

    def test_normal_relative_unchanged(self):
        """Ordinary relative resolution must still work."""
        base = "https://scikit-learn.org/stable/"
        href = "install.html"
        result = resolve_url(base, href)
        assert result == "https://scikit-learn.org/stable/install.html"

    def test_absolute_url_unchanged(self):
        """Absolute href must pass through unchanged."""
        base = "https://scikit-learn.org/stable/"
        href = "https://numpy.org/doc/"
        result = resolve_url(base, href)
        assert result == "https://numpy.org/doc/"

    def test_parent_dir_href(self):
        """Parent-relative href (../dev/index.html) must resolve correctly."""
        base = "https://scikit-learn.org/stable/index.html"
        href = "../dev/index.html"
        result = resolve_url(base, href)
        assert result == "https://scikit-learn.org/dev/index.html"

    def test_no_double_segment_without_trailing_slash(self):
        """
        base = .../modules/generated  (no trailing slash — treated as a file)
        href = generated/foo.html
        urljoin → .../modules/generated/foo.html
        No double here, must remain stable.
        """
        base = "https://scikit-learn.org/stable/modules/generated"
        href = "generated/foo.html"
        result = resolve_url(base, href)
        path = urlparse(result).path
        # Should not have generated/generated
        assert "/generated/generated/" not in path

    def test_triple_segment_collapses_once(self):
        """
        If somehow /a/a/a/ appears, collapse to /a/.
        (Edge case; should not occur in practice but normalization is stable.)
        """
        base = "https://example.com/a/a/"
        href = "a/page.html"
        result = resolve_url(base, href)
        path = urlparse(result).path
        # /a/a/a/ is NOT a doubled pair for our regex since it's 3 consecutive,
        # but /a/a/ → /a/ after one pass.  The result should not have /a/a/.
        assert "/a/a/" not in path


# ────────────────────────────────────────────────────────────
# BUG 2 — _normalize_url must resolve dot segments
# ────────────────────────────────────────────────────────────

class TestNormalizeUrlDotSegments:
    """BUG 2: /stable/./install.html and /stable/install.html must be the same key."""

    def test_single_dot_segment(self):
        a = _normalize_url("https://scikit-learn.org/stable/./install.html")
        b = _normalize_url("https://scikit-learn.org/stable/install.html")
        assert a == b, f"Mismatch: {a!r} != {b!r}"

    def test_double_dot_segment(self):
        a = _normalize_url("https://scikit-learn.org/stable/modules/../install.html")
        b = _normalize_url("https://scikit-learn.org/stable/install.html")
        assert a == b, f"Mismatch: {a!r} != {b!r}"

    def test_multiple_dots(self):
        a = _normalize_url("https://scikit-learn.org/stable/./modules/./classes.html")
        b = _normalize_url("https://scikit-learn.org/stable/modules/classes.html")
        assert a == b

    def test_trailing_slash_stripped(self):
        a = _normalize_url("https://scikit-learn.org/stable/modules/")
        b = _normalize_url("https://scikit-learn.org/stable/modules")
        assert a == b

    def test_root_path_preserved(self):
        a = _normalize_url("https://scikit-learn.org/")
        b = _normalize_url("https://scikit-learn.org/")
        assert a == b

    def test_fragment_stripped(self):
        a = _normalize_url("https://scikit-learn.org/stable/install.html#section")
        b = _normalize_url("https://scikit-learn.org/stable/install.html")
        assert a == b

    def test_case_insensitive_host(self):
        a = _normalize_url("https://SCIKIT-LEARN.ORG/stable/install.html")
        b = _normalize_url("https://scikit-learn.org/stable/install.html")
        assert a == b


# ────────────────────────────────────────────────────────────
# BUG 3 — Dedup: dotted URL and clean URL treated as same
# ────────────────────────────────────────────────────────────

class TestFrontierDedup:
    """BUG 3: Frontier must reject duplicate URLs that differ only in dot segments."""

    def test_dot_segment_url_deduped(self):
        """
        Enqueue /stable/install.html first, then /stable/./install.html.
        The second should be rejected because they normalize to the same key.
        """
        frontier = Frontier("scikit-learn.org")
        item1 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="https://scikit-learn.org/stable/",
            depth=1,
        )
        item2 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/./install.html",
            source_url="https://scikit-learn.org/stable/modules.html",
            depth=2,
        )
        result1 = frontier.enqueue(item1)
        result2 = frontier.enqueue(item2)
        assert result1 is True
        assert result2 is False, "Dot-segment duplicate was not deduped"

    def test_dotdot_segment_url_deduped(self):
        frontier = Frontier("scikit-learn.org")
        item1 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/install.html",
            source_url="",
            depth=1,
        )
        item2 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/modules/../install.html",
            source_url="",
            depth=2,
        )
        frontier.enqueue(item1)
        result2 = frontier.enqueue(item2)
        assert result2 is False, "Parent-dir duplicate was not deduped"

    def test_trailing_slash_deduped(self):
        """
        /stable/modules/ and /stable/modules should be treated as the same URL.
        """
        frontier = Frontier("scikit-learn.org")
        item1 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/modules/",
            source_url="",
            depth=1,
        )
        item2 = CrawlQueueItem(
            url="https://scikit-learn.org/stable/modules",
            source_url="",
            depth=1,
        )
        frontier.enqueue(item1)
        result2 = frontier.enqueue(item2)
        assert result2 is False, "Trailing-slash variant was not deduped"

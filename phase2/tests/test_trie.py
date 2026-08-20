"""
Unit tests for utils/trie.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.trie import TrieNode, build_trie
from models import ParsedURL, URLClassification, SegmentType


def make_parsed_url(path: str, url: str = None) -> ParsedURL:
    """Helper to create a minimal ParsedURL for testing."""
    segments = path.strip("/").split("/")
    return ParsedURL(
        original_url=url or f"https://example.com{path}",
        canonical_url=url or f"https://example.com{path}",
        domain="example.com",
        scheme="https",
        path=path,
        segments=[],  # Not needed for trie tests
        classification=URLClassification.INTERNAL,
        is_asset=False,
        version_free_path=path,
    )


class TestTrie:
    def test_add_single_url(self):
        root = TrieNode()
        root.add_url(["modules", "generated", "file.html"], "https://example.com/modules/generated/file.html")
        assert "modules" in root.children
        assert "generated" in root.children["modules"].children
        assert root.children["modules"].children["generated"].children["file.html"].is_leaf

    def test_common_prefix(self):
        root = TrieNode()
        root.add_url(["a", "b", "c.html"], "url1")
        root.add_url(["a", "b", "d.html"], "url2")
        # "a" and "b" should be shared
        assert len(root.children) == 1
        assert len(root.children["a"].children) == 1
        assert len(root.children["a"].children["b"].children) == 2

    def test_different_prefixes(self):
        root = TrieNode()
        root.add_url(["x", "file.html"], "url1")
        root.add_url(["y", "file.html"], "url2")
        assert len(root.children) == 2

    def test_collect_leaves(self):
        root = TrieNode()
        root.add_url(["a", "1.html"], "url1")
        root.add_url(["a", "2.html"], "url2")
        root.add_url(["b", "3.html"], "url3")
        leaves = root.collect_leaves()
        assert len(leaves) == 3
        assert "url1" in leaves
        assert "url3" in leaves

    def test_build_trie_from_parsed_urls(self):
        urls = [
            make_parsed_url("/modules/generated/a.html"),
            make_parsed_url("/modules/generated/b.html"),
            make_parsed_url("/user_guide/c.html"),
        ]
        root = build_trie(urls)
        assert "modules" in root.children
        assert "user_guide" in root.children

    def test_metrics_computed(self):
        urls = [
            make_parsed_url("/modules/generated/a.html"),
            make_parsed_url("/modules/generated/b.html"),
        ]
        root = build_trie(urls)
        # After compute_metrics, root should know about descendants
        assert root.descendant_count == 2

    def test_to_string_runs(self):
        """to_string should not crash."""
        urls = [
            make_parsed_url("/a/b/c.html"),
            make_parsed_url("/a/d.html"),
        ]
        root = build_trie(urls)
        s = root.to_string()
        assert isinstance(s, str)
        assert len(s) > 0

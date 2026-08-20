"""
Unit tests for pipeline/canonicalization.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.canonicalization import canonicalize


class TestCanonicalize:
    def test_removes_fragment(self):
        result = canonicalize("https://example.com/page.html#section")
        assert "#" not in result
        assert result == "https://example.com/page.html"

    def test_removes_utm_params(self):
        result = canonicalize("https://example.com/page.html?utm_source=docs&utm_medium=link")
        assert "utm_" not in result
        assert result == "https://example.com/page.html"

    def test_keeps_important_params(self):
        result = canonicalize("https://example.com/search?q=test&page=2")
        assert "q=test" in result
        assert "page=2" in result

    def test_sorts_params(self):
        result = canonicalize("https://example.com/page?b=2&a=1")
        assert "a=1" in result
        # a=1 should come before b=2
        assert result.index("a=1") < result.index("b=2")

    def test_removes_trailing_slash(self):
        result = canonicalize("https://example.com/stable/modules/")
        assert result == "https://example.com/stable/modules"

    def test_preserves_root_slash(self):
        result = canonicalize("https://example.com/")
        assert result == "https://example.com/"

    def test_lowercase_host(self):
        result = canonicalize("https://EXAMPLE.COM/page.html")
        assert "example.com" in result

    def test_removes_default_port(self):
        result = canonicalize("https://example.com:443/page.html")
        assert ":443" not in result

    def test_normalizes_double_slash(self):
        result = canonicalize("https://example.com//stable//modules//page.html")
        assert "//" not in result.replace("https://", "")

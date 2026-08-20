"""
Integration test: run a mock crawl and verify output structure.
Uses mock HTTP responses so no real HTTP calls are made.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from crawler.crawler import Crawler
from models import CrawlStatus
import config as _config


FIXTURE_DIR = Path(__file__).parent / "fixtures"

# Empty page for subsequent fetches (no new links → crawl terminates quickly)
_EMPTY_PAGE = "<html><head><title>Empty</title></head><body><p>No links here.</p></body></html>"


def _make_response(html: str, url: str, status: int = 200, content_type: str = "text/html") -> MagicMock:
    """Helper to create a mock requests Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"Content-Type": f"{content_type}; charset=utf-8"}
    resp.content = html.encode("utf-8")
    resp.text = html
    resp.url = url
    resp.history = []
    return resp


def _limited_get_factory(seed_html: str, seed_url: str, max_fetches: int = 2):
    """
    Returns a mock GET function that serves seed_html for the seed page,
    then _EMPTY_PAGE for all subsequent URLs (stops infinite recursion).
    """
    call_count = [0]

    def mock_get(url, **kwargs):
        call_count[0] += 1
        if url == seed_url or call_count[0] <= 1:
            return _make_response(seed_html, url)
        else:
            return _make_response(_EMPTY_PAGE, url)

    return mock_get


def _mock_head(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 404
    return resp


class TestFullCrawl:
    """Test full crawl with mocked HTTP."""

    @pytest.fixture(autouse=True)
    def zero_delay(self):
        """Zero politeness delay so tests are fast."""
        import crawler.fetcher as _fetcher_mod
        orig = _fetcher_mod.POLITENESS_DELAY
        _fetcher_mod.POLITENESS_DELAY = 0
        yield
        _fetcher_mod.POLITENESS_DELAY = orig

    @pytest.fixture
    def sample_html(self):
        with open(FIXTURE_DIR / "sample_page.html") as f:
            return f.read()

    def _run_crawl(self, sample_html: str, seed_url: str = "https://scikit-learn.org/stable/"):
        """Run a quick crawl where only the seed page has real content."""
        mock_get = _limited_get_factory(sample_html, seed_url)

        # Build a mock session that routes get/head through our factories
        mock_session = MagicMock()
        mock_session.get.side_effect = mock_get
        mock_session.head.side_effect = _mock_head

        # max_workers=1 keeps test execution deterministic (no thread races)
        crawler = Crawler(seed_url, on_progress=None, max_workers=1)
        with patch.object(crawler.fetcher, "_session", return_value=mock_session):
            return crawler.crawl()

    def test_crawl_produces_result(self, sample_html):
        """Crawler should return a CrawlResult."""
        seed_url = "https://scikit-learn.org/stable/"
        result = self._run_crawl(sample_html, seed_url)

        assert result is not None
        assert result.crawl_id.startswith("crawl_")
        assert result.root_domain == "scikit-learn.org"
        assert result.root_urls == [seed_url]

    def test_crawl_records_internal_urls(self, sample_html):
        """Internal URLs should be recorded."""
        result = self._run_crawl(sample_html)

        assert len(result.internal_urls) >= 1
        urls = [u.url for u in result.internal_urls]
        assert any("scikit-learn.org" in u for u in urls)

    def test_crawl_records_external_urls(self, sample_html):
        """External URLs (numpy.org, scipy.org) should be recorded."""
        result = self._run_crawl(sample_html)

        ext_domains = {u.url.split("/")[2] for u in result.external_urls}
        assert "numpy.org" in ext_domains or "scipy.org" in ext_domains

    def test_crawl_has_signals(self, sample_html):
        """Signals dict should be populated."""
        result = self._run_crawl(sample_html)

        assert isinstance(result.signals, dict)
        # Should have detected sphinx from meta generator in sample_page.html
        detected = result.signals.get("detected_tool")
        assert detected is not None

    def test_no_asset_urls_in_internal(self, sample_html):
        """Asset URLs (.css, .js, .png) should not appear in internal_urls."""
        result = self._run_crawl(sample_html)

        for u in result.internal_urls:
            assert not u.url.endswith(".css"), f"Asset in internal: {u.url}"
            assert not u.url.endswith(".js"), f"Asset in internal: {u.url}"
            assert not u.url.endswith(".png"), f"Asset in internal: {u.url}"

    def test_output_schema_matches_phase2_input(self, sample_html, tmp_path):
        """
        The output JSON must be valid Phase 2 input.
        Verify required keys and structure.
        """
        seed_url = "https://scikit-learn.org/stable/"
        output_file = tmp_path / "test_output.json"

        from main import save_output
        result = self._run_crawl(sample_html, seed_url)
        save_output(result, str(output_file))

        with open(output_file) as f:
            data = json.load(f)

        # Check required top-level keys
        required_keys = {"crawl_id", "root_domain", "root_urls", "signals",
                         "internal_urls", "external_urls"}
        assert required_keys.issubset(set(data.keys()))

        # Stats should NOT be in output (Phase 1 internal only)
        assert "stats" not in data
        assert "started_at" not in data
        assert "finished_at" not in data

        # Each internal URL must have required fields
        for u in data["internal_urls"]:
            assert "url" in u
            assert "source_url" in u
            assert "depth" in u
            assert "link_text" in u


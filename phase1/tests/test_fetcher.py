"""
Unit tests for Fetcher.
These tests use mock responses to avoid real HTTP calls.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from crawler.fetcher import Fetcher
from models import CrawlStatus


class TestFetcher:
    def test_init(self):
        fetcher = Fetcher()
        # Session is now thread-local; just verify the object constructs.
        assert fetcher.redirects_followed == 0
        assert fetcher.total_bytes_downloaded == 0
        fetcher.close()

    def test_fetch_success(self):
        fetcher = Fetcher()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.content = b"<html><body>Hello</body></html>"
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.url = "https://example.com/page.html"
        mock_response.history = []

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch.object(fetcher, "_session", return_value=mock_session):
            result = fetcher.fetch("https://example.com/page.html")

        assert result.crawl_status == CrawlStatus.SUCCESS
        assert result.html == "<html><body>Hello</body></html>"
        assert result.status_code == 200
        fetcher.close()

    def test_fetch_non_html_content_type(self):
        fetcher = Fetcher()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.content = b"%PDF..."
        mock_response.text = ""
        mock_response.url = "https://example.com/doc.pdf"
        mock_response.history = []

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch.object(fetcher, "_session", return_value=mock_session):
            result = fetcher.fetch("https://example.com/doc.pdf")

        assert result.crawl_status == CrawlStatus.SKIPPED_CONTENT_TYPE
        assert result.html is None
        fetcher.close()

    def test_fetch_404(self):
        fetcher = Fetcher()
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.content = b"Not found"
        mock_response.url = "https://example.com/missing.html"
        mock_response.history = []

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with patch.object(fetcher, "_session", return_value=mock_session):
            result = fetcher.fetch("https://example.com/missing.html")

        assert result.crawl_status == CrawlStatus.HTTP_ERROR
        assert result.status_code == 404
        fetcher.close()

    def test_fetch_timeout(self):
        import requests as req
        fetcher = Fetcher()
        mock_session = MagicMock()
        mock_session.get.side_effect = req.Timeout()

        with patch.object(fetcher, "_session", return_value=mock_session):
            result = fetcher.fetch("https://example.com/slow.html")

        assert result.crawl_status == CrawlStatus.TIMEOUT
        fetcher.close()

    def test_fetch_redirect(self):
        fetcher = Fetcher()
        redirect_response = MagicMock()
        redirect_response.url = "https://example.com/old.html"
        redirect_response.status_code = 301

        final_response = MagicMock()
        final_response.status_code = 200
        final_response.headers = {"Content-Type": "text/html"}
        final_response.content = b"<html>Final</html>"
        final_response.text = "<html>Final</html>"
        final_response.url = "https://example.com/new.html"
        final_response.history = [redirect_response]

        mock_session = MagicMock()
        mock_session.get.return_value = final_response

        with patch.object(fetcher, "_session", return_value=mock_session):
            result = fetcher.fetch("https://example.com/old.html")

        assert result.crawl_status == CrawlStatus.REDIRECT
        assert result.final_url == "https://example.com/new.html"
        fetcher.close()

    def test_close(self):
        fetcher = Fetcher()
        fetcher.close()  # Should not raise

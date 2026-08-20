"""
Test HTML parsing and link extraction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from crawler.parser import extract_links, should_skip_url, classify_link


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class TestExtractLinks:
    """Test link extraction from HTML."""

    @pytest.fixture
    def page_html(self):
        with open(FIXTURE_DIR / "sample_page.html") as f:
            return f.read()

    @pytest.fixture
    def metadata(self, page_html):
        return extract_links(page_html, "https://scikit-learn.org/stable/index.html")

    def test_finds_title(self, metadata):
        assert metadata.title == "scikit-learn"

    def test_finds_meta_generator(self, metadata):
        assert "sphinx" in metadata.meta_generator.lower()

    def test_finds_meta_description(self, metadata):
        assert "scikit-learn" in metadata.meta_description.lower() or \
               "predictive" in metadata.meta_description.lower() or \
               len(metadata.meta_description) > 0

    def test_finds_canonical(self, metadata):
        assert "index.html" in metadata.canonical_url

    def test_extracts_anchor_links(self, metadata):
        hrefs = [link.href for link in metadata.anchor_tags]
        # install.html, user_guide.html, modules/classes.html,
        # auto_examples/index.html, two generated links, two external links, dev/index.html, logo.png
        assert len(hrefs) >= 8

    def test_resolves_relative_urls(self, metadata):
        hrefs = [link.href for link in metadata.anchor_tags]
        # All should be resolved to absolute
        full_urls = [h for h in hrefs if h.startswith("https://")]
        assert len(full_urls) >= 8

    def test_preserves_link_text(self, metadata):
        lr_link = [l for l in metadata.anchor_tags if "LogisticRegression" in l.href and ".html" in l.href]
        assert len(lr_link) == 1
        assert lr_link[0].text == "LogisticRegression"

    def test_skips_fragment_only_links(self, metadata):
        hrefs = [link.href for link in metadata.anchor_tags]
        fragment_only = [h for h in hrefs if h.startswith("#")]
        assert len(fragment_only) == 0

    def test_skips_mailto_links(self, metadata):
        hrefs = [link.href for link in metadata.anchor_tags]
        mailto = [h for h in hrefs if h.startswith("mailto:")]
        assert len(mailto) == 0

    def test_extracts_stylesheet_link(self, metadata):
        link_hrefs = [l.href for l in metadata.link_tags]
        css_links = [h for h in link_hrefs if "theme.css" in h]
        assert len(css_links) >= 1

    def test_external_links_resolved(self, metadata):
        hrefs = [link.href for link in metadata.anchor_tags]
        numpy_links = [h for h in hrefs if "numpy.org" in h]
        assert len(numpy_links) >= 1
        assert numpy_links[0].startswith("https://numpy.org/")


class TestShouldSkipUrl:
    def test_skip_css(self):
        assert should_skip_url("https://example.com/_static/theme.css") is True

    def test_skip_js(self):
        assert should_skip_url("https://example.com/assets/app.js") is True

    def test_skip_image(self):
        assert should_skip_url("https://example.com/img/logo.png") is True

    def test_skip_pdf(self):
        assert should_skip_url("https://example.com/docs/manual.pdf") is True

    def test_dont_skip_html(self):
        assert should_skip_url("https://example.com/docs/page.html") is False

    def test_dont_skip_directory(self):
        assert should_skip_url("https://example.com/docs/") is False

    def test_skip_no_scheme(self):
        assert should_skip_url("not-a-url") is True

    def test_skip_rss_feed(self):
        assert should_skip_url("https://example.com/feed") is True


class TestClassifyLink:
    def test_internal_exact(self):
        result = classify_link(
            "https://scikit-learn.org/stable/install.html",
            "scikit-learn.org"
        )
        assert result.value == "internal"

    def test_external(self):
        result = classify_link(
            "https://numpy.org/doc/",
            "scikit-learn.org"
        )
        assert result.value == "external"

    def test_internal_subdomain(self):
        result = classify_link(
            "https://docs.scikit-learn.org/stable/",
            "scikit-learn.org"
        )
        assert result.value == "internal"

    def test_www_normalization(self):
        result = classify_link(
            "https://www.scikit-learn.org/stable/",
            "scikit-learn.org"
        )
        assert result.value == "internal"

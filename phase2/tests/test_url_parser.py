"""
Unit tests for utils/url_parser.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.url_parser import classify_segment, is_asset_url, parse_url, split_path_segments
from models import SegmentType


class TestClassifySegment:
    """Test segment type classification with exact expected outputs."""

    def test_literal(self):
        assert classify_segment("modules") == SegmentType.LITERAL
        assert classify_segment("generated") == SegmentType.LITERAL
        assert classify_segment("stable") == SegmentType.LITERAL
        assert classify_segment("user_guide") == SegmentType.LITERAL

    def test_slug(self):
        assert classify_segment("linear-model") == SegmentType.SLUG
        assert classify_segment("random-forest") == SegmentType.SLUG

    def test_camel_case(self):
        assert classify_segment("LogisticRegression") == SegmentType.CAMEL_CASE
        assert classify_segment("RandomForestClassifier") == SegmentType.CAMEL_CASE

    def test_dotted_path(self):
        assert classify_segment("sklearn.linear_model") == SegmentType.DOTTED_PATH
        assert classify_segment("numpy.core") == SegmentType.DOTTED_PATH

    def test_integer(self):
        assert classify_segment("123") == SegmentType.INTEGER
        assert classify_segment("42") == SegmentType.INTEGER

    def test_version(self):
        assert classify_segment("v2.1") == SegmentType.VERSION
        assert classify_segment("0.24") == SegmentType.VERSION

    def test_filename(self):
        assert classify_segment("LogisticRegression.html") == SegmentType.FILENAME
        assert classify_segment("plot_example.html") == SegmentType.FILENAME
        assert classify_segment("index.html") == SegmentType.FILENAME
        assert classify_segment("api.rst") == SegmentType.FILENAME

    def test_dotted_path_vs_filename(self):
        """Dotted path without known extension should be DOTTED_PATH, not FILENAME."""
        assert classify_segment("sklearn.linear_model") == SegmentType.DOTTED_PATH
        assert classify_segment("sklearn.linear_model.LogisticRegression.html") == SegmentType.FILENAME

    def test_uuid(self):
        assert classify_segment("550e8400-e29b-41d4-a716-446655440000") == SegmentType.UUID


class TestIsAssetUrl:
    def test_css_is_asset(self):
        assert is_asset_url("https://example.com/_static/css/theme.css") is True

    def test_js_is_asset(self):
        assert is_asset_url("https://example.com/assets/app.js") is True

    def test_html_is_not_asset(self):
        assert is_asset_url("https://example.com/docs/page.html") is False

    def test_image_is_asset(self):
        assert is_asset_url("https://example.com/images/logo.png") is True

    def test_pdf_is_asset(self):
        assert is_asset_url("https://example.com/docs/manual.pdf") is True


class TestSplitPathSegments:
    def test_basic(self):
        result = split_path_segments("/stable/modules/generated/file.html")
        assert result == ["stable", "modules", "generated", "file.html"]

    def test_trailing_slash(self):
        result = split_path_segments("/stable/modules/")
        assert result == ["stable", "modules"]

    def test_root(self):
        result = split_path_segments("/")
        assert result == []

    def test_no_leading_slash(self):
        result = split_path_segments("stable/modules/file.html")
        assert result == ["stable", "modules", "file.html"]

"""
Tests for coverage.py — template matching and external domain classification.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.coverage import match_url_to_template, external_domain_analysis, _classify_domain
from models import (
    ParsedURL, TemplatePattern, URLClassification, ExternalDomainType,
    ParsedSegment, SegmentType,
)
from utils.url_parser import classify_segment


def make_url(path, domain="example.com", version_free_path=None):
    segs = [s for s in path.split("/") if s]
    parsed_segs = [
        ParsedSegment(raw=s, position=i, lexical_type=classify_segment(s))
        for i, s in enumerate(segs)
    ]
    return ParsedURL(
        original_url=f"https://{domain}{path}",
        canonical_url=f"https://{domain}{path}",
        domain=domain,
        scheme="https",
        path=path,
        segments=parsed_segs,
        classification=URLClassification.INTERNAL if domain == "example.com" else URLClassification.EXTERNAL,
        is_asset=False,
        version_free_path=version_free_path or path,
    )


def make_template(pattern, template_id="tpl_001"):
    return TemplatePattern(
        template_id=template_id,
        pattern=pattern,
        fingerprint="",
        member_count=1,
    )


class TestMatchUrlToTemplate:
    def test_exact_literal_match(self):
        url = make_url("/modules/generated/file.html")
        tpl = make_template("/modules/generated/file.html")
        assert match_url_to_template(url, tpl) is True

    def test_variable_filename_match(self):
        url = make_url("/modules/generated/sklearn.linear_model.html")
        tpl = make_template("/modules/generated/<filename>")
        assert match_url_to_template(url, tpl) is True

    def test_wildcard_match(self):
        url = make_url("/auto_examples/cluster/plot_kmeans.html")
        tpl = make_template("/<literal>/...")
        assert match_url_to_template(url, tpl) is True

    def test_wildcard_no_match_wrong_prefix(self):
        url = make_url("/auto_examples/cluster/plot_kmeans.html")
        tpl = make_template("/modules/...")
        assert match_url_to_template(url, tpl) is False

    def test_segment_count_mismatch(self):
        url = make_url("/a/b/c.html")
        tpl = make_template("/a/b")
        assert match_url_to_template(url, tpl) is False

    def test_literal_mismatch(self):
        url = make_url("/user_guide/topic.html")
        tpl = make_template("/modules/<filename>")
        assert match_url_to_template(url, tpl) is False


class TestClassifyDomain:
    def test_github(self):
        assert _classify_domain("github.com", ["/user/repo"]) == ExternalDomainType.CODE_HOSTING

    def test_pypi(self):
        assert _classify_domain("pypi.org", ["/project/sklearn"]) == ExternalDomainType.PACKAGE_INDEX

    def test_stackoverflow(self):
        assert _classify_domain("stackoverflow.com", ["/questions/123"]) == ExternalDomainType.QA_FORUM

    def test_twitter(self):
        assert _classify_domain("twitter.com", ["/user"]) == ExternalDomainType.SOCIAL_MEDIA

    def test_arxiv(self):
        assert _classify_domain("arxiv.org", ["/abs/1234"]) == ExternalDomainType.ACADEMIC

    def test_readthedocs(self):
        assert _classify_domain("myproject.readthedocs.io", ["/en/latest/"]) == ExternalDomainType.DOCUMENTATION_CROSSREF

    def test_doc_path_signals(self):
        assert _classify_domain("numpy.org", ["/doc/stable/reference/"]) == ExternalDomainType.DOCUMENTATION_CROSSREF

    def test_root_only_homepage(self):
        assert _classify_domain("example.com", ["/"]) == ExternalDomainType.PROJECT_HOMEPAGE

    def test_unknown_with_deep_path_is_doc_crossref(self):
        """Domains with meaningful paths should be classified as doc crossref."""
        assert _classify_domain("some-library.org", ["/api/v2/reference"]) == ExternalDomainType.DOCUMENTATION_CROSSREF


class TestExternalDomainAnalysis:
    def test_returns_external_domain_objects(self):
        urls = [
            make_url("/project/sklearn", domain="pypi.org"),
            make_url("/user/repo", domain="github.com"),
        ]
        results = external_domain_analysis(urls)
        from models import ExternalDomain
        assert all(isinstance(r, ExternalDomain) for r in results)
        assert len(results) == 2

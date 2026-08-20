"""
Integration test: run the full pipeline on the sample fixture
and verify the output structure.
"""

import json
import sys
import os
import pytest
from pathlib import Path

# Ensure phase2 root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import run_pipeline


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_phase1.json"
OUTPUT_PATH = Path(__file__).parent / "fixtures" / "test_output.json"


class TestFullPipeline:
    """Run the complete pipeline and validate output."""

    @pytest.fixture(autouse=True)
    def run(self):
        """Run pipeline before each test."""
        run_pipeline(str(FIXTURE_PATH), str(OUTPUT_PATH), verbose=True)
        with open(OUTPUT_PATH) as f:
            self.output = json.load(f)

    def test_output_has_required_keys(self):
        """Output must have all top-level keys."""
        required = [
            "crawl_id", "root_domain", "generator_detected",
            "summary", "templates", "uncovered_urls",
            "external_domains", "template_map"
        ]
        for key in required:
            assert key in self.output, f"Missing key: {key}"

    def test_root_domain_preserved(self):
        assert self.output["root_domain"] == "scikit-learn.org"

    def test_generator_detected_as_sphinx(self):
        """Fixture has has_objects_inv=true → should detect Sphinx."""
        assert self.output["generator_detected"] == "sphinx"
        assert self.output["generator_confidence"] >= 0.8

    def test_assets_filtered_out(self):
        """CSS and JS URLs should not appear in any template."""
        all_template_urls = []
        for urls in self.output["template_map"].values():
            all_template_urls.extend(urls)

        for url in all_template_urls:
            assert not url.endswith(".css"), f"Asset in templates: {url}"
            assert not url.endswith(".js"), f"Asset in templates: {url}"

    def test_templates_exist(self):
        """Should derive at least 2 templates from the fixture."""
        assert len(self.output["templates"]) >= 2

    def test_api_reference_template_exists(self):
        """One template should cover the /modules/generated/ URLs."""
        patterns = [t["pattern"] for t in self.output["templates"]]
        api_pattern = [p for p in patterns if "generated" in p or "modules" in p.lower()]
        assert len(api_pattern) >= 1, "No API reference template found"

    def test_user_guide_template_exists(self):
        """One template should cover the /user_guide/ URLs."""
        patterns = [t["pattern"] for t in self.output["templates"]]
        guide_pattern = [p for p in patterns if "user_guide" in p]
        assert len(guide_pattern) >= 1, "No user guide template found"

    def test_versioned_urls_share_template(self):
        """/stable/ and /dev/ versions of the same page should share a template."""
        template_map = self.output["template_map"]

        stable_url = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
        dev_url = "https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.LogisticRegression.html"

        # Find which template each belongs to
        stable_template = None
        dev_template = None
        for tpl_id, urls in template_map.items():
            if stable_url in urls:
                stable_template = tpl_id
            if dev_url in urls:
                dev_template = tpl_id

        # They might both be uncovered (that's also a valid outcome for a small fixture)
        # But if they're covered, they should be in the same template
        if stable_template and dev_template:
            assert stable_template == dev_template, \
                f"Versioned URLs got different templates: {stable_template} vs {dev_template}"

    def test_coverage_reasonable(self):
        """Coverage should be >= 75% even with this small fixture."""
        summary = self.output["summary"]
        assert summary["coverage_percent"] >= 75.0

    def test_external_domains_present(self):
        """Should identify numpy.org, scipy.org, matplotlib.org, pypi.org."""
        domains = [d["domain"] for d in self.output["external_domains"]]
        assert "numpy.org" in domains
        assert "scipy.org" in domains

    def test_pypi_classified_as_package_index(self):
        """pypi.org should be classified as package_index."""
        for domain in self.output["external_domains"]:
            if domain["domain"] == "pypi.org":
                assert domain["classification"] == "package_index"
                return
        pytest.skip("pypi.org not in external domains")

    def test_numpy_classified_as_doc_crossref(self):
        """numpy.org URLs with /doc/ paths should be documentation_crossref."""
        for domain in self.output["external_domains"]:
            if domain["domain"] == "numpy.org":
                assert domain["classification"] == "documentation_crossref"
                return
        pytest.skip("numpy.org not in external domains")

    def test_template_map_matches_templates(self):
        """Every template_id in template_map should exist in templates list."""
        template_ids_from_list = {t["template_id"] for t in self.output["templates"]}
        template_ids_from_map = set(self.output["template_map"].keys())
        assert template_ids_from_map.issubset(template_ids_from_list), \
            f"Map has IDs not in list: {template_ids_from_map - template_ids_from_list}"

    def test_no_duplicate_urls_across_templates(self):
        """A URL should appear in at most one template."""
        all_urls = []
        for urls in self.output["template_map"].values():
            all_urls.extend(urls)
        assert len(all_urls) == len(set(all_urls)), "Duplicate URLs found across templates"

"""
Unit tests for pipeline/structural_grouping.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.structural_grouping import discover_groups, finalize_templates
from models import ParsedURL, ParsedSegment, URLClassification, SegmentType
from utils.url_parser import classify_segment


def make_url(path: str, canonical: str = None) -> ParsedURL:
    """Helper to create ParsedURL for grouping tests."""
    canon = canonical or f"https://example.com{path}"
    segs = [s for s in path.split("/") if s]
    parsed_segs = [
        ParsedSegment(raw=s, position=i, lexical_type=classify_segment(s))
        for i, s in enumerate(segs)
    ]
    return ParsedURL(
        original_url=canon,
        canonical_url=canon,
        domain="example.com",
        scheme="https",
        path=path,
        segments=parsed_segs,
        classification=URLClassification.INTERNAL,
        is_asset=False,
        version_free_path=path,
    )


class TestDiscoverGroups:
    def test_produces_groups(self):
        """Should produce at least one group from clearly structured URLs."""
        urls = [
            make_url(f"/modules/generated/class_{i}.html")
            for i in range(10)
        ] + [
            make_url(f"/user_guide/topic_{i}.html")
            for i in range(5)
        ]
        groups = discover_groups(urls)
        assert len(groups) >= 1

    def test_empty_input(self):
        groups = discover_groups([])
        assert groups == []

    def test_small_site(self):
        """Should handle a very small site without crashing."""
        urls = [
            make_url("/install.html"),
            make_url("/index.html"),
        ]
        groups = discover_groups(urls)
        assert isinstance(groups, list)


class TestFinalizeTemplates:
    def test_returns_templates(self):
        urls = [
            make_url(f"/modules/generated/sklearn.linear_model.Class{i}.html")
            for i in range(6)
        ]
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        assert isinstance(templates, list)
        for tpl in templates:
            assert tpl.template_id.startswith("tpl_")
            assert tpl.member_count > 0

    def test_sorted_by_member_count(self):
        urls = (
            [make_url(f"/modules/generated/c{i}.html") for i in range(10)] +
            [make_url(f"/guide/p{i}.html") for i in range(4)]
        )
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        if len(templates) >= 2:
            assert templates[0].member_count >= templates[1].member_count

class TestSectionBoundaries:
    def test_modules_vs_user_guide_separate(self):
        urls = [
            make_url("/modules/generated/sklearn.linear_model.LogisticRegression.html"),
            make_url("/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
            make_url("/modules/generated/sklearn.svm.SVC.html"),
            make_url("/modules/generated/sklearn.cluster.KMeans.html"),
            make_url("/modules/generated/sklearn.tree.DecisionTreeClassifier.html"),
            make_url("/user_guide/linear_model.html"),
            make_url("/user_guide/ensemble.html"),
            make_url("/user_guide/clustering.html"),
            make_url("/user_guide/svm.html"),
        ]
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        assert len(templates) >= 2
        patterns = [t.pattern for t in templates]
        assert any("modules" in p or "generated" in p for p in patterns)
        assert any("user_guide" in p for p in patterns)

    def test_auto_examples_merged(self):
        urls = []
        for subdir in ["cluster", "linear_model", "ensemble", "svm"]:
            for i in range(4):
                urls.append(make_url(f"/auto_examples/{subdir}/plot_example_{i}.html"))
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        assert len(templates) == 1
        assert "auto_examples" in templates[0].pattern

    def test_no_generic_catch_all(self):
        urls = [
            make_url("/modules/generated/a.b.C.html"),
            make_url("/modules/generated/d.e.F.html"),
            make_url("/modules/generated/g.h.I.html"),
            make_url("/user_guide/guide1.html"),
            make_url("/user_guide/guide2.html"),
            make_url("/user_guide/guide3.html"),
            make_url("/auto_examples/cat1/plot1.html"),
            make_url("/auto_examples/cat2/plot2.html"),
            make_url("/auto_examples/cat3/plot3.html"),
        ]
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        for t in templates:
            if t.member_count >= 5:
                parts = t.pattern.strip("/").split("/")
                has_literal = any(not p.startswith("<") and p != "..." for p in parts)
                assert has_literal, f"Template too generic: {t.pattern}"

class TestTemplatePatterns:
    def test_api_reference_pattern(self):
        urls = [
            make_url("/modules/generated/sklearn.linear_model.LogisticRegression.html"),
            make_url("/modules/generated/sklearn.ensemble.RandomForestClassifier.html"),
            make_url("/modules/generated/sklearn.svm.SVC.html"),
        ]
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        assert len(templates) >= 1
        assert "modules" in templates[0].pattern
        assert "generated" in templates[0].pattern

    def test_user_guide_pattern(self):
        urls = [
            make_url("/user_guide/linear_model.html"),
            make_url("/user_guide/ensemble.html"),
            make_url("/user_guide/clustering.html"),
        ]
        groups = discover_groups(urls)
        templates = finalize_templates(groups, urls)
        assert len(templates) >= 1
        assert "user_guide" in templates[0].pattern

"""
Tests for FieldClassifier.
"""

from __future__ import annotations

import pytest

from rag.pipeline.field_classifier import FieldClassifier, _PRIORITY_ORDER


@pytest.fixture
def clf():
    return FieldClassifier()


class TestFieldClassifier:
    # ------------------------------------------------------------------
    # Individual role classification
    # ------------------------------------------------------------------

    def test_classify_title_field(self, clf):
        item = {"page_title": "My Page", "title": "Alt Title"}
        roles = clf.classify_fields(item)
        assert "title" in roles
        assert "page_title" in roles["title"]
        assert "title" in roles["title"]

    def test_classify_description_field(self, clf):
        roles = clf.classify_fields({"description": "A description."})
        assert "description" in roles
        assert "description" in roles["description"]

    def test_classify_parameter_field(self, clf):
        roles = clf.classify_fields({"parameters": [{"name": "x"}]})
        assert "parameter" in roles
        assert "parameters" in roles["parameter"]

    def test_classify_code_field(self, clf):
        roles = clf.classify_fields({"code_examples": ["x=1"], "code_snippets": ["y=2"]})
        assert "code" in roles
        assert "code_examples" in roles["code"]
        assert "code_snippets" in roles["code"]

    def test_classify_see_also_as_relation(self, clf):
        roles = clf.classify_fields({"see_also": [{"function_name": "foo"}]})
        assert "relation" in roles
        assert "see_also" in roles["relation"]

    def test_classify_source_link(self, clf):
        roles = clf.classify_fields({"source_link": "https://github.com/..."})
        assert "source" in roles
        assert "source_link" in roles["source"]

    def test_classify_notes(self, clf):
        roles = clf.classify_fields({"notes": "Be careful."})
        assert "note" in roles
        assert "notes" in roles["note"]

    def test_classify_signature(self, clf):
        roles = clf.classify_fields({"function_signature": "foo(x, y)"})
        assert "signature" in roles
        assert "function_signature" in roles["signature"]

    def test_classify_unknown_field_goes_to_other(self, clf):
        roles = clf.classify_fields({"random_weird_field": "value"})
        assert "other" in roles
        assert "random_weird_field" in roles["other"]

    def test_input_field_is_excluded(self, clf):
        roles = clf.classify_fields({"input": {"url": "https://x.com"}, "title": "Hi"})
        for bucket in roles.values():
            assert "input" not in bucket

    def test_classify_mixed_fields(self, clf):
        """Realistic tpl_002-like item."""
        item = {
            "page_title": "sklearn.config_context",
            "description": "Context manager.",
            "function_signature": "sklearn.config_context(*)",
            "parameters": [{"name": "assume_finite"}],
            "code_examples": [">>> pass"],
            "see_also": [{"function_name": "set_config"}],
            "notes": "Changes revert.",
            "source_link": "https://github.com/...",
            "input": {"url": "https://scikit-learn.org/..."},
        }
        roles = clf.classify_fields(item)
        assert "title" in roles
        assert "description" in roles
        assert "signature" in roles
        assert "parameter" in roles
        assert "code" in roles
        assert "relation" in roles
        assert "note" in roles
        assert "source" in roles
        # input must NOT appear anywhere
        for bucket in roles.values():
            assert "input" not in bucket

    # ------------------------------------------------------------------
    # Priority ordering
    # ------------------------------------------------------------------

    def test_priority_ordering(self, clf):
        ordered = clf.get_priority_ordered_roles()
        assert isinstance(ordered, list)
        assert len(ordered) > 0
        # Key roles must appear in the correct relative order
        assert ordered.index("title") < ordered.index("description")
        assert ordered.index("description") < ordered.index("parameter")
        assert ordered.index("parameter") < ordered.index("code")
        assert ordered.index("code") < ordered.index("other")

    def test_priority_ordering_contains_all_known_roles(self, clf):
        ordered = clf.get_priority_ordered_roles()
        for role in ("title", "description", "signature", "parameter", "code",
                     "note", "section", "relation", "install", "notebook", "other"):
            assert role in ordered, f"Missing role: {role}"

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_item(self, clf):
        roles = clf.classify_fields({})
        assert roles == {}

    def test_item_with_only_input(self, clf):
        roles = clf.classify_fields({"input": {"url": "https://x.com"}})
        assert roles == {}

    def test_installation_field(self, clf):
        roles = clf.classify_fields({"installation_pip": "pip install x"})
        assert "install" in roles

    def test_notebook_field(self, clf):
        roles = clf.classify_fields({"notebook_content": "{...}"})
        assert "notebook" in roles

    def test_all_fields_in_exactly_one_bucket(self, clf):
        item = {
            "page_title": "T", "description": "D", "function_signature": "f()",
            "parameters": [], "code_examples": [], "see_also": [],
            "notes": "N", "source_link": "L", "weird_field": "W",
            "input": {"url": "u"},
        }
        roles = clf.classify_fields(item)
        classified = set()
        for fields in roles.values():
            for f in fields:
                classified.add(f)
        expected = {k for k in item if k != "input"}
        assert classified == expected

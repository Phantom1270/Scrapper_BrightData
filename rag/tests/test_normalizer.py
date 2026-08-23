"""
Tests for UniversalNormalizer.
"""

from __future__ import annotations

import pytest

from rag.models.document import NormalizedDocument
from rag.pipeline.normalizer import UniversalNormalizer


@pytest.fixture
def normalizer():
    return UniversalNormalizer()


# ------------------------------------------------------------------
# Helper entries that mirror real Phase 3 output
# ------------------------------------------------------------------

API_ENTRY = {
    "url": "https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html",
    "status": "extracted",
    "data": [
        {
            "page_title": "sklearn.config_context",
            "description": "Context manager for global scikit-learn configuration.",
            "function_signature": "sklearn.config_context(*, assume_finite=False)",
            "parameters": [
                {"name": "assume_finite", "type_info": "bool", "description": "Skip validation."},
                {"name": "working_memory", "type_info": "int", "description": "Memory limit."},
            ],
            "code_examples": [">>> with sklearn.config_context(assume_finite=True):\n...     pass"],
            "see_also": [{"function_name": "set_config", "function_description": "Set config."}],
            "notes": "Changes revert after with block.",
            "source_link": "https://github.com/scikit-learn/sklearn",
            "input": {"url": "https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html"},
        }
    ],
}

TUTORIAL_ENTRY = {
    "url": "https://scikit-learn.org/stable/auto_examples/release_highlights.html",
    "status": "extracted",
    "data": [
        {
            "page_title": "Release Highlights",
            "description": "This page describes what is new in scikit-learn 1.4.",
            "introduction": "Scikit-learn 1.4 brings exciting new features.",
            "sections": [
                {"section_title": "New Features", "section_content": "This version adds quantile regression."},
                {"section_title": "Bug Fixes", "section_content": "Several stability improvements."},
            ],
            "code_examples": ["from sklearn.preprocessing import SplineTransformer"],
            "input": {"url": "..."},
        }
    ],
}

NOTEBOOK_ENTRY = {
    "url": "https://scikit-learn.org/stable/auto_examples/notebook.html",
    "status": "extracted",
    "data": [
        {
            "notebook_title": "Demo Notebook",
            "notebook_content": '{"cells": [{"cell_type": "markdown", "source": ["# Hello"]}, {"cell_type": "code", "source": ["import sklearn"]}]}',
            "input": {"url": "..."},
        }
    ],
}

ERROR_ENTRY = {
    "url": "https://scikit-learn.org/stable/dead.html",
    "status": "failed",
    "data": [],
    "error": "HTTP 404",
}

MINIMAL_ENTRY = {
    "url": "https://scikit-learn.org/stable/minimal.html",
    "status": "extracted",
    "data": [
        {"page_title": "Just a Title", "input": {"url": "..."}}
    ],
}

MULTI_DATA_ENTRY = {
    "url": "https://scikit-learn.org/stable/multi.html",
    "status": "extracted",
    "data": [
        {"page_title": "Item 1", "description": "First item."},
        {"page_title": "Item 2", "description": "Second item."},
        {"page_title": "Item 3", "description": "Third item."},
    ],
}


class TestNormalizerAPIReference:
    def test_normalize_api_reference_returns_document(self, normalizer):
        docs = normalizer.normalize_entry(API_ENTRY)
        assert len(docs) == 1

    def test_api_reference_title(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        assert "config_context" in doc.title

    def test_api_reference_description(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        assert "context manager" in doc.description.lower()

    def test_api_reference_content_type(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        assert doc.content_type == "api_reference"

    def test_api_reference_has_signature_block(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        sig_blocks = [b for b in doc.content_blocks if b.block_type == "function_signature"]
        assert len(sig_blocks) >= 1

    def test_api_reference_has_parameter_blocks(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        param_blocks = [b for b in doc.content_blocks if b.block_type == "parameter_list"]
        assert len(param_blocks) >= 1

    def test_api_reference_has_code_blocks(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        code_blocks = [b for b in doc.content_blocks if b.block_type == "code"]
        assert len(code_blocks) >= 1

    def test_api_reference_has_relation_block(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        # see_also → prose block with "set_config"
        prose_blocks = [b for b in doc.content_blocks if b.block_type == "prose"]
        combined = " ".join(b.text for b in prose_blocks)
        assert "set_config" in combined

    def test_api_reference_source_link(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        assert doc.source_link and "github" in doc.source_link

    def test_api_reference_input_not_in_metadata(self, normalizer):
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        assert "input" not in doc.metadata

    def test_content_blocks_ordered_by_priority(self, normalizer):
        """Signature should precede description which should precede parameter blocks."""
        doc = normalizer.normalize_entry(API_ENTRY)[0]
        block_types = [b.block_type for b in doc.content_blocks]
        # Find first occurrence of each
        sig_idx = next((i for i, t in enumerate(block_types) if t == "function_signature"), 999)
        param_idx = next((i for i, t in enumerate(block_types) if t == "parameter_list"), 999)
        code_idx = next((i for i, t in enumerate(block_types) if t == "code"), 999)
        # Signature must come before parameter and code
        assert sig_idx < param_idx
        assert param_idx < code_idx


class TestNormalizerTutorial:
    def test_tutorial_content_type(self, normalizer):
        doc = normalizer.normalize_entry(TUTORIAL_ENTRY)[0]
        assert doc.content_type == "tutorial"

    def test_tutorial_has_section_blocks(self, normalizer):
        doc = normalizer.normalize_entry(TUTORIAL_ENTRY)[0]
        # Sections become prose blocks
        prose = [b for b in doc.content_blocks if b.block_type == "prose"]
        combined = " ".join(b.text for b in prose)
        assert "quantile" in combined.lower() or "new features" in combined.lower()

    def test_tutorial_has_code_blocks(self, normalizer):
        doc = normalizer.normalize_entry(TUTORIAL_ENTRY)[0]
        code = [b for b in doc.content_blocks if b.block_type == "code"]
        assert len(code) >= 1

    def test_tutorial_introduction_captured(self, normalizer):
        doc = normalizer.normalize_entry(TUTORIAL_ENTRY)[0]
        prose = [b for b in doc.content_blocks if b.block_type == "prose"]
        combined = " ".join(b.text for b in prose)
        assert "exciting" in combined.lower() or "new features" in combined.lower()


class TestNormalizerNotebook:
    def test_notebook_content_type(self, normalizer):
        doc = normalizer.normalize_entry(NOTEBOOK_ENTRY)[0]
        assert doc.content_type == "notebook"

    def test_notebook_has_prose_from_markdown(self, normalizer):
        doc = normalizer.normalize_entry(NOTEBOOK_ENTRY)[0]
        prose = [b for b in doc.content_blocks if b.block_type == "prose"]
        assert len(prose) >= 1
        assert any("hello" in b.text.lower() for b in prose)

    def test_notebook_has_code_from_code_cell(self, normalizer):
        doc = normalizer.normalize_entry(NOTEBOOK_ENTRY)[0]
        code = [b for b in doc.content_blocks if b.block_type == "code"]
        assert len(code) >= 1
        assert any("sklearn" in b.text for b in code)


class TestNormalizerEdgeCases:
    def test_normalize_error_entry(self, normalizer):
        docs = normalizer.normalize_entry(ERROR_ENTRY)
        # Error entry has empty data — handled by normalize_file, not normalize_entry
        assert docs == []

    def test_normalize_empty_data(self, normalizer):
        entry = {"url": "https://x.com", "status": "extracted", "data": []}
        docs = normalizer.normalize_entry(entry)
        assert docs == []

    def test_normalize_handles_missing_fields(self, normalizer):
        docs = normalizer.normalize_entry(MINIMAL_ENTRY)
        assert len(docs) == 1
        doc = docs[0]
        assert "Just a Title" in doc.title
        # Should not crash even with minimal fields

    def test_normalize_preserves_metadata(self, normalizer):
        entry = {
            "url": "https://x.com",
            "status": "extracted",
            "data": [
                {
                    "page_title": "Test",
                    "version": "1.4",            # short — should go to metadata
                    "release_date": "2024-01-01", # short — should go to metadata
                    "input": {"url": "..."},
                }
            ],
        }
        doc = normalizer.normalize_entry(entry)[0]
        # Short scalar fields that don't match any role should be in metadata
        assert "version" in doc.metadata or "release_date" in doc.metadata

    def test_normalize_multiple_data_items(self, normalizer):
        docs = normalizer.normalize_entry(MULTI_DATA_ENTRY)
        assert len(docs) == 3

    def test_normalize_doc_id_deterministic(self, normalizer):
        docs1 = normalizer.normalize_entry(MINIMAL_ENTRY)
        docs2 = normalizer.normalize_entry(MINIMAL_ENTRY)
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_normalize_file_returns_docs(self, normalizer, sample_scraped_json):
        docs = normalizer.normalize_file(str(sample_scraped_json))
        assert len(docs) > 0

    def test_normalize_file_stamps_template_id(self, normalizer, sample_scraped_json):
        docs = normalizer.normalize_file(str(sample_scraped_json))
        live = [d for d in docs if not d.error]
        for doc in live:
            assert doc.template_id in ("tpl_002", "tpl_005")

    def test_normalize_file_not_found(self, normalizer):
        with pytest.raises(FileNotFoundError):
            normalizer.normalize_file("/no/such/file.json")

    def test_normalize_file_stats(self, normalizer, sample_scraped_json):
        normalizer.normalize_file(str(sample_scraped_json))
        stats = normalizer.get_stats()
        assert "total_entries" in stats
        assert "total_docs" in stats
        assert stats["total_entries"] > 0

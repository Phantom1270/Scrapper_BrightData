"""
Tests for DataPipeline (end-to-end orchestrator).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.pipeline.data_pipeline import DataPipeline
from rag.storage.sqlite_store import SQLiteStore


@pytest.fixture
def pipeline(tmp_path):
    """DataPipeline with an in-memory SQLite store."""
    store = SQLiteStore(db_path=":memory:")
    return DataPipeline(store=store)


@pytest.fixture
def empty_scraped_json(tmp_path) -> Path:
    payload = {
        "domain": "empty.org",
        "total_processed": 0,
        "total_healed": 0,
        "failed": 0,
        "results": {},
    }
    p = tmp_path / "empty.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def dedup_scraped_json(tmp_path) -> Path:
    """JSON with 2 identical entries to test dedup reporting."""
    identical_data = {
        "page_title": "Identical Page",
        "description": "This page has the same content as another.",
        "function_signature": "identical_func(x)",
        "parameters": [{"name": "x", "type_info": "int", "description": "Input value."}],
        "code_examples": [">>> identical_func(1)"],
        "input": {"url": "https://x.com/a"},
    }
    payload = {
        "domain": "test.org",
        "total_processed": 2,
        "total_healed": 0,
        "failed": 0,
        "results": {
            "tpl_001": [
                {"url": "https://x.com/a", "status": "extracted", "data": [identical_data]},
                {"url": "https://x.com/b", "status": "extracted", "data": [identical_data]},
            ]
        },
    }
    p = tmp_path / "dedup_test.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


class TestDataPipelineEndToEnd:
    def test_pipeline_end_to_end(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))

        # Report must have all expected keys
        expected_keys = [
            "input_file", "domain", "total_raw_entries",
            "total_normalized", "total_after_cleaning", "total_after_dedup",
            "errors_skipped", "empties_skipped",
            "exact_duplicates_removed", "near_duplicates_removed",
            "by_template", "by_content_type",
            "storage_backend", "documents_stored",
        ]
        for key in expected_keys:
            assert key in report, f"Missing key: {key}"

    def test_pipeline_domain_populated(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        assert report["domain"] == "scikit-learn.org"

    def test_pipeline_documents_stored(self, pipeline, sample_scraped_json):
        store = pipeline._store
        pipeline.run(str(sample_scraped_json))
        # At least some documents should be stored
        assert store.count_documents() > 0

    def test_pipeline_stored_count_matches_report(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        assert pipeline._store.count_documents() == report["documents_stored"]

    def test_pipeline_stored_docs_have_content(self, pipeline, sample_scraped_json):
        pipeline.run(str(sample_scraped_json))
        docs = pipeline._store.get_all_documents()
        live_docs = [d for d in docs if not d.error]
        for doc in live_docs:
            assert len(doc.content_blocks) > 0, f"Doc {doc.url} has no content blocks"

    def test_pipeline_all_templates_represented(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        assert "tpl_002" in report["by_template"]
        assert "tpl_005" in report["by_template"]

    def test_pipeline_by_template_has_counts(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        for tpl_id, tdata in report["by_template"].items():
            assert "input" in tdata
            assert "output" in tdata
            assert "errors" in tdata

    def test_pipeline_with_empty_input(self, pipeline, empty_scraped_json):
        """Pipeline should complete without error on empty input."""
        report = pipeline.run(str(empty_scraped_json))
        assert report["total_raw_entries"] == 0
        assert report["documents_stored"] == 0

    def test_pipeline_report_accuracy(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        # After dedup the stored count should equal total_after_dedup + error docs
        stored = report["documents_stored"]
        deduped = report["total_after_dedup"]
        errors = report["errors_skipped"]
        # Stored = live (deduped) + error docs carried through
        assert stored >= deduped  # errors may be stored too

    def test_pipeline_dedup_in_report(self, pipeline, dedup_scraped_json):
        """Pipeline with known duplicates should show dedup in report."""
        report = pipeline.run(str(dedup_scraped_json))
        total_removed = (
            report["exact_duplicates_removed"] + report["near_duplicates_removed"]
        )
        # At least one of the two identical entries should be removed
        assert total_removed >= 1

    def test_pipeline_file_not_found(self, pipeline):
        with pytest.raises(FileNotFoundError):
            pipeline.run("/nonexistent/path/output.json")

    def test_get_processing_report(self, pipeline, sample_scraped_json):
        pipeline.run(str(sample_scraped_json))
        report = pipeline.get_processing_report()
        assert "domain" in report
        assert "documents_stored" in report

    def test_pipeline_storage_backend_name_in_report(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        assert "SQLiteStore" in report["storage_backend"]

    def test_pipeline_by_content_type(self, pipeline, sample_scraped_json):
        report = pipeline.run(str(sample_scraped_json))
        # api_reference documents should be detected from tpl_002
        ct = report["by_content_type"]
        assert isinstance(ct, dict)
        # At least one content type should be present
        assert sum(ct.values()) > 0

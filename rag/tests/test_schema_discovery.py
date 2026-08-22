"""
Tests for SchemaDiscovery.
"""

from __future__ import annotations

import json

import pytest

from rag.pipeline.schema_discovery import SchemaDiscovery


class TestSchemaDiscovery:
    def test_discover_basic_structure(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        assert report["domain"] == "scikit-learn.org"
        assert report["total_processed"] == 6
        assert "tpl_002" in report["templates"]
        assert "tpl_005" in report["templates"]

    def test_discover_template_entry_counts(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        tpl002 = report["templates"]["tpl_002"]
        assert tpl002["entry_count"] == 3
        assert tpl002["extracted_count"] == 2
        assert tpl002["error_count"] == 1

    def test_discover_fields_per_template(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        fields = report["templates"]["tpl_002"]["fields"]
        assert "page_title" in fields
        assert "description" in fields
        assert "parameters" in fields
        assert "code_examples" in fields

    def test_discover_field_types(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        page_title_info = report["templates"]["tpl_002"]["fields"]["page_title"]
        assert "string" in page_title_info["types"]

        params_info = report["templates"]["tpl_002"]["fields"]["parameters"]
        assert "array" in params_info["types"]
        assert params_info["is_array"] is True

    def test_discover_array_item_fields(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        params_info = report["templates"]["tpl_002"]["fields"]["parameters"]
        assert "name" in params_info["array_item_fields"]
        assert "type_info" in params_info["array_item_fields"]
        assert "description" in params_info["array_item_fields"]

    def test_discover_error_entries(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        tpl002 = report["templates"]["tpl_002"]
        assert tpl002["error_count"] == 1
        assert tpl002["error_types"]  # at least one error type recorded

    def test_discover_sample_values(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        page_title_info = report["templates"]["tpl_002"]["fields"]["page_title"]
        assert len(page_title_info["sample_values"]) > 0
        assert any("sklearn" in sv or "set_config" in sv
                   for sv in page_title_info["sample_values"])

    def test_discover_content_vs_metadata_fields(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        tpl002 = report["templates"]["tpl_002"]
        # Content fields should include the known content fields
        assert "page_title" in tpl002["content_fields"] or "description" in tpl002["content_fields"]

    def test_discover_empty_file(self, tmp_path):
        empty_json = {
            "domain": "empty.org",
            "total_processed": 0,
            "total_healed": 0,
            "failed": 0,
            "results": {},
        }
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(empty_json), encoding="utf-8")

        sd = SchemaDiscovery(str(p))
        report = sd.discover()

        assert report["domain"] == "empty.org"
        assert report["templates"] == {}

    def test_discover_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            SchemaDiscovery("/nonexistent/path/output.json")

    def test_print_report_does_not_crash(self, sample_scraped_json, capsys):
        sd = SchemaDiscovery(str(sample_scraped_json))
        sd.discover()
        sd.print_report()  # should not raise
        out = capsys.readouterr().out
        assert "scikit-learn.org" in out

    def test_discover_second_template(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()

        tpl005 = report["templates"]["tpl_005"]
        assert tpl005["entry_count"] == 3
        assert tpl005["extracted_count"] == 3
        assert tpl005["error_count"] == 0
        assert "page_title" in tpl005["fields"]

    def test_discover_report_has_all_top_keys(self, sample_scraped_json):
        sd = SchemaDiscovery(str(sample_scraped_json))
        report = sd.discover()
        for key in ("domain", "total_processed", "total_healed", "failed", "templates"):
            assert key in report

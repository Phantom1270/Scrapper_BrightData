"""
Integration tests for Phase 3 pipeline.
No real API calls. Tests logic only.
"""

import json
import pytest
from pathlib import Path

from models import (
    Phase2Output, FetchedPage, ExtractedRecord,
    RecordStatus, ValidationSchema, FieldSchema,
    FieldImportance, FieldType,
)
from pipeline.sampling import select_samples
from pipeline.extraction import extract_single
from pipeline.validation import validate_single


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_phase2.json"


class TestSampling:
    def load_phase2(self):
        with open(FIXTURE_PATH) as f:
            return Phase2Output(**json.load(f))

    def test_selects_samples(self):
        phase2 = self.load_phase2()
        samples = select_samples(phase2, samples_per_template=3)
        assert len(samples) == 2
        for tid, urls in samples.items():
            assert len(urls) >= 1
            assert len(urls) <= 3

    def test_no_duplicate_samples(self):
        phase2 = self.load_phase2()
        samples = select_samples(phase2, samples_per_template=3)
        for urls in samples.values():
            assert len(urls) == len(set(urls))

    def test_samples_belong_to_template(self):
        phase2 = self.load_phase2()
        samples = select_samples(phase2, samples_per_template=3)
        for tid, urls in samples.items():
            valid_urls = set(phase2.template_map.get(tid, []))
            for url in urls:
                assert url in valid_urls


class TestExtraction:
    def test_extract_with_matching_html(self):
        html = """
        <html><body>
            <h1>LogisticRegression</h1>
            <p>A logistic regression classifier.</p>
        </body></html>
        """
        schema = ValidationSchema(
            template_id="tpl_001",
            template_pattern="/modules/generated/<filename>",
            fields=[
                FieldSchema(
                    name="title", field_type=FieldType.TEXT,
                    importance=FieldImportance.REQUIRED, css_selector="h1",
                ),
                FieldSchema(
                    name="description", field_type=FieldType.TEXT,
                    importance=FieldImportance.REQUIRED, css_selector="p",
                ),
            ],
        )
        page = FetchedPage(
            url="https://example.com/modules/generated/LR.html",
            template_id="tpl_001",
            html=html,
            fetch_success=True,
        )
        record = extract_single(page, schema)
        assert record.data["title"] == "LogisticRegression"
        assert "logistic" in record.data["description"].lower()

    def test_extract_with_no_html(self):
        schema = ValidationSchema(
            template_id="tpl_001",
            template_pattern="/test",
            fields=[
                FieldSchema(
                    name="title", field_type=FieldType.TEXT,
                    importance=FieldImportance.REQUIRED, css_selector="h1",
                ),
            ],
        )
        page = FetchedPage(
            url="https://example.com/fail",
            template_id="tpl_001",
            html=None,
            fetch_success=False,
        )
        record = extract_single(page, schema)
        assert record.status == RecordStatus.FAILED

    def test_extract_with_partial_html(self):
        html = "<html><body><h1>Title Only</h1></body></html>"
        schema = ValidationSchema(
            template_id="tpl_001",
            template_pattern="/test",
            fields=[
                FieldSchema(
                    name="title", field_type=FieldType.TEXT,
                    importance=FieldImportance.REQUIRED, css_selector="h1",
                ),
                FieldSchema(
                    name="description", field_type=FieldType.TEXT,
                    importance=FieldImportance.REQUIRED, css_selector=".desc",
                ),
            ],
        )
        page = FetchedPage(
            url="https://example.com/partial",
            template_id="tpl_001",
            html=html,
            fetch_success=True,
        )
        record = extract_single(page, schema)
        assert record.data["title"] == "Title Only"
        assert record.data["description"] is None
        assert record.fields_missing >= 1


class TestValidation:
    def test_pass_when_all_required_present(self):
        schema = ValidationSchema(
            template_id="tpl_001",
            template_pattern="/test",
            fields=[
                FieldSchema(name="title", importance=FieldImportance.REQUIRED),
                FieldSchema(name="desc", importance=FieldImportance.REQUIRED),
            ],
        )
        record = ExtractedRecord(
            url="https://example.com/test",
            template_id="tpl_001",
            data={"title": "X", "desc": "Y"},
        )
        vr = validate_single(record, schema)
        assert vr.passed is True

    def test_fail_when_required_missing(self):
        schema = ValidationSchema(
            template_id="tpl_001",
            template_pattern="/test",
            fields=[
                FieldSchema(name="title", importance=FieldImportance.REQUIRED),
                FieldSchema(name="desc", importance=FieldImportance.REQUIRED),
            ],
        )
        record = ExtractedRecord(
            url="https://example.com/test",
            template_id="tpl_001",
            data={"title": "X"},
        )
        vr = validate_single(record, schema)
        assert vr.passed is False
        assert "desc" in vr.missing_required

from pipeline.validation import validate_single
from models import (
    ExtractedRecord, ValidationSchema, FieldSchema,
    FieldImportance, FieldType, RecordStatus,
)


def make_schema(required_names, optional_names=None):
    fields = []
    for name in required_names:
        fields.append(FieldSchema(name=name, importance=FieldImportance.REQUIRED))
    for name in (optional_names or []):
        fields.append(FieldSchema(name=name, importance=FieldImportance.OPTIONAL))
    return ValidationSchema(
        template_id="test", template_pattern="/test", fields=fields,
    )


def make_record(data):
    return ExtractedRecord(
        url="https://example.com/test",
        template_id="test",
        status=RecordStatus.EXTRACTED,
        data=data,
    )


class TestValidation:
    def test_all_required_present(self):
        schema = make_schema(["title", "description"])
        record = make_record({"title": "Hello", "description": "World"})
        vr = validate_single(record, schema)
        assert vr.passed is True

    def test_required_missing(self):
        schema = make_schema(["title", "description"])
        record = make_record({"title": "Hello"})
        vr = validate_single(record, schema)
        assert vr.passed is False
        assert "description" in vr.missing_required

    def test_required_empty_string(self):
        schema = make_schema(["title", "description"])
        record = make_record({"title": "Hello", "description": ""})
        vr = validate_single(record, schema)
        assert vr.passed is False
        assert "description" in vr.empty_required

    def test_required_empty_list(self):
        schema = make_schema(["title", "params"])
        record = make_record({"title": "Hello", "params": []})
        vr = validate_single(record, schema)
        assert vr.passed is False
        assert "params" in vr.empty_required

    def test_optional_missing_still_passes(self):
        schema = make_schema(["title"], ["image", "author"])
        record = make_record({"title": "Hello"})
        vr = validate_single(record, schema)
        assert vr.passed is True
        assert "image" in vr.missing_optional

    def test_score_partial(self):
        schema = make_schema(["title", "description", "author"])
        record = make_record({"title": "Hello", "description": "World"})
        vr = validate_single(record, schema)
        assert 0.6 <= vr.score <= 0.7

    def test_score_perfect(self):
        schema = make_schema(["title", "description"])
        record = make_record({"title": "Hello", "description": "World"})
        vr = validate_single(record, schema)
        assert vr.score == 1.0

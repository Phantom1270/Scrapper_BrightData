"""
Tests for quality_auditor.py
"""

from rag.chunking.quality_auditor import ChunkQualityAuditor
from rag.models.chunk import Chunk


class TestChunkQualityAuditor:
    def test_audit_counts_total_chunks(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["A"], chunk_index=0, token_count=150, metadata={"document_content_type": "api_reference"}),
            Chunk(chunk_id="c2", doc_id="d1", url="http://test.com", content="2", content_type="code", heading_path=["B"], chunk_index=1, token_count=200, metadata={"document_content_type": "api_reference"})
        ]
        report = auditor.audit(chunks)
        assert report.total_chunks == 2
        assert report.avg_tokens == 175.0
        assert report.by_content_type["prose"] == 1
        assert report.by_content_type["code"] == 1
        assert report.chunks_per_document["d1"] == 2

    def test_audit_computes_average_tokens(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["A"], chunk_index=0, token_count=100),
            Chunk(chunk_id="c2", doc_id="d1", url="http://test.com", content="2", content_type="prose", heading_path=["B"], chunk_index=1, token_count=300)
        ]
        report = auditor.audit(chunks)
        assert report.avg_tokens == 200.0
        assert report.min_tokens == 100
        assert report.max_tokens == 300

    def test_audit_detects_oversized_chunks(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["A"], chunk_index=0, token_count=600)
        ]
        report = auditor.audit(chunks)
        assert len(report.oversized_chunks) == 1
        assert report.oversized_chunks[0]["chunk_id"] == "c1"

    def test_audit_detects_tiny_chunks(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=["A"], chunk_index=0, token_count=50)
        ]
        report = auditor.audit(chunks)
        assert len(report.tiny_chunks) == 1
        assert report.tiny_chunks[0]["chunk_id"] == "c1"

    def test_audit_detects_no_heading_chunks(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=[], chunk_index=0, token_count=150)
        ]
        report = auditor.audit(chunks)
        assert len(report.no_heading_chunks) == 1

    def test_audit_generates_issue_descriptions(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        chunks = [
            Chunk(chunk_id="c1", doc_id="d1", url="http://test.com", content="1", content_type="prose", heading_path=[], chunk_index=0, token_count=50, metadata={"document_content_type": "unknown"})
        ]
        report = auditor.audit(chunks)
        issues_text = " ".join(report.issues)
        assert "under min_tokens" in issues_text
        assert "empty heading_path" in issues_text
        assert "only 1 chunk" in issues_text
        assert "content_type='unknown'" in issues_text

    def test_audit_handles_empty_chunk_list(self):
        auditor = ChunkQualityAuditor({"max_tokens": 512, "min_tokens": 100})
        report = auditor.audit([])
        assert report.total_chunks == 0
        assert report.avg_tokens == 0.0

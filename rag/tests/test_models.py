"""
Tests for all RAG data models.

Covers: ContentBlock, NormalizedDocument, Chunk, RetrievalResult,
        QueryRequest, QueryResponse.
"""

from __future__ import annotations

import pytest

from rag.models.chunk import Chunk
from rag.models.document import ContentBlock, NormalizedDocument
from rag.models.query import QueryRequest, QueryResponse
from rag.models.retrieval import RetrievalResult
from rag.utils.ids import generate_chunk_id, generate_doc_id


# ---------------------------------------------------------------------------
# ContentBlock
# ---------------------------------------------------------------------------


class TestContentBlock:
    def test_content_block_creation(self):
        block = ContentBlock(
            block_type="prose",
            text="Hello world",
            heading="Intro",
            language="",
        )
        assert block.block_type == "prose"
        assert block.text == "Hello world"
        assert block.heading == "Intro"
        assert block.language == ""
        assert block.structured_data is None

    def test_content_block_to_dict_roundtrip(self):
        block = ContentBlock(
            block_type="code",
            text="print('hello')",
            heading="Example",
            language="python",
            structured_data={"key": "value"},
        )
        d = block.to_dict()
        restored = ContentBlock.from_dict(d)
        assert block == restored

    def test_content_block_all_valid_types(self):
        valid = ["prose", "code", "table", "parameter_list",
                 "function_signature", "note", "example", "unknown"]
        for t in valid:
            block = ContentBlock(block_type=t, text="text")
            assert block.block_type == t

    def test_content_block_rejects_invalid_type(self):
        with pytest.raises(ValueError, match="not valid"):
            ContentBlock(block_type="invalid_type", text="text")

    def test_content_block_rejects_empty_type(self):
        with pytest.raises(ValueError):
            ContentBlock(block_type="", text="text")

    def test_content_block_defaults(self):
        block = ContentBlock(block_type="prose", text="text")
        assert block.heading == ""
        assert block.language == ""
        assert block.structured_data is None


# ---------------------------------------------------------------------------
# NormalizedDocument
# ---------------------------------------------------------------------------


class TestNormalizedDocument:
    def _make_doc(self, **overrides) -> NormalizedDocument:
        defaults = dict(
            doc_id=generate_doc_id("https://example.com/page"),
            url="https://example.com/page",
            title="Test Page",
            description="A test page.",
            content_blocks=[
                ContentBlock(block_type="prose", text="Hello world"),
                ContentBlock(block_type="code", text="print(1)", language="python"),
            ],
            metadata={"version": "1.0"},
            template_id="tpl_001",
            content_type="api_reference",
        )
        defaults.update(overrides)
        return NormalizedDocument(**defaults)

    def test_normalized_document_creation(self):
        doc = self._make_doc()
        assert doc.title == "Test Page"
        assert len(doc.content_blocks) == 2
        assert doc.error is None
        assert doc.source_link is None

    def test_normalized_document_full_text(self):
        doc = self._make_doc()
        full = doc.full_text
        assert "Hello world" in full
        assert "print(1)" in full

    def test_normalized_document_to_dict_roundtrip(self):
        doc = self._make_doc()
        d = doc.to_dict()
        restored = NormalizedDocument.from_dict(d)
        assert doc == restored

    def test_normalized_document_to_dict_roundtrip_with_error(self):
        doc = self._make_doc(
            content_blocks=[],
            content_type="unknown",
            error="HTTP 404",
        )
        d = doc.to_dict()
        restored = NormalizedDocument.from_dict(d)
        assert restored.error == "HTTP 404"

    def test_normalized_document_to_dict_has_all_keys(self):
        doc = self._make_doc()
        d = doc.to_dict()
        for key in ["doc_id", "url", "title", "description", "content_blocks",
                    "metadata", "template_id", "content_type"]:
            assert key in d

    def test_invalid_document_rejects_empty_doc_id(self):
        with pytest.raises(ValueError, match="doc_id"):
            self._make_doc(doc_id="")

    def test_invalid_document_rejects_empty_url(self):
        with pytest.raises(ValueError, match="url"):
            self._make_doc(url="")

    def test_invalid_document_rejects_bad_content_type(self):
        with pytest.raises(ValueError, match="content_type"):
            self._make_doc(content_type="blog_post")

    def test_all_valid_content_types(self):
        for ct in ["api_reference", "tutorial", "notebook", "example", "unknown"]:
            doc = self._make_doc(content_type=ct)
            assert doc.content_type == ct


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def _make_chunk(self, **overrides) -> Chunk:
        doc_id = generate_doc_id("https://example.com/doc")
        defaults = dict(
            chunk_id=generate_chunk_id(doc_id, 0),
            doc_id=doc_id,
            url="https://example.com/doc",
            content="This is a chunk.",
            content_type="prose",
            heading_path=["Section A", "Subsection B"],
            chunk_index=0,
            token_count=5,
        )
        defaults.update(overrides)
        return Chunk(**defaults)

    def test_chunk_creation(self):
        chunk = self._make_chunk()
        assert chunk.content == "This is a chunk."
        assert chunk.chunk_index == 0
        assert chunk.token_count == 5
        assert chunk.parent_chunk_id is None

    def test_chunk_content_with_heading_property(self):
        chunk = self._make_chunk()
        cwh = chunk.content_with_heading
        assert "Section A" in cwh
        assert "Subsection B" in cwh
        assert "This is a chunk." in cwh
        assert ">" in cwh  # breadcrumb separator

    def test_chunk_content_with_heading_no_path(self):
        chunk = self._make_chunk(heading_path=[])
        assert chunk.content_with_heading == "This is a chunk."

    def test_chunk_to_dict_roundtrip(self):
        chunk = self._make_chunk(
            language="python",
            block_type="code",
            metadata={"source": "test"},
        )
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert chunk == restored

    def test_chunk_rejects_empty_chunk_id(self):
        with pytest.raises(ValueError, match="chunk_id"):
            self._make_chunk(chunk_id="")

    def test_chunk_rejects_empty_doc_id(self):
        with pytest.raises(ValueError, match="doc_id"):
            self._make_chunk(doc_id="")

    def test_chunk_rejects_negative_index(self):
        with pytest.raises(ValueError, match="chunk_index"):
            self._make_chunk(chunk_index=-1)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_retrieval_result_creation(self):
        result = RetrievalResult(
            chunk_id="abc123",
            content="This is relevant.",
            url="https://example.com",
            heading_path=["Docs", "API"],
            content_type="prose",
            score=0.87,
            source="vector",
        )
        assert result.score == 0.87
        assert result.source == "vector"

    def test_retrieval_result_all_valid_sources(self):
        for src in ["vector", "bm25", "hybrid", "reranked"]:
            r = RetrievalResult("id", "text", "https://x.com", [], "prose", 0.5, src)
            assert r.source == src

    def test_retrieval_result_rejects_invalid_source(self):
        with pytest.raises(ValueError, match="source"):
            RetrievalResult("id", "text", "https://x.com", [], "prose", 0.5, "unknown_source")

    def test_retrieval_result_to_dict_roundtrip(self):
        r = RetrievalResult(
            chunk_id="abc",
            content="text",
            url="https://example.com",
            heading_path=["A", "B"],
            content_type="code",
            score=0.9,
            source="hybrid",
            metadata={"k": "v"},
        )
        assert RetrievalResult.from_dict(r.to_dict()) == r


# ---------------------------------------------------------------------------
# QueryRequest / QueryResponse
# ---------------------------------------------------------------------------


class TestQueryModels:
    def test_query_request_defaults(self):
        req = QueryRequest(question="What is config_context?")
        assert req.top_k == 5
        assert req.filter_content_type is None
        assert req.filter_doc_id is None
        assert req.use_reranking is True

    def test_query_request_rejects_empty_question(self):
        with pytest.raises(ValueError, match="question"):
            QueryRequest(question="")

    def test_query_request_rejects_whitespace_question(self):
        with pytest.raises(ValueError):
            QueryRequest(question="   ")

    def test_query_request_rejects_zero_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            QueryRequest(question="valid?", top_k=0)

    def test_query_request_to_dict_roundtrip(self):
        req = QueryRequest(
            question="What is LinearRegression?",
            top_k=3,
            filter_content_type="api_reference",
            use_reranking=False,
        )
        restored = QueryRequest.from_dict(req.to_dict())
        assert restored.question == req.question
        assert restored.top_k == req.top_k

    def test_query_response_creation(self):
        resp = QueryResponse(
            answer="LinearRegression fits a linear model.",
            sources=[{"chunk_id": "abc", "url": "https://x.com", "score": 0.9}],
            retrieval_time_ms=12.5,
            generation_time_ms=200.0,
            total_time_ms=212.5,
        )
        assert resp.cached is False
        assert len(resp.sources) == 1

    def test_query_response_to_dict_roundtrip(self):
        resp = QueryResponse(
            answer="An answer.",
            sources=[],
            retrieval_time_ms=5.0,
            generation_time_ms=100.0,
            total_time_ms=105.0,
            cached=True,
        )
        assert QueryResponse.from_dict(resp.to_dict()) == resp

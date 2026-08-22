"""
Integration tests for the ChunkingEngine.
"""

import json
import pytest

from rag.chunking.engine import ChunkingEngine
from rag.models.document import NormalizedDocument, ContentBlock
from rag.storage.sqlite_store import SQLiteStore


class TestChunkingEngine:
    @pytest.fixture
    def store_with_docs(self, tmp_store: SQLiteStore, sample_document: NormalizedDocument, sample_document_tutorial: NormalizedDocument):
        # The prompt mentioned they are saved in conftest, but looking at conftest, they aren't saved automatically.
        # So we save them here to be sure.
        tmp_store.save_documents([sample_document, sample_document_tutorial])
        return tmp_store
        
    @pytest.fixture
    def engine(self, store_with_docs: SQLiteStore):
        # We need a small config for fast testing, or just use defaults
        config = {
            "max_tokens": 512,
            "min_tokens": 100,
            "overlap_tokens": 75,
            "encoding_name": "cl100k_base",
            "parent_max_tokens": 1500
        }
        return ChunkingEngine(config=config, store=store_with_docs)

    def test_engine_chunks_single_document(self, engine: ChunkingEngine, sample_document: NormalizedDocument):
        chunks = engine.chunk_document(sample_document)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.doc_id == sample_document.doc_id
            
    def test_engine_selects_correct_strategy(self, engine: ChunkingEngine, sample_document: NormalizedDocument):
        # sample_document is api_reference, so it should use ApiReferenceChunkingStrategy
        # We can verify this by checking if signature is atomic and has correct heading
        chunks = engine.chunk_document(sample_document)
        sig_chunks = [c for c in chunks if c.content_type == "function_signature"]
        assert len(sig_chunks) == 1
        assert sig_chunks[0].heading_path == [sample_document.title, "Signature"]

    def test_engine_chunks_all_documents(self, engine: ChunkingEngine):
        result = engine.chunk_all_documents(save=False)
        assert result.total_documents == 2
        assert result.total_chunks > 0
        # Check by_strategy counts
        assert result.by_strategy.get("api_reference", 0) == 1
        assert result.by_strategy.get("tutorial", 0) == 1

    def test_engine_saves_to_store(self, engine: ChunkingEngine):
        result = engine.chunk_all_documents(save=True)
        assert engine.store.count_chunks() == result.total_chunks
        
    def test_engine_with_parent_child(self, engine: ChunkingEngine, sample_document: NormalizedDocument):
        chunks = engine.chunk_document(sample_document, use_parent_child=True)
        # Should contain both parents and children
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if not c.metadata.get("is_parent")]
        
        assert len(parents) > 0
        assert len(children) > 0
        
        # Verify result format from chunk_all_documents
        result = engine.chunk_all_documents(use_parent_child=True, save=False)
        assert result.total_parents > 0
        assert result.total_children > 0

    def test_engine_audit_report(self, engine: ChunkingEngine, tmp_path):
        report_path = str(tmp_path / "audit.json")
        result = engine.chunk_and_save_report(report_path)
        
        # Report is valid JSON
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert data["total_chunks"] == result.audit.total_chunks
        assert "issues" in data

    def test_engine_handles_empty_document(self, engine: ChunkingEngine):
        doc = NormalizedDocument(doc_id="d1", url="http://test.com", title="Title", description="", content_blocks=[], metadata={}, template_id="", content_type="api_reference")
        chunks = engine.chunk_document(doc)
        assert len(chunks) == 0

    def test_engine_handles_document_with_only_code(self, engine: ChunkingEngine):
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="code", text="print('1')"),
                ContentBlock(block_type="code", text="print('2')"),
            ],
            metadata={}, template_id="", content_type="tutorial"
        )
        chunks = engine.chunk_document(doc)
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.content_type == "code"

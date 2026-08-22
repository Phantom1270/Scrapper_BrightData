"""
Tests for chunk_enricher.py
"""

from rag.chunking.chunk_enricher import ChunkEnricher
from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument, ContentBlock


class TestChunkEnricher:
    def test_enricher_adds_heading_text(self):
        enricher = ChunkEnricher({"max_tokens": 512})
        
        doc = NormalizedDocument(
            doc_id="doc1", url="url1", title="Title1", description="desc",
            content_blocks=[], metadata={}, template_id="tpl1", content_type="api_reference"
        )
        
        chunk = Chunk(
            chunk_id="c1", doc_id="doc1", url="url1", content="test",
            content_type="prose", heading_path=["API", "Method"], chunk_index=0, token_count=10
        )
        
        enriched = enricher.enrich([chunk], doc)[0]
        assert enriched.metadata["heading_text"] == "API > Method"

    def test_enricher_adds_word_count(self):
        enricher = ChunkEnricher({"max_tokens": 512})
        doc = NormalizedDocument(
            doc_id="doc1", url="url1", title="Title1", description="desc",
            content_blocks=[], metadata={}, template_id="tpl1", content_type="api_reference"
        )
        chunk = Chunk(
            chunk_id="c1", doc_id="doc1", url="url1", content="one two three",
            content_type="prose", heading_path=[], chunk_index=0, token_count=3
        )
        enriched = enricher.enrich([chunk], doc)[0]
        assert enriched.metadata["word_count"] == 3
        assert enriched.metadata["char_count"] == 13

    def test_enricher_adds_document_title(self):
        enricher = ChunkEnricher({"max_tokens": 512})
        doc = NormalizedDocument(
            doc_id="doc1", url="url1", title="Title1", description="desc",
            content_blocks=[], metadata={}, template_id="tpl1", content_type="api_reference"
        )
        chunk = Chunk(
            chunk_id="c1", doc_id="doc1", url="url1", content="test",
            content_type="prose", heading_path=[], chunk_index=0, token_count=1
        )
        enriched = enricher.enrich([chunk], doc)[0]
        assert enriched.metadata["document_title"] == "Title1"
        assert enriched.metadata["document_url"] == "url1"
        assert enriched.metadata["document_content_type"] == "api_reference"
        assert enriched.metadata["template_id"] == "tpl1"
        assert enriched.metadata["source_section"] == ""

    def test_enricher_adds_template_id(self):
        # Implicitly tested in test_enricher_adds_document_title
        pass

    def test_enricher_sets_is_oversized_for_large_chunks(self):
        enricher = ChunkEnricher({"max_tokens": 10})
        doc = NormalizedDocument(
            doc_id="doc1", url="url1", title="Title1", description="desc",
            content_blocks=[], metadata={}, template_id="tpl1", content_type="api_reference"
        )
        
        # token_count = 11 > max_tokens (10)
        chunk = Chunk(
            chunk_id="c1", doc_id="doc1", url="url1", content="test",
            content_type="prose", heading_path=["API"], chunk_index=0, token_count=11
        )
        
        enriched = enricher.enrich([chunk], doc)[0]
        assert enriched.metadata["is_oversized"] is True
        assert enriched.metadata["source_section"] == "API"
        
        # token_count = 5 < max_tokens (10)
        chunk2 = Chunk(
            chunk_id="c2", doc_id="doc1", url="url1", content="test",
            content_type="prose", heading_path=["API"], chunk_index=1, token_count=5
        )
        enriched2 = enricher.enrich([chunk2], doc)[0]
        assert enriched2.metadata["is_oversized"] is False

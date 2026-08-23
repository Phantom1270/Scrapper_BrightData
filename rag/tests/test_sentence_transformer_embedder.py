"""
Tests for sentence-transformers embedder.
"""

import math
import pytest
from unittest.mock import Mock

try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

if HAS_SENTENCE_TRANSFORMERS:
    from rag.search.embeddings.sentence_transformer import SentenceTransformerEmbedder


@pytest.fixture
def test_settings():
    """Mock settings to use a small fast model for testing."""
    settings = Mock()
    settings.embedding.model_name = "all-MiniLM-L6-v2"
    settings.embedding.device = "cpu"
    settings.embedding.batch_size = 4
    return settings


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestSentenceTransformerEmbedder:
    
    def test_embed_documents_returns_vectors(self, test_settings):
        embedder = SentenceTransformerEmbedder(settings=test_settings)
        texts = ["text one", "text two", "text three", "text four", "text five"]
        results = embedder.embed_documents(texts)
        assert len(results) == 5
        assert isinstance(results, list)
        assert isinstance(results[0], list)
        assert isinstance(results[0][0], float)

    def test_embed_query_returns_vector(self, test_settings):
        embedder = SentenceTransformerEmbedder(settings=test_settings)
        result = embedder.embed_query("search query")
        assert len(result) == embedder.get_dimension()
        assert isinstance(result, list)
        assert isinstance(result[0], float)

    def test_dimension_correct(self, test_settings):
        embedder = SentenceTransformerEmbedder(settings=test_settings)
        # all-MiniLM-L6-v2 dimension is 384
        assert embedder.get_dimension() == 384

    def test_embeddings_are_normalized(self, test_settings):
        embedder = SentenceTransformerEmbedder(settings=test_settings)
        result = embedder.embed_query("normalize me")
        # L2 norm should be ~1.0
        norm = math.sqrt(sum(x * x for x in result))
        assert math.isclose(norm, 1.0, abs_tol=1e-4)

    def test_embed_documents_and_query_produce_consistent_dimensions(self, test_settings):
        embedder = SentenceTransformerEmbedder(settings=test_settings)
        docs = embedder.embed_documents(["doc"])
        query = embedder.embed_query("query")
        assert len(docs[0]) == len(query)

    def test_query_prefix_applied_for_bge_models(self):
        # We just test the initialization logic, we don't need to load the model for this
        bge_settings = Mock()
        bge_settings.embedding.model_name = "BAAI/bge-large-en-v1.5"
        bge_settings.embedding.device = "cpu"
        bge_settings.embedding.batch_size = 4
        
        # We can just mock SentenceTransformer to avoid loading the heavy BAAI model
        import sys
        
        # We will test this by overriding __init__ partially or inspecting attributes after it loads
        # However, it will try to load the model. To prevent large download, we just mock the module.
        # It's easier to just mock the model creation but testing attributes is enough.
        # Actually, let's just patch SentenceTransformer init.
        
        from unittest.mock import patch
        
        with patch("rag.search.embeddings.sentence_transformer.SentenceTransformer") as mock_st:
            embedder = SentenceTransformerEmbedder(settings=bge_settings)
            assert embedder.query_prefix == "Represent this sentence for searching relevant passages: "
            assert embedder.doc_prefix == ""

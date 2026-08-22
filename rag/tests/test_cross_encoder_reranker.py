import pytest

try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from rag.models.retrieval import RetrievalResult
from rag.config.settings import Settings


@pytest.fixture
def mock_settings():
    settings = Settings()
    # Use a tiny, fast model for testing
    settings.reranker.model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    settings.embedding.device = "cpu"
    return settings


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestCrossEncoderReranker:
    def test_rerank_returns_results(self, mock_settings):
        from rag.retrieval.reranker.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker(mock_settings)
        candidates = [
            RetrievalResult("c1", "the quick brown fox", "", [], "", 0.1, "vector"),
            RetrievalResult("c2", "jumps over the lazy dog", "", [], "", 0.2, "bm25"),
            RetrievalResult("c3", "hello world", "", [], "", 0.3, "vector"),
            RetrievalResult("c4", "python programming", "", [], "", 0.4, "bm25"),
            RetrievalResult("c5", "data science", "", [], "", 0.5, "vector"),
        ]
        
        results = reranker.rerank("fox", candidates, top_k=5)
        assert len(results) == 5

    def test_rerank_reorders_by_relevance(self, mock_settings):
        from rag.retrieval.reranker.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker(mock_settings)
        candidates = [
            # Start with the best answer at the worst original score (last)
            RetrievalResult("c1", "Paris is the capital of France.", "", [], "", 0.1, "vector"),
            RetrievalResult("c2", "A completely unrelated document about dogs.", "", [], "", 0.9, "vector"),
            RetrievalResult("c3", "Apples are delicious fruits.", "", [], "", 0.8, "vector"),
        ]
        
        results = reranker.rerank("What is the capital of France?", candidates, top_k=3)
        
        # Verify it moved to the top
        assert results[0].chunk_id == "c1"

    def test_rerank_scores_are_floats(self, mock_settings):
        from rag.retrieval.reranker.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker(mock_settings)
        candidates = [
            RetrievalResult("c1", "test content", "", [], "", 0.1, "vector")
        ]
        results = reranker.rerank("test query", candidates)
        
        assert isinstance(results[0].score, float)

    def test_model_name_returns_string(self, mock_settings):
        from rag.retrieval.reranker.cross_encoder import CrossEncoderReranker
        
        reranker = CrossEncoderReranker(mock_settings)
        name = reranker.get_model_name()
        
        assert isinstance(name, str)
        assert len(name) > 0
        assert name == "cross-encoder/ms-marco-MiniLM-L-2-v2"

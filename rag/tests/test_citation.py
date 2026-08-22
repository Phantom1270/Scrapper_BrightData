import pytest

from rag.generation.citation import CitationExtractor
from rag.models.retrieval import RetrievalResult


@pytest.fixture
def sample_results():
    return [
        RetrievalResult("c1", "content 1", "https://example.com/api", ["API Reference", "config_context"], "api", 0.95, "vector"),
        RetrievalResult("c2", "content 2", "https://example.com/tutorial", ["Tutorials", "Getting Started"], "tut", 0.8, "vector"),
        RetrievalResult("c3", "content 3", "https://example.com/notes", [], "tut", 0.7, "vector"),
    ]


class TestCitationExtractor:
    def test_extract_citations_basic(self, sample_results):
        extractor = CitationExtractor()
        answer = "The function allows you to configure things [Source: API Reference > config_context]"
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 1
        assert citations[0]["reference_text"] == "API Reference > config_context"
        assert citations[0]["matched"] is True
        assert citations[0]["chunk_id"] == "c1"

    def test_extract_citations_multiple(self, sample_results):
        extractor = CitationExtractor()
        answer = "Configure it [Source: API Reference > config_context] or read more [Source: Tutorials > Getting Started]."
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 2
        assert citations[0]["chunk_id"] == "c1"
        assert citations[1]["chunk_id"] == "c2"

    def test_extract_citations_no_citations(self, sample_results):
        extractor = CitationExtractor()
        answer = "This is an answer without citations."
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert citations == []

    def test_extract_citations_unmatched(self, sample_results):
        extractor = CitationExtractor()
        answer = "Bad reference [Source: Nonexistent Section]"
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 1
        assert citations[0]["matched"] is False
        assert citations[0]["chunk_id"] is None
        assert citations[0]["reference_text"] == "Nonexistent Section"

    def test_extract_citations_partial_match(self, sample_results):
        extractor = CitationExtractor()
        answer = "Partial [Source: config_context]"
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 1
        assert citations[0]["matched"] is True
        assert citations[0]["chunk_id"] == "c1"

    def test_extract_citations_url_match(self, sample_results):
        extractor = CitationExtractor()
        answer = "URL match [Source: https://example.com/notes]"
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 1
        assert citations[0]["matched"] is True
        assert citations[0]["chunk_id"] == "c3"

    def test_extract_citations_fuzzy_match(self, sample_results):
        extractor = CitationExtractor()
        # "config context" instead of "config_context"
        answer = "Fuzzy [Source: config context API]"
        
        citations = extractor.extract_citations(answer, sample_results)
        
        assert len(citations) == 1
        assert citations[0]["matched"] is True
        assert citations[0]["chunk_id"] == "c1"

    def test_format_sources_footer(self, sample_results):
        extractor = CitationExtractor()
        footer = extractor.format_sources_footer(sample_results)
        
        assert "Sources:" in footer
        assert "API Reference > config_context (https://example.com/api)" in footer
        assert "Tutorials > Getting Started (https://example.com/tutorial)" in footer

    def test_format_sources_footer_deduplicates_urls(self, sample_results):
        extractor = CitationExtractor()
        # Add another result with same URL
        results = sample_results + [
            RetrievalResult("c4", "content 4", "https://example.com/api", ["API Reference", "other"], "api", 0.6, "vector")
        ]
        
        footer = extractor.format_sources_footer(results)
        
        # URL should only appear once
        assert footer.count("https://example.com/api") == 1

    def test_has_citations_true(self):
        extractor = CitationExtractor()
        assert extractor.has_citations("Here is info [Source: Docs]") is True

    def test_has_citations_false(self):
        extractor = CitationExtractor()
        assert extractor.has_citations("Here is info without citations") is False

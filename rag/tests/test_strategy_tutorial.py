"""
Tests for tutorial.py
"""

from rag.chunking.strategies.tutorial import TutorialChunkingStrategy
from rag.models.document import NormalizedDocument, ContentBlock


class TestTutorialChunkingStrategy:
    def setup_method(self):
        self.strategy = TutorialChunkingStrategy({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        self.doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Tutorial Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="Intro text.", heading=""),
                ContentBlock(block_type="prose", text="Section 1 text. " * 50, heading="Section 1"),
                ContentBlock(block_type="code", text="print('s1')", heading=""),
                ContentBlock(block_type="prose", text="Section 2 text. " * 50, heading="Section 2"),
                ContentBlock(block_type="code", text="print('s2')", heading="Code Title"),
                ContentBlock(block_type="note", text="A note.", heading="Note")
            ],
            metadata={}, template_id="", content_type="tutorial"
        )

    def test_sections_become_chunks(self):
        chunks = self.strategy.chunk(self.doc)
        prose_chunks = [c for c in chunks if c.content_type == "prose"]
        assert len(prose_chunks) >= 3  # Intro, Section 1, Section 2
        
        # We can identify them by heading
        s1_chunks = [c for c in prose_chunks if c.heading_path == ["Tutorial Title", "Section 1"]]
        assert len(s1_chunks) >= 1

    def test_code_blocks_are_atomic(self):
        # Even long code blocks are atomic
        doc_large_code = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="code", text="code_word " * 500, heading="Large Code"),
            ],
            metadata={}, template_id="", content_type="tutorial"
        )
        chunks = self.strategy.chunk(doc_large_code)
        assert len(chunks) == 1
        assert chunks[0].content_type == "code"
        assert chunks[0].metadata.get("is_oversized") is True

    def test_heading_paths_include_section_titles(self):
        chunks = self.strategy.chunk(self.doc)
        
        # Intro has no heading -> just title
        assert chunks[0].heading_path == ["Tutorial Title"]
        
        # Section 1
        assert chunks[1].heading_path == ["Tutorial Title", "Section 1"]
        
        # Code without heading -> Code Example
        assert chunks[2].heading_path == ["Tutorial Title", "Code Example"]
        
        # Code with heading -> Code Title
        code_w_heading = [c for c in chunks if c.content_type == "code" and c.content == "print('s2')"][0]
        assert code_w_heading.heading_path == ["Tutorial Title", "Code Title"]

    def test_long_section_is_split(self):
        strategy = TutorialChunkingStrategy({"max_tokens": 10, "min_tokens": 5, "overlap_tokens": 2, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="word " * 100, heading="Section 1"),
            ],
            metadata={}, template_id="", content_type="tutorial"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.heading_path == ["Title", "Section 1"]
            assert chunk.content_type == "prose"

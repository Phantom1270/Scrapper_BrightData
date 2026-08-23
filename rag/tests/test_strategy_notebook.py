"""
Tests for notebook.py
"""

from rag.chunking.strategies.notebook import NotebookChunkingStrategy
from rag.models.document import NormalizedDocument, ContentBlock


class TestNotebookChunkingStrategy:
    def setup_method(self):
        self.strategy = NotebookChunkingStrategy({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        
    def test_markdown_cells_become_prose_chunks(self):
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Notebook Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="Markdown 1. " * 50, heading=""),
                ContentBlock(block_type="prose", text="Markdown 2. " * 50, heading="Section 1"),
            ],
            metadata={}, template_id="", content_type="notebook"
        )
        chunks = self.strategy.chunk(doc)
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.content_type == "prose"

    def test_code_cells_become_code_chunks(self):
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Notebook Title", description="",
            content_blocks=[
                ContentBlock(block_type="code", text="print('1')", heading=""),
                ContentBlock(block_type="code", text="print('2')", heading=""),
            ],
            metadata={}, template_id="", content_type="notebook"
        )
        chunks = self.strategy.chunk(doc)
        assert len(chunks) == 2
        for chunk in chunks:
            assert chunk.content_type == "code"
            assert chunk.heading_path == ["Notebook Title", "Code"]

    def test_current_section_heading_is_tracked(self):
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Notebook Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="Training. " * 100, heading="Training"),
                ContentBlock(block_type="code", text="train()", heading=""),
            ],
            metadata={}, template_id="", content_type="notebook"
        )
        chunks = self.strategy.chunk(doc)
        
        # Prose chunk
        assert chunks[0].heading_path == ["Notebook Title", "Training"]
        
        # Code chunk
        assert chunks[1].heading_path == ["Notebook Title", "Training", "Code"]
        assert chunks[1].metadata.get("preceding_heading") == "Training"

    def test_short_prose_before_code_is_merged(self):
        # Short prose < min_tokens (100 tokens)
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Notebook Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="This is short.", heading="Training"),
                ContentBlock(block_type="code", text="train()", heading=""),
            ],
            metadata={}, template_id="", content_type="notebook"
        )
        chunks = self.strategy.chunk(doc)
        assert len(chunks) == 1
        
        chunk = chunks[0]
        assert chunk.content_type == "code"
        assert "This is short." in chunk.content
        assert "train()" in chunk.content
        assert chunk.heading_path == ["Notebook Title", "Training", "Code"]
        assert chunk.metadata.get("context_prefix") == "Training"

    def test_long_prose_before_code_is_not_merged(self):
        # Long prose >= min_tokens
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="Notebook Title", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="word " * 100, heading="Training"),
                ContentBlock(block_type="code", text="train()", heading=""),
            ],
            metadata={}, template_id="", content_type="notebook"
        )
        chunks = self.strategy.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].content_type == "prose"
        assert chunks[1].content_type == "code"

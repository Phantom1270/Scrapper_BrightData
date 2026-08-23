"""
Tests for api_reference.py
"""

from rag.chunking.strategies.api_reference import ApiReferenceChunkingStrategy
from rag.models.document import NormalizedDocument, ContentBlock


class TestApiReferenceChunkingStrategy:
    def setup_method(self):
        self.strategy = ApiReferenceChunkingStrategy({"max_tokens": 512, "min_tokens": 100, "overlap_tokens": 75, "encoding_name": "cl100k_base"})
        self.doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="DocTitle", description="",
            content_blocks=[
                ContentBlock(block_type="function_signature", text="def foo():", heading=""),
                ContentBlock(block_type="parameter_list", text="param1: int", heading="Param1"),
                ContentBlock(block_type="parameter_list", text="param2: str", heading="Param2"),
                ContentBlock(block_type="parameter_list", text="param3: bool", heading="Param3"),
                ContentBlock(block_type="prose", text="Description " * 100, heading="Desc"),
                ContentBlock(block_type="code", text="print(foo())", heading="Code"),
                ContentBlock(block_type="note", text="A note.", heading=""),
                ContentBlock(block_type="prose", text="See also bar().", heading="See Also")
            ],
            metadata={}, template_id="", content_type="api_reference"
        )

    def test_signature_is_atomic(self):
        chunks = self.strategy.chunk(self.doc)
        sig_chunks = [c for c in chunks if c.content_type == "function_signature"]
        assert len(sig_chunks) == 1
        assert sig_chunks[0].content == "def foo():"
        assert sig_chunks[0].heading_path == ["DocTitle", "Signature"]

    def test_parameters_are_individual_chunks(self):
        # We need to make the parameters large enough to avoid grouping
        doc_large_params = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="DocTitle", description="",
            content_blocks=[
                ContentBlock(block_type="parameter_list", text="param1: int " * 100, heading="Param1"),
                ContentBlock(block_type="parameter_list", text="param2: str " * 100, heading="Param2"),
            ],
            metadata={}, template_id="", content_type="api_reference"
        )
        chunks = self.strategy.chunk(doc_large_params)
        param_chunks = [c for c in chunks if c.content_type == "parameter_list"]
        assert len(param_chunks) == 2
        assert "param1" in param_chunks[0].content
        assert "param2" in param_chunks[1].content

    def test_code_block_is_never_split(self):
        strategy = ApiReferenceChunkingStrategy({"max_tokens": 10, "min_tokens": 5, "overlap_tokens": 0, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="DocTitle", description="",
            content_blocks=[
                ContentBlock(block_type="code", text="print('hello world this is a long code block that exceeds limit')", heading="Code"),
            ],
            metadata={}, template_id="", content_type="api_reference"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content_type == "code"
        assert chunks[0].metadata.get("is_oversized") is True

    def test_prose_is_split_when_long(self):
        strategy = ApiReferenceChunkingStrategy({"max_tokens": 50, "min_tokens": 5, "overlap_tokens": 10, "encoding_name": "cl100k_base"})
        doc = NormalizedDocument(
            doc_id="d1", url="http://test.com", title="DocTitle", description="",
            content_blocks=[
                ContentBlock(block_type="prose", text="word " * 100, heading="Desc"),
            ],
            metadata={}, template_id="", content_type="api_reference"
        )
        chunks = strategy.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.heading_path == ["DocTitle", "Desc"]

    def test_heading_paths_are_correct(self):
        chunks = self.strategy.chunk(self.doc)
        for c in chunks:
            if c.content_type == "function_signature":
                assert c.heading_path == ["DocTitle", "Signature"]
            elif "param" in c.content:
                # Based on parameter grouping, the heading should be ["DocTitle", "Parameters"]
                assert c.heading_path == ["DocTitle", "Parameters"]
            elif c.content_type == "code":
                assert c.heading_path == ["DocTitle", "Code"]

    def test_chunk_order_matches_block_order(self):
        chunks = self.strategy.chunk(self.doc)
        content_types = [c.content_type for c in chunks]
        
        # After param grouping, the param chunks will be merged into 1 chunk
        # Original block types: signature, param (x3), prose, code, note, prose
        # New chunk types: function_signature, parameter_list, prose, code, note, prose
        expected = ["function_signature", "parameter_list", "prose", "code", "note", "prose"]
        
        # Ensure we have the same order
        filtered_types = []
        last_type = None
        for ct in content_types:
            if ct != last_type:
                filtered_types.append(ct)
                last_type = ct
                
        assert filtered_types == expected

    def test_total_content_preserved(self):
        chunks = self.strategy.chunk(self.doc)
        combined = " ".join(c.content for c in chunks)
        assert "def foo():" in combined
        assert "param1: int" in combined
        assert "Description" in combined
        assert "print(foo())" in combined
        assert "A note." in combined

    def test_short_parameters_are_grouped(self):
        chunks = self.strategy.chunk(self.doc)
        param_chunks = [c for c in chunks if c.content_type == "parameter_list"]
        
        # The 3 parameters in self.doc are short, so they should be grouped into 1 chunk
        assert len(param_chunks) == 1
        assert "param1: int" in param_chunks[0].content
        assert "param3: bool" in param_chunks[0].content
        assert param_chunks[0].heading_path == ["DocTitle", "Parameters"]

"""
Tests for heading_builder.py
"""

from rag.chunking.heading_builder import HeadingBuilder


class TestHeadingBuilder:
    def test_build_path_simple(self):
        builder = HeadingBuilder()
        path = builder.build_path("config_context", "Parameters")
        assert path == ["config_context", "Parameters"]

    def test_build_path_empty_heading(self):
        builder = HeadingBuilder()
        path = builder.build_path("config_context")
        assert path == ["config_context"]

    def test_build_path_strips_trailing_hash(self):
        builder = HeadingBuilder()
        path = builder.build_path("config_context #", "Parameters #")
        assert path == ["config_context", "Parameters"]

    def test_build_path_deduplicates(self):
        builder = HeadingBuilder()
        path = builder.build_path("API", "API")
        assert path == ["API"]
        
        path2 = builder.build_path("API", "Params", ["API", "Params"])
        assert path2 == ["API", "Params"]

    def test_path_to_string(self):
        builder = HeadingBuilder()
        assert builder.path_to_string(["API", "Params", "x"]) == "API > Params > x"
        assert builder.path_to_string(["Single"]) == "Single"
        assert builder.path_to_string([]) == ""

    def test_path_to_prefixed_text(self):
        builder = HeadingBuilder()
        text = builder.path_to_prefixed_text(["API", "Params", "x"], "The actual content...")
        assert text == "## API > Params > x\n\nThe actual content..."
        
    def test_path_to_prefixed_text_empty_path(self):
        builder = HeadingBuilder()
        text = builder.path_to_prefixed_text([], "The actual content...")
        assert text == "The actual content..."

"""
Tests for ContentCleaner.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from rag.models.document import ContentBlock, NormalizedDocument
from rag.pipeline.cleaner import ContentCleaner
from rag.utils.ids import generate_doc_id


@pytest.fixture
def cleaner():
    return ContentCleaner()


def _make_doc(
    title: str = "Test",
    blocks: list = None,
    content_type: str = "api_reference",
) -> NormalizedDocument:
    return NormalizedDocument(
        doc_id=generate_doc_id("https://example.com/test"),
        url="https://example.com/test",
        title=title,
        description="A description.",
        content_blocks=blocks or [ContentBlock(block_type="prose", text="Hello world.")],
        metadata={},
        template_id="tpl_001",
        content_type=content_type,
    )


class TestCleanCodeBlock:
    def test_clean_code_removes_copy_to_clipboard(self, cleaner):
        block = ContentBlock(block_type="code", text="print('hi')\nCopy to clipboard\nx = 1")
        cleaned = cleaner.clean_block(block)
        assert "copy to clipboard" not in cleaned.text.lower()
        assert "print('hi')" in cleaned.text

    def test_clean_code_removes_progress_bars(self, cleaner):
        block = ContentBlock(block_type="code", text="Loading ━━━━━━━━━━━━ 100%\nprint('done')")
        cleaned = cleaner.clean_block(block)
        assert "━━━" not in cleaned.text
        assert "print('done')" in cleaned.text

    def test_clean_code_preserves_actual_code(self, cleaner):
        code = "import sklearn\nfrom sklearn.linear_model import LinearRegression\nx = 1 + 2"
        block = ContentBlock(block_type="code", text=code)
        cleaned = cleaner.clean_block(block)
        assert "import sklearn" in cleaned.text
        assert "LinearRegression" in cleaned.text
        # No normalization applied to code (no collapsing spaces etc.)
        assert "from sklearn" in cleaned.text

    def test_clean_code_preserves_indentation(self, cleaner):
        block = ContentBlock(block_type="code", text="def foo():\n    return 1")
        cleaned = cleaner.clean_block(block)
        assert "    return 1" in cleaned.text

    def test_clean_code_language_preserved(self, cleaner):
        block = ContentBlock(block_type="code", text="print(1)", language="python")
        cleaned = cleaner.clean_block(block)
        assert cleaned.language == "python"


class TestCleanSignatureBlock:
    def test_clean_signature_removes_source_marker(self, cleaner):
        block = ContentBlock(
            block_type="function_signature",
            text="sklearn.config_context(*) [source] #",
        )
        cleaned = cleaner.clean_block(block)
        assert "[source]" not in cleaned.text
        assert "config_context" in cleaned.text

    def test_clean_signature_truncates_long_dumps(self, cleaner):
        # Simulate a signature where param docs got appended inline
        long_sig = "func(a, b, c)" + " " * 10 + "Parameters : a : int Description of a. b : str..."
        long_sig = long_sig + "X" * 600  # ensure > 500 chars
        block = ContentBlock(block_type="function_signature", text=long_sig)
        cleaned = cleaner.clean_block(block)
        # Should truncate at "Parameters :"
        assert "func(a, b, c)" in cleaned.text

    def test_clean_signature_collapses_spaces(self, cleaner):
        block = ContentBlock(
            block_type="function_signature",
            text="sklearn.config_context(*,   assume_finite=False,   working_memory=1024)",
        )
        cleaned = cleaner.clean_block(block)
        assert "  " not in cleaned.text  # no double spaces


class TestCleanProseBlock:
    def test_clean_prose_applies_all_cleaning(self, cleaner):
        text = "hello   world\n\n\n\ntest &amp; more"
        block = ContentBlock(block_type="prose", text=text)
        cleaned = cleaner.clean_block(block)
        # No triple newlines
        assert "\n\n\n" not in cleaned.text
        # HTML entity fixed
        assert "&amp;" not in cleaned.text
        assert "&" in cleaned.text

    def test_clean_prose_removes_boilerplate(self, cleaner):
        block = ContentBlock(block_type="prose", text="Useful text.\nCopy to clipboard\nMore text.")
        cleaned = cleaner.clean_block(block)
        assert "useful text" in cleaned.text.lower()
        assert "copy to clipboard" not in cleaned.text.lower()

    def test_clean_note_block(self, cleaner):
        block = ContentBlock(block_type="note", text="  Note: be careful.  ")
        cleaned = cleaner.clean_block(block)
        assert "careful" in cleaned.text


class TestCleanTitle:
    def test_clean_title_strips_trailing_hash(self, cleaner):
        doc = _make_doc(title="config_context #")
        cleaned = cleaner.clean_document(doc)
        assert cleaned.title == "config_context"

    def test_clean_title_strips_multiple_hashes(self, cleaner):
        doc = _make_doc(title="My Function ##")
        cleaned = cleaner.clean_document(doc)
        assert "#" not in cleaned.title
        assert "My Function" in cleaned.title

    def test_clean_title_replaces_long_artifact(self, cleaner):
        """A title > 200 chars is a scraping artifact — should be replaced."""
        long_title = "A" * 300 + " more text"
        doc = _make_doc(
            title=long_title,
            blocks=[ContentBlock(block_type="prose", text="Content.", heading="Introduction")],
        )
        cleaned = cleaner.clean_document(doc)
        assert len(cleaned.title) <= 200

    def test_clean_title_replaces_html_artifact(self, cleaner):
        html_title = "<div>Page Title</div>"
        doc = _make_doc(
            title=html_title,
            blocks=[ContentBlock(block_type="prose", text="Content.", heading="Section")],
        )
        cleaned = cleaner.clean_document(doc)
        # HTML tag artifact replaced — should not contain raw HTML
        assert "<div>" not in cleaned.title


class TestCleanDocument:
    def test_clean_document_removes_empty_blocks(self, cleaner):
        blocks = [
            ContentBlock(block_type="prose", text="Real content."),
            ContentBlock(block_type="prose", text="   "),  # becomes empty after cleaning
            ContentBlock(block_type="code", text="print(1)"),
        ]
        doc = _make_doc(blocks=blocks)
        cleaned = cleaner.clean_document(doc)
        # The whitespace-only block should be removed
        for b in cleaned.content_blocks:
            assert b.text.strip() != ""

    def test_clean_document_does_not_mutate_original(self, cleaner):
        original_title = "Original Title #"
        blocks = [ContentBlock(block_type="prose", text="Hello &amp; world")]
        doc = _make_doc(title=original_title, blocks=blocks)

        # Deep copy check: store original values
        orig_block_text = doc.content_blocks[0].text

        cleaner.clean_document(doc)

        # Original should be unchanged
        assert doc.title == original_title
        assert doc.content_blocks[0].text == orig_block_text

    def test_clean_document_cleans_description(self, cleaner):
        doc = NormalizedDocument(
            doc_id=generate_doc_id("https://example.com"),
            url="https://example.com",
            title="Test",
            description="Hello &lt;world&gt;\n\n\n\nExtra",
            content_blocks=[ContentBlock(block_type="prose", text="Body.")],
            metadata={},
            template_id="tpl_001",
            content_type="api_reference",
        )
        cleaned = cleaner.clean_document(doc)
        assert "&lt;" not in cleaned.description
        assert "\n\n\n" not in cleaned.description

    def test_clean_document_returns_normalized_document(self, cleaner):
        doc = _make_doc()
        cleaned = cleaner.clean_document(doc)
        assert isinstance(cleaned, NormalizedDocument)

    def test_clean_document_preserves_url_and_template(self, cleaner):
        doc = _make_doc()
        cleaned = cleaner.clean_document(doc)
        assert cleaned.url == doc.url
        assert cleaned.template_id == doc.template_id
        assert cleaned.doc_id == doc.doc_id

    def test_clean_block_heading_stripped(self, cleaner):
        block = ContentBlock(block_type="prose", text="Text.", heading="  Introduction  ")
        cleaned = cleaner.clean_block(block)
        assert cleaned.heading == "Introduction"

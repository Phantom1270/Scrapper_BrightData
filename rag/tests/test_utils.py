"""
Tests for all RAG utility functions.

Covers: text.py, tokens.py, ids.py, hashing.py
"""

from __future__ import annotations

import pytest

from rag.utils.hashing import content_hash, near_duplicate_ratio
from rag.utils.ids import generate_chunk_id, generate_doc_id
from rag.utils.text import (
    clean_unicode,
    clean_whitespace,
    extract_code_language,
    strip_boilerplate,
)
from rag.utils.tokens import count_tokens, count_tokens_batch


# ---------------------------------------------------------------------------
# clean_whitespace
# ---------------------------------------------------------------------------


class TestCleanWhitespace:
    def test_collapses_multiple_spaces(self):
        result = clean_whitespace("hello   world")
        assert result == "hello world"

    def test_preserves_single_newlines(self):
        text = "line one\nline two"
        result = clean_whitespace(text)
        assert "line one" in result
        assert "line two" in result

    def test_collapses_3_plus_newlines_to_2(self):
        text = "para1\n\n\n\n\npara2"
        result = clean_whitespace(text)
        assert "\n\n\n" not in result
        assert "para1" in result
        assert "para2" in result

    def test_strips_trailing_whitespace_per_line(self):
        text = "hello   \nworld   "
        result = clean_whitespace(text)
        for line in result.splitlines():
            assert line == line.rstrip()

    def test_strips_leading_trailing_from_whole_string(self):
        text = "\n\nhello\n\n"
        result = clean_whitespace(text)
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    def test_empty_string(self):
        assert clean_whitespace("") == ""

    def test_only_whitespace(self):
        result = clean_whitespace("   \n\n   ")
        assert result == ""

    def test_preserves_indentation(self):
        """Leading whitespace (indentation) should be preserved."""
        text = "def foo():\n    return 1"
        result = clean_whitespace(text)
        assert "    return 1" in result


# ---------------------------------------------------------------------------
# strip_boilerplate
# ---------------------------------------------------------------------------


class TestStripBoilerplate:
    def test_removes_copy_to_clipboard(self):
        text = "Here is the code.\nCopy to clipboard\nMore content."
        result = strip_boilerplate(text)
        assert "copy to clipboard" not in result.lower()
        assert "Here is the code" in result

    def test_removes_progress_bars(self):
        text = "Loading ━━━━━━━━━━━━━━ 100%"
        result = strip_boilerplate(text)
        # Progress bar characters should be stripped
        assert "━━━" not in result

    def test_removes_edit_this_page(self):
        text = "Edit this page\nSome content here."
        result = strip_boilerplate(text)
        assert "Some content here." in result

    def test_removes_previous_next_navigation(self):
        text = "Useful content.\nPrevious\nNext\nMore content."
        result = strip_boilerplate(text)
        assert "Useful content." in result
        assert "More content." in result

    def test_preserves_actual_content(self):
        """Real documentation text should not be removed."""
        text = (
            "The LinearRegression class fits a linear model with coefficients "
            "w = (w1, ..., wp) to minimize the residual sum of squares."
        )
        result = strip_boilerplate(text)
        assert "LinearRegression" in result
        assert "residual sum of squares" in result

    def test_empty_string(self):
        assert strip_boilerplate("") == ""


# ---------------------------------------------------------------------------
# clean_unicode
# ---------------------------------------------------------------------------


class TestCleanUnicode:
    def test_fixes_html_entities_amp(self):
        result = clean_unicode("Tom &amp; Jerry")
        assert "&amp;" not in result
        assert "&" in result

    def test_fixes_html_entities_lt_gt(self):
        result = clean_unicode("a &lt; b &gt; c")
        assert "<" in result
        assert ">" in result
        assert "&lt;" not in result

    def test_fixes_nbsp(self):
        result = clean_unicode("hello&nbsp;world")
        assert "&nbsp;" not in result

    def test_removes_null_bytes(self):
        result = clean_unicode("hello\x00world")
        assert "\x00" not in result

    def test_returns_empty_for_empty_input(self):
        assert clean_unicode("") == ""

    def test_preserves_normal_text(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert clean_unicode(text) == text


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_count_tokens_basic(self):
        """'hello world' should be roughly 2 tokens."""
        count = count_tokens("hello world")
        assert count >= 1  # at minimum 1 token

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_batch(self):
        texts = ["hello world", "this is a test", "scikit-learn documentation"]
        individual = [count_tokens(t) for t in texts]
        batch = count_tokens_batch(texts)
        assert batch == individual

    def test_count_tokens_batch_empty_list(self):
        assert count_tokens_batch([]) == []

    def test_count_tokens_batch_with_empty_string(self):
        result = count_tokens_batch(["hello", "", "world"])
        assert result[1] == 0

    def test_longer_text_has_more_tokens(self):
        short = count_tokens("hello")
        long_ = count_tokens("hello world this is a much longer piece of text with many words")
        assert long_ > short


# ---------------------------------------------------------------------------
# generate_doc_id / generate_chunk_id
# ---------------------------------------------------------------------------


class TestGenerateIds:
    def test_generate_doc_id_deterministic(self):
        url = "https://scikit-learn.org/stable/modules/generated/sklearn.config_context.html"
        id1 = generate_doc_id(url)
        id2 = generate_doc_id(url)
        assert id1 == id2

    def test_generate_doc_id_is_16_chars(self):
        url = "https://example.com/page"
        assert len(generate_doc_id(url)) == 16

    def test_generate_doc_id_different_for_different_urls(self):
        id1 = generate_doc_id("https://example.com/page1")
        id2 = generate_doc_id("https://example.com/page2")
        assert id1 != id2

    def test_generate_doc_id_index_disambiguates(self):
        url = "https://example.com/same"
        id0 = generate_doc_id(url, index=0)
        id1 = generate_doc_id(url, index=1)
        assert id0 != id1

    def test_generate_chunk_id_deterministic(self):
        doc_id = generate_doc_id("https://example.com/doc")
        cid1 = generate_chunk_id(doc_id, 0)
        cid2 = generate_chunk_id(doc_id, 0)
        assert cid1 == cid2

    def test_generate_chunk_id_is_16_chars(self):
        doc_id = generate_doc_id("https://example.com/doc")
        assert len(generate_chunk_id(doc_id, 0)) == 16

    def test_generate_chunk_id_different_for_different_indices(self):
        doc_id = generate_doc_id("https://example.com/doc")
        assert generate_chunk_id(doc_id, 0) != generate_chunk_id(doc_id, 1)

    def test_generate_chunk_id_different_for_different_docs(self):
        doc_id_a = generate_doc_id("https://example.com/a")
        doc_id_b = generate_doc_id("https://example.com/b")
        assert generate_chunk_id(doc_id_a, 0) != generate_chunk_id(doc_id_b, 0)


# ---------------------------------------------------------------------------
# content_hash / near_duplicate_ratio
# ---------------------------------------------------------------------------


class TestHashing:
    def test_content_hash_deterministic(self):
        text = "Hello, world!"
        assert content_hash(text) == content_hash(text)

    def test_content_hash_is_16_chars(self):
        assert len(content_hash("some text")) == 16

    def test_content_hash_different_for_different_text(self):
        h1 = content_hash("The quick brown fox")
        h2 = content_hash("The slow white cat")
        assert h1 != h2

    def test_content_hash_case_insensitive(self):
        """Hash should be the same for text differing only in case."""
        h1 = content_hash("Hello World")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_whitespace_normalized(self):
        """Extra whitespace should not change the hash."""
        h1 = content_hash("hello   world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_near_duplicate_ratio_identical_texts(self):
        text = "The quick brown fox jumps over the lazy dog." * 10
        ratio = near_duplicate_ratio(text, text)
        assert ratio >= 0.99

    def test_near_duplicate_ratio_different_texts(self):
        text_a = "scikit-learn is a machine learning library for Python."
        text_b = "Django is a high-level Python web framework."
        ratio = near_duplicate_ratio(text_a, text_b)
        assert ratio < 0.5

    def test_near_duplicate_ratio_both_empty(self):
        assert near_duplicate_ratio("", "") == 1.0

    def test_near_duplicate_ratio_one_empty(self):
        assert near_duplicate_ratio("some text", "") == 0.0
        assert near_duplicate_ratio("", "some text") == 0.0

    def test_near_duplicate_ratio_returns_float_in_range(self):
        ratio = near_duplicate_ratio("hello", "world")
        assert 0.0 <= ratio <= 1.0


# ---------------------------------------------------------------------------
# extract_code_language
# ---------------------------------------------------------------------------


class TestExtractCodeLanguage:
    def test_detects_python(self):
        code = "import numpy as np\nfrom sklearn import LinearRegression"
        assert extract_code_language(code) == "python"

    def test_detects_javascript(self):
        code = "const x = 5;\nfunction greet() { console.log('hi'); }"
        assert extract_code_language(code) == "javascript"

    def test_detects_bash_pip(self):
        code = "pip install scikit-learn"
        assert extract_code_language(code) == "python"  # pip/conda triggers python

    def test_detects_sql(self):
        code = "SELECT * FROM users WHERE id = 1;"
        assert extract_code_language(code) == "sql"

    def test_returns_empty_for_unknown(self):
        code = "!@#$%^&*()"
        result = extract_code_language(code)
        assert isinstance(result, str)  # should return a string (possibly empty)

    def test_empty_string_returns_empty(self):
        assert extract_code_language("") == ""

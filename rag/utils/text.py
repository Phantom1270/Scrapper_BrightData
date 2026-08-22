"""
Text cleaning and normalization utilities.

Pure functions. No I/O. No external state.
"""

from __future__ import annotations

import html
import re


# ---------------------------------------------------------------------------
# Boilerplate patterns (compiled once at module load)
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    # "Edit this page" or "Edit on GitHub"
    re.compile(r"\bedit\s+(this\s+page|on\s+github)\b", re.IGNORECASE),
    # "Was this page helpful?" variants
    re.compile(r"\bwas\s+this\s+(page|article)\s+helpful\b.*", re.IGNORECASE),
    # "Copy to clipboard"
    re.compile(r"\bcopy\s+to\s+clipboard\b", re.IGNORECASE),
    # "Previous" / "Next" standalone navigation
    re.compile(r"^\s*(Previous|Next)\s*$", re.MULTILINE),
    # Progress bar characters (common in terminal output pasted into docs)
    re.compile(r"[━╸█▉▊▋▌▍▎▏▐░▒▓]{3,}"),
    # Breadcrumb separators (lone ">" on a line or surrounded by spaces)
    re.compile(r"(?<!\S)>\s*(?!\S)"),
    # "Table of contents" header
    re.compile(r"^#+\s*Table\s+of\s+(Contents|Content)\s*$", re.MULTILINE | re.IGNORECASE),
    # Repeated "note" / "warning" prefixes stripped of context
    re.compile(r"^\s*(Note|Warning|Tip|Important)\s*:\s*$", re.MULTILINE | re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# HTML entity mapping (supplement html.unescape with common extras)
# ---------------------------------------------------------------------------

_EXTRA_ENTITIES: dict[str, str] = {
    "&nbsp;": " ",
    "&ndash;": "–",
    "&mdash;": "—",
    "&laquo;": "«",
    "&raquo;": "»",
    "&hellip;": "…",
    "&times;": "×",
    "&divide;": "÷",
    "&copy;": "©",
    "&reg;": "®",
    "&trade;": "™",
    "&#x27;": "'",
    "&#39;": "'",
    "&#34;": '"',
}

_ENTITY_RE = re.compile("|".join(re.escape(k) for k in _EXTRA_ENTITIES))

# ---------------------------------------------------------------------------
# Language detection patterns
# ---------------------------------------------------------------------------

_LANG_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("python", re.compile(r"\b(import|from|def|class|elif|lambda|print\s*\(|pip\s+install|conda\s+install)\b")),
    ("javascript", re.compile(r"\b(function|const\s+\w|let\s+\w|var\s+\w|=>\s*\{|require\(|module\.exports|console\.log)\b")),
    ("bash", re.compile(r"(^\$\s+|pip\s+install|conda\s+install|apt-get|brew\s+install|echo\s+|\.\/\w)", re.MULTILINE)),
    ("sql", re.compile(r"\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN|CREATE\s+TABLE)\b", re.IGNORECASE)),
    ("html", re.compile(r"<(html|head|body|div|span|a\s|p>|ul>|li>)", re.IGNORECASE)),
    ("css", re.compile(r"\{[^}]*:\s*[^}]+\}", re.DOTALL)),
    ("json", re.compile(r'^\s*\{[\s\S]*"[\w]+"\s*:')),
    ("yaml", re.compile(r"^\s*[\w_]+\s*:\s*\S", re.MULTILINE)),
]


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace within text.

    - Collapse multiple spaces/tabs on a line into a single space.
    - Strip trailing whitespace from each line.
    - Collapse 3+ consecutive blank lines into 2 blank lines.
    - Strip leading/trailing whitespace from the full string.
    """
    if not text:
        return text

    # Normalize tabs to spaces
    text = text.replace("\t", "    ")

    # Per-line: strip trailing whitespace and collapse interior multiple spaces
    lines = []
    for line in text.splitlines():
        # Preserve leading whitespace (indentation) but collapse interior runs
        stripped = line.rstrip()
        # Collapse multiple spaces (but not leading spaces for indentation)
        leading = len(stripped) - len(stripped.lstrip())
        indent = stripped[:leading]
        body = re.sub(r" {2,}", " ", stripped[leading:])
        lines.append(indent + body)

    # Collapse 3+ consecutive blank lines to exactly 2
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def strip_boilerplate(text: str) -> str:
    """
    Remove common scraping artifacts from text.

    Conservative — only removes clearly non-content strings.
    """
    if not text:
        return text

    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)

    # Collapse any blank lines introduced by removal
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_unicode(text: str) -> str:
    """
    Fix common encoding issues.

    - Unescape HTML entities (both standard and extras).
    - Remove null bytes.
    - Normalize unicode to NFC form.
    """
    if not text:
        return text

    import unicodedata

    # Replace extra HTML entities first
    text = _ENTITY_RE.sub(lambda m: _EXTRA_ENTITIES[m.group(0)], text)

    # Standard HTML entity unescaping (&amp; &lt; etc.)
    text = html.unescape(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # NFC normalization (canonical decomposition then composition)
    text = unicodedata.normalize("NFC", text)

    return text


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> str:
    """
    Truncate text to fit within max_tokens.

    Tries to break at a sentence boundary (". ", "! ", "? ").
    Falls back to hard truncation if no suitable boundary is found.
    """
    from rag.utils.tokens import count_tokens

    if count_tokens(text, encoding_name) <= max_tokens:
        return text

    # Binary-search approach: shrink by sentence chunks
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result_parts: list[str] = []
    running = 0

    for sentence in sentences:
        tok = count_tokens(sentence, encoding_name)
        if running + tok > max_tokens:
            break
        result_parts.append(sentence)
        running += tok

    if result_parts:
        return " ".join(result_parts)

    # Hard truncation as last resort — encode, slice, decode
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        token_ids = enc.encode(text)[:max_tokens]
        return enc.decode(token_ids)
    except Exception:
        # Rough character approximation (4 chars ≈ 1 token)
        return text[: max_tokens * 4]


def extract_code_language(code_text: str) -> str:
    """
    Detect the programming language of a code snippet.

    Returns a language string (e.g. "python", "javascript", "bash")
    or an empty string if the language cannot be determined.
    """
    if not code_text or not code_text.strip():
        return ""

    for lang, pattern in _LANG_PATTERNS:
        if pattern.search(code_text):
            return lang

    return ""

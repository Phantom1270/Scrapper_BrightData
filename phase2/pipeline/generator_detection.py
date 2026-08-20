"""
Stage 2.4 — Generator Detection
Try to identify which documentation generator produced the site.
This sets prior expectations for URL patterns.
"""

from models import ParsedURL, Phase1Output
from config import GENERATOR_SIGNALS
from utils.url_parser import split_path_segments


def detect_generator(
    phase1: Phase1Output,
    parsed_urls: list[ParsedURL],
) -> tuple[str | None, float]:
    """
    Analyze available signals to guess the documentation generator.

    Returns: (generator_name, confidence)

    Signals to check (in order of reliability):
    1. Phase 1 signals dict (if Phase 1 checked for objects.inv, etc.)
    2. URL path patterns
    3. File extensions in URLs
    4. Fallback: (None, 0.0)
    """
    scores: dict[str, float] = {gen: 0.0 for gen in GENERATOR_SIGNALS}

    # ── Signal 1: Phase 1 explicit signals ──
    signals = phase1.signals or {}

    if signals.get("has_objects_inv") is True:
        scores["sphinx"] = max(scores["sphinx"], 0.95)

    if signals.get("static_dir") == "_static":
        scores["sphinx"] = max(scores["sphinx"], 0.7)

    # ── Signal 2: URL path patterns ──
    all_paths = [u.canonical_url for u in parsed_urls]

    sphinx_indicators = ["/_static/", "/_sources/", "/objects.inv"]
    mkdocs_indicators = ["/search/search_index.json", "mkdocs"]
    docusaurus_indicators = ["/assets/js/docusaurus", "docusaurus"]

    for url_str in all_paths:
        for ind in sphinx_indicators:
            if ind in url_str:
                scores["sphinx"] = max(scores["sphinx"], 0.8)
                break
        for ind in mkdocs_indicators:
            if ind in url_str:
                scores["mkdocs"] = max(scores["mkdocs"], 0.75)
                break
        for ind in docusaurus_indicators:
            if ind in url_str:
                scores["docusaurus"] = max(scores["docusaurus"], 0.75)
                break

    # ── Signal 3: Path structure heuristics ──
    # Sphinx-style: paths with /modules/generated/
    sphinx_path_hits = sum(
        1 for u in parsed_urls
        if "/modules/generated/" in u.path or "/_sources" in u.path
    )
    if sphinx_path_hits > 0:
        scores["sphinx"] = max(scores["sphinx"], 0.6)

    # MkDocs: flat .html files with simple names
    flat_html_count = sum(
        1 for u in parsed_urls
        if u.path.count('/') <= 2 and u.path.endswith('.html')
    )
    total = len(parsed_urls) or 1
    if flat_html_count / total > 0.5 and scores["sphinx"] < 0.5:
        scores["mkdocs"] = max(scores["mkdocs"], 0.4)

    # ── Select winner ──
    best_gen = max(scores, key=lambda g: scores[g])
    best_score = scores[best_gen]

    if best_score < 0.3:
        return None, 0.0

    return best_gen, round(best_score, 4)

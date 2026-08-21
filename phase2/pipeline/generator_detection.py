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
    """
    
    # ── Priority 1: Phase 1 signals (highest reliability) ──────
    signals = phase1.signals or {}
    
    if signals.get("has_objects_inv") is True:
        return ("sphinx", 0.95)  # MUST be 0.95
        
    if signals.get("search_index_json") is True:
        return ("mkdocs", 0.95)
        
    if signals.get("detected_tool"):
        tool = signals["detected_tool"]
        if tool in ("sphinx", "mkdocs", "docusaurus", "vuepress", "hugo"):
            return (tool, 0.90)
    
    # ── Priority 2: URL path pattern analysis ──────────────────
    # Collect all paths for analysis
    paths = []
    for url in parsed_urls:
        path = url.version_free_path or url.path
        paths.append(path.lower())
    
    path_blob = " ".join(paths)  # One big string for substring checks
    
    scores: dict[str, float] = {}
    
    # Sphinx signals
    sphinx_score = 0.0
    if "/_static/searchtools.js" in path_blob:
        sphinx_score = max(sphinx_score, 0.90)
    if "/_static/" in path_blob:
        sphinx_score = max(sphinx_score, 0.70)
    if "/_sources/" in path_blob:
        sphinx_score = max(sphinx_score, 0.75)
    if "/objects.inv" in path_blob:
        sphinx_score = max(sphinx_score, 0.95)
    # Sphinx typically has /modules/generated/ pattern
    modules_generated_count = sum(1 for p in paths if "/modules/generated/" in p)
    if modules_generated_count > 10:
        sphinx_score = max(sphinx_score, 0.60)
    
    if sphinx_score > 0:
        scores["sphinx"] = sphinx_score
    
    # MkDocs signals
    mkdocs_score = 0.0
    if "/search/search_index.json" in path_blob:
        mkdocs_score = max(mkdocs_score, 0.90)
    if "mkdocs" in path_blob:
        mkdocs_score = max(mkdocs_score, 0.60)
    
    if mkdocs_score > 0:
        scores["mkdocs"] = mkdocs_score
    
    # Docusaurus signals
    docusaurus_score = 0.0
    if "/assets/js/docusaurus" in path_blob:
        docusaurus_score = max(docusaurus_score, 0.85)
    if "docusaurus" in path_blob:
        docusaurus_score = max(docusaurus_score, 0.50)
    
    if docusaurus_score > 0:
        scores["docusaurus"] = docusaurus_score
    
    # VuePress signals
    if "/.vuepress/" in path_blob:
        scores["vuepress"] = 0.80
    
    # Hugo signals
    if "/index.json" in path_blob:
        scores["hugo"] = max(scores.get("hugo", 0), 0.50)
    
    # ── Priority 3: Return best match ──────────────────────────
    if not scores:
        return (None, 0.0)
    
    best_generator = max(scores, key=scores.get)
    best_confidence = scores[best_generator]
    
    if best_confidence < 0.30:
        return (None, 0.0)
    
    return (best_generator, best_confidence)

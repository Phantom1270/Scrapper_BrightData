"""
Signal detection.
Probes the target site to identify the documentation generator.
Runs once at the start of the crawl.
"""

from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from crawler.fetcher import Fetcher
from config import SIGNAL_CHECK_URLS


def detect_signals(
    root_url: str,
    fetcher: Fetcher,
    homepage_html: str = None,
) -> dict:
    """
    Detect documentation generator signals from a website.

    Called ONCE at the start of the crawl, after fetching the seed URL.
    Uses the seed page's HTML + proactive URL checks.

    Returns: dict of signals
    """
    signals: dict = {}

    # Parse root_url to get base (scheme + host)
    parsed_root = urlparse(root_url)
    base = f"{parsed_root.scheme}://{parsed_root.netloc}"

    # ── 1. CHECK ROBOTS.TXT ──
    robots_url = urljoin(base + "/", "robots.txt")
    exists, status = fetcher.check_url_exists(robots_url)
    signals["has_robots_txt"] = exists

    # ── 2. CHECK OBJECTS.INV (Sphinx indicator) ──
    objects_inv_url = urljoin(base + "/", "objects.inv")
    exists, status = fetcher.check_url_exists(objects_inv_url)
    signals["has_objects_inv"] = exists

    # ── 3. CHECK SITEMAP ──
    sitemap_url = urljoin(base + "/", "sitemap.xml")
    exists, status = fetcher.check_url_exists(sitemap_url)
    signals["has_sitemap"] = exists

    # ── 4. CHECK SEARCH INDEX (MkDocs indicator) ──
    search_index_url = urljoin(root_url.rstrip("/") + "/", "search/search_index.json")
    exists, status = fetcher.check_url_exists(search_index_url)
    signals["has_search_index"] = exists

    # ── 5. PARSE HOMEPAGE HTML ──
    if homepage_html:
        try:
            soup = BeautifulSoup(homepage_html, "lxml")

            # a. Meta generator
            gen_tag = soup.find("meta", attrs={"name": lambda v: v and v.lower() == "generator"})
            if gen_tag and gen_tag.get("content"):
                signals["meta_generator"] = gen_tag["content"]

            # b. <link> tags with known static dirs
            for link_tag in soup.find_all("link", href=True):
                href = link_tag.get("href", "")
                if "/_static/" in href or href.startswith("_static/"):
                    signals["static_dir"] = "_static"
                if "/assets/" in href or href.startswith("assets/"):
                    signals.setdefault("assets_dir", "assets")
                if "/.vuepress/" in href or ".vuepress" in href:
                    signals["vuepress_dir"] = ".vuepress"

            # c. <script> tags with known tools
            for script_tag in soup.find_all("script", src=True):
                src = script_tag.get("src", "")
                if "docusaurus" in src.lower():
                    signals["docusaurus_detected"] = True
                if "mkdocs" in src.lower():
                    signals["mkdocs_detected"] = True
                if "searchtools.js" in src.lower():
                    signals["sphinx_searchtools"] = True

            # d. Canonical link
            canonical = soup.find("link", attrs={"rel": lambda v: v and "canonical" in (v if isinstance(v, list) else [v])})
            if canonical:
                signals["has_canonical"] = True

            # e. Body class names for known themes
            body = soup.find("body")
            all_text = str(soup)
            if "wy-grid-for-nav" in all_text:
                signals["rtd_theme"] = True
            if "md-container" in all_text:
                signals["mkdocs_material"] = True
            if "docusaurus" in all_text.lower():
                signals["docusaurus_in_html"] = True

        except Exception:
            pass  # Don't crash on signal detection failures

    # ── 7. DETERMINE BEST GUESS ──
    detected_tool = None
    detected_confidence = 0.0

    meta_gen = signals.get("meta_generator", "").lower()

    if meta_gen:
        if "sphinx" in meta_gen:
            detected_tool = "sphinx"
            detected_confidence = 0.95
        elif "mkdocs" in meta_gen:
            detected_tool = "mkdocs"
            detected_confidence = 0.95
        elif "docusaurus" in meta_gen:
            detected_tool = "docusaurus"
            detected_confidence = 0.95

    if not detected_tool:
        if signals.get("has_objects_inv"):
            detected_tool = "sphinx"
            detected_confidence = 0.95
        elif signals.get("has_search_index"):
            detected_tool = "mkdocs"
            detected_confidence = 0.90
        elif signals.get("static_dir") == "_static" or signals.get("sphinx_searchtools"):
            detected_tool = "sphinx"
            detected_confidence = 0.70
        elif signals.get("docusaurus_detected") or signals.get("docusaurus_in_html"):
            detected_tool = "docusaurus"
            detected_confidence = 0.85
        elif signals.get("rtd_theme"):
            detected_tool = "sphinx_rtd_theme"
            detected_confidence = 0.80
        elif signals.get("mkdocs_material") or signals.get("mkdocs_detected"):
            detected_tool = "mkdocs_material"
            detected_confidence = 0.80

    signals["detected_tool"] = detected_tool
    signals["detected_confidence"] = detected_confidence

    return signals

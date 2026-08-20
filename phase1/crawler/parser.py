"""
HTML parser.
Takes raw HTML string. Extracts all links and metadata.
No HTTP. No crawling logic. Just parsing.
"""

import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Optional
from config import SKIP_EXTENSIONS
from models import URLClassification


# Regex that matches any path segment repeated back-to-back: /foo/foo/ → /foo/
_DOUBLED_SEGMENT_RE = re.compile(r'(/[^/]+)\1(?=/|$)')


def resolve_url(base_url: str, href: str) -> str:
    """
    Resolve a (possibly relative) href against base_url.

    Steps:
    1. urljoin(base_url, href)  — handles relative paths, absolute URLs, etc.
    2. Collapse duplicate consecutive path segments produced by joining a
       page URL that already contains a segment with a relative href that
       repeats that segment.
       e.g. base=/stable/modules/generated/ + href=generated/sklearn.Foo.html
            → /stable/modules/generated/generated/sklearn.Foo.html  (wrong)
            → /stable/modules/generated/sklearn.Foo.html            (fixed)
    3. Return the cleaned absolute URL.
    """
    joined = urljoin(base_url, href)
    parsed = urlparse(joined)
    path = parsed.path
    # Iteratively collapse doubled consecutive segments until stable
    prev = None
    while prev != path:
        prev = path
        path = _DOUBLED_SEGMENT_RE.sub(r'\1', path)
    if path != parsed.path:
        joined = urlunparse(parsed._replace(path=path))
    return joined


class LinkInfo:
    """A single extracted link."""
    def __init__(self, href: str, text: str, rel: str, tag_name: str):
        self.href = href            # Raw href attribute value (resolved to absolute)
        self.text = text            # Link text (stripped whitespace)
        self.rel = rel              # rel attribute (nofollow, etc.)
        self.tag_name = tag_name    # "a", "link", "area", etc.


class PageMetadata:
    """Metadata extracted from an HTML page."""
    def __init__(self):
        self.title: str = ""
        self.meta_generator: str = ""        # <meta name="generator" content="...">
        self.meta_description: str = ""
        self.canonical_url: str = ""         # <link rel="canonical" href="...">
        self.charset: str = "utf-8"
        self.link_tags: list[LinkInfo] = []  # <link> tags (stylesheets, etc.)
        self.anchor_tags: list[LinkInfo] = [] # <a> tags


# Prefixes to skip during link extraction
_SKIP_PREFIXES = ("#", "mailto:", "tel:", "javascript:", "data:")


def extract_links(html: str, base_url: str) -> PageMetadata:
    """
    Parse HTML and extract all links and metadata.

    Parameters:
    - html: raw HTML string
    - base_url: the URL of the page (used to resolve relative links)

    Returns: PageMetadata object with all extracted data.
    """
    metadata = PageMetadata()
    soup = BeautifulSoup(html, "lxml")

    # ── 2. Extract metadata ──

    # a. Title
    title_tag = soup.find("title")
    if title_tag:
        metadata.title = title_tag.get_text(strip=True)

    # b. Meta generator
    gen_tag = soup.find("meta", attrs={"name": lambda v: v and v.lower() == "generator"})
    if gen_tag and gen_tag.get("content"):
        metadata.meta_generator = gen_tag["content"]

    # c. Meta description
    desc_tag = soup.find("meta", attrs={"name": lambda v: v and v.lower() == "description"})
    if desc_tag and desc_tag.get("content"):
        metadata.meta_description = desc_tag["content"]

    # d. Canonical URL
    canonical_tag = soup.find("link", attrs={"rel": lambda v: v and "canonical" in (v if isinstance(v, list) else [v])})
    if canonical_tag and canonical_tag.get("href"):
        metadata.canonical_url = resolve_url(base_url, canonical_tag["href"])

    # e. Charset
    charset_tag = soup.find("meta", attrs={"charset": True})
    if charset_tag:
        metadata.charset = charset_tag.get("charset", "utf-8")
    else:
        ct_tag = soup.find("meta", attrs={"http-equiv": lambda v: v and v.lower() == "content-type"})
        if ct_tag and ct_tag.get("content"):
            content = ct_tag["content"]
            if "charset=" in content.lower():
                metadata.charset = content.lower().split("charset=")[-1].strip()

    # ── 3. Extract all <a href="..."> tags ──
    for tag in soup.find_all("a", href=True):
        href = tag.get("href", "").strip()
        if not href:
            continue
        # Skip non-navigational links
        if any(href.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        resolved = resolve_url(base_url, href)
        text = tag.get_text(strip=True)
        rel = " ".join(tag.get("rel", []))

        metadata.anchor_tags.append(LinkInfo(
            href=resolved,
            text=text,
            rel=rel,
            tag_name=tag.name,
        ))

    # ── 4. Extract <link> tags ──
    for tag in soup.find_all("link", href=True):
        href = tag.get("href", "").strip()
        if not href:
            continue
        resolved = resolve_url(base_url, href)
        rel = " ".join(tag.get("rel", []))
        metadata.link_tags.append(LinkInfo(
            href=resolved,
            text="",
            rel=rel,
            tag_name=tag.name,
        ))

    return metadata


def should_skip_url(url: str) -> bool:
    """
    Check if a URL should be skipped entirely (not even queued).

    Skip if:
    1. URL has no scheme (not http or https)
    2. URL path ends with an extension in SKIP_EXTENSIONS
    3. URL path contains common non-page patterns
    """
    parsed = urlparse(url)

    # 1. Must have http or https scheme
    if parsed.scheme not in ("http", "https"):
        return True

    # 2. Extension check — strip query and fragment
    path = parsed.path.lower()
    # Get the last path component
    last_seg = path.rstrip("/").rsplit("/", 1)[-1]
    if "." in last_seg:
        ext = "." + last_seg.rsplit(".", 1)[-1]
        if ext in SKIP_EXTENSIONS:
            return True

    # 3. Non-page path patterns
    if path.rstrip("/") in ("/feed", "/rss"):
        return True
    if "/feed/" in path or "/rss/" in path:
        return True
    if "/wp-json/" in path:
        return True

    return False


def classify_link(href: str, root_domain: str) -> URLClassification:
    """
    Classify a resolved URL as internal or external.

    Compares the hostname of href to root_domain.

    Rules:
    - Exact match → INTERNAL
    - Subdomain match → INTERNAL
    - www prefix normalization
    - Everything else → EXTERNAL
    """
    parsed = urlparse(href)
    host = parsed.netloc.lower()

    # Strip port if present
    if ":" in host:
        host = host.rsplit(":", 1)[0]

    # Normalize www
    host = host.removeprefix("www.")
    root = root_domain.lower().removeprefix("www.")

    # Exact match
    if host == root:
        return URLClassification.INTERNAL

    # Subdomain match: host ends with ".root_domain"
    if host.endswith("." + root):
        return URLClassification.INTERNAL

    return URLClassification.EXTERNAL

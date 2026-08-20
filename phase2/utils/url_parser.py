"""
URL parsing utilities.
Takes a raw URL string. Returns structured components.
Pure functions. No side effects. No I/O.
"""

from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
from models import ParsedSegment, ParsedURL, URLClassification, SegmentType
from config import ASSET_EXTENSIONS
from typing import Optional
import re


def parse_url(raw_url: str) -> dict:
    """
    Decompose a URL into its components using urllib.parse.

    Input:  "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#section1"
    Output: {
        "scheme": "https",
        "host": "scikit-learn.org",
        "path": "/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        "segments": ["stable", "modules", "generated", "sklearn.linear_model.LogisticRegression.html"],
        "query": {},
        "fragment": "section1"
    }

    EDGE CASES TO HANDLE:
    - URL with no path → path becomes "/"
    - URL with trailing slash → keep trailing slash info but strip for segment parsing
    - URL with query parameters → parse into dict
    - URL with no scheme → assume https
    """
    # Add scheme if missing
    if raw_url and not raw_url.startswith(("http://", "https://", "ftp://")):
        raw_url = "https://" + raw_url

    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower() or "https"
    host = parsed.netloc.lower()

    # Remove default ports
    if host.endswith(":80") and scheme == "http":
        host = host[:-3]
    elif host.endswith(":443") and scheme == "https":
        host = host[:-4]

    path = parsed.path or "/"
    query_dict = parse_qs(parsed.query, keep_blank_values=True)
    # Convert parse_qs result (lists) to flat dict
    query_flat = {k: v[0] if len(v) == 1 else v for k, v in query_dict.items()}

    segments = split_path_segments(path)

    return {
        "scheme": scheme,
        "host": host,
        "path": path,
        "segments": segments,
        "query": query_flat,
        "fragment": parsed.fragment or None,
    }


def split_path_segments(path: str) -> list[str]:
    """
    Split a URL path into its segments.

    Input:  "/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    Output: ["stable", "modules", "generated", "sklearn.linear_model.LogisticRegression.html"]

    RULES:
    - Split on "/"
    - Remove empty strings from result (handles leading slash and trailing slash)
    - Do NOT split on dots within a segment (dots are part of the segment value)
    """
    parts = path.split("/")
    return [p for p in parts if p]


def classify_segment(raw_segment: str) -> SegmentType:
    """
    Determine the lexical type of a single URL path segment.

    ORDER MATTERS. Check in this exact order:
    1. UUID
    2. DATE
    3. VERSION
    4. INTEGER
    5. FLOAT (not filename-like)
    6. HASH
    7. DOTTED_PATH (before FILENAME)
    8. FILENAME (known extension)
    9. CAMEL_CASE
    10. SLUG
    11. Default: LITERAL
    """
    # 1. UUID: exactly 8-4-4-4-12 hex chars
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    if re.match(uuid_pattern, raw_segment, re.IGNORECASE):
        return SegmentType.UUID

    # 2. DATE: YYYY-MM-DD
    date_pattern = r'^\d{4}-\d{2}-\d{2}$'
    if re.match(date_pattern, raw_segment):
        return SegmentType.DATE

    # 3. VERSION: v?digit(.digit)* — must check before INTEGER/FLOAT
    version_pattern = r'^v?\d+(\.\d+)*$'
    if re.match(version_pattern, raw_segment):
        # Distinguish from pure INTEGER (no dots) — if it starts with 'v', always VERSION
        if raw_segment.startswith('v'):
            return SegmentType.VERSION
        # e.g. "0.24", "3.1.2" → VERSION; "123" → INTEGER
        if '.' in raw_segment:
            return SegmentType.VERSION
        # Pure digits fall through to INTEGER check below

    # 4. INTEGER
    if re.match(r'^\d+$', raw_segment):
        return SegmentType.INTEGER

    # 5. FLOAT (digits.digits, NOT a filename extension)
    known_extensions = {'html', 'htm', 'rst', 'md', 'php', 'aspx', 'txt',
                        'json', 'xml', 'csv', 'css', 'js', 'mjs', 'png',
                        'jpg', 'jpeg', 'gif', 'svg', 'ico', 'webp', 'woff',
                        'woff2', 'ttf', 'eot', 'mp4', 'mp3', 'wav', 'zip',
                        'tar', 'gz', 'pdf', 'map', 'inv'}
    float_match = re.match(r'^\d+\.(\d+)$', raw_segment)
    if float_match:
        ext_part = float_match.group(1)
        if ext_part not in known_extensions:
            return SegmentType.FLOAT

    # 6. HASH: all hex chars, 8+ length, no dashes
    if re.match(r'^[0-9a-f]{8,}$', raw_segment, re.IGNORECASE) and '-' not in raw_segment:
        return SegmentType.HASH

    # 7. DOTTED_PATH: identifier.identifier[.identifier...] where each part is
    #    alphanumeric/underscore and the last part is NOT a known extension
    dotted_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$'
    if re.match(dotted_pattern, raw_segment):
        last_part = raw_segment.rsplit('.', 1)[-1].lower()
        if last_part not in known_extensions:
            return SegmentType.DOTTED_PATH

    # 8. FILENAME: ends with a known extension
    filename_pattern = r'^.+\.(html|htm|rst|md|php|aspx|txt|json|xml|csv)$'
    if re.match(filename_pattern, raw_segment, re.IGNORECASE):
        return SegmentType.FILENAME

    # 9. CAMEL_CASE: starts uppercase, only alphanumeric
    if re.match(r'^[A-Z][a-zA-Z0-9]*$', raw_segment):
        return SegmentType.CAMEL_CASE

    # 10. SLUG: lowercase words separated by hyphens only (NOT underscores)
    # user_guide is LITERAL; linear-model is SLUG
    # Must contain at least one hyphen to distinguish from plain LITERAL
    if re.match(r'^[a-z][a-z0-9]*(-[a-z0-9]+)+$', raw_segment):
        return SegmentType.SLUG

    # 11. Default
    return SegmentType.LITERAL


def is_asset_url(url: str) -> bool:
    """
    Check if a URL points to an asset file (CSS, JS, image, font, etc.)

    Uses ASSET_EXTENSIONS from config.py.

    Input:  "https://scikit-learn.org/stable/_static/css/theme.css"
    Output: True

    Input:  "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    Output: False
    """
    # Strip query and fragment before checking extension
    path = url.split('?')[0].split('#')[0]
    # Get the extension of the last path component
    last_segment = path.rstrip('/').rsplit('/', 1)[-1]
    if '.' in last_segment:
        ext = '.' + last_segment.rsplit('.', 1)[-1].lower()
        return ext in ASSET_EXTENSIONS
    return False


def build_parsed_url(
    raw_url: str,
    classification: URLClassification,
    root_domain: str,
    source_url: str = "",
    depth: int = 0,
    link_text: str = ""
) -> ParsedURL:
    """
    Full URL processing. Combines parse_url + classify_segment + is_asset_url.

    This is the main function called by the ingestion pipeline.

    Steps:
    1. Call parse_url(raw_url)
    2. Call is_asset_url(raw_url)
    3. Call split_path_segments(path)
    4. For each segment, call classify_segment(segment) → build ParsedSegment
    5. Return a complete ParsedURL object

    Note: version and version_free_path are NOT filled in here.
    They are filled in by the version_detection pipeline stage.
    """
    parsed = parse_url(raw_url)
    asset = is_asset_url(raw_url)

    path = parsed["path"]
    raw_segments = split_path_segments(path)

    parsed_segments = []
    for i, seg in enumerate(raw_segments):
        seg_type = classify_segment(seg)
        parsed_segments.append(ParsedSegment(
            raw=seg,
            position=i,
            lexical_type=seg_type,
        ))

    # Reconstruct canonical URL (scheme + host + path + query, no fragment)
    from urllib.parse import urlencode
    query_str = ""
    if parsed["query"]:
        query_str = urlencode(sorted(parsed["query"].items()))
    canonical = f"{parsed['scheme']}://{parsed['host']}{path}"
    if query_str:
        canonical += f"?{query_str}"

    return ParsedURL(
        original_url=raw_url,
        canonical_url=canonical,
        domain=parsed["host"],
        scheme=parsed["scheme"],
        path=path,
        segments=parsed_segments,
        query=query_str or None,
        fragment=parsed["fragment"],
        classification=classification,
        is_asset=asset,
        version=None,
        version_free_path=None,
        source_url=source_url,
        depth=depth,
        link_text=link_text,
    )

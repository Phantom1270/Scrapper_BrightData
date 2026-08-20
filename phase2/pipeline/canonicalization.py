"""
Stage 2.2 — URL Canonicalization
Normalize URLs so that equivalent URLs become identical strings.
"""

from models import ParsedURL
from urllib.parse import urlparse, urlunparse, urlencode, parse_qs
from utils.url_parser import split_path_segments, classify_segment, build_parsed_url
from models import URLClassification
import re
import posixpath


# Tracking query parameters to strip
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign",
    "utm_term", "utm_content", "ref", "source",
    "fbclid", "gclid",
}


def canonicalize(url: str) -> str:
    """
    Normalize a URL to its canonical form.

    Transformations (apply in this order):
    1. Parse with urlparse
    2. Lowercase the scheme and host
    3. Remove default ports (:80 for http, :443 for https)
    4. Remove fragment (#...)
    5. Remove tracking query parameters
    6. Sort remaining query parameters alphabetically
    7. Remove trailing slash from path UNLESS path is just "/"
    8. Resolve "." and ".." in path
    9. Remove duplicate slashes: "//" → "/"
    10. Reconstruct URL
    """
    # Add scheme if missing
    if url and not url.startswith(("http://", "https://", "ftp://")):
        url = "https://" + url

    parsed = urlparse(url)

    # 2. Lowercase scheme and host
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # 3. Remove default ports
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # 4. Remove fragment — just don't include it

    # 5 & 6. Filter and sort query params
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        filtered = {k: v for k, v in params.items() if k.lower() not in _TRACKING_PARAMS}
        # Sort and flatten (use first value)
        sorted_params = sorted(
            [(k, v[0] if v else "") for k, v in filtered.items()]
        )
        query_str = urlencode(sorted_params)
    else:
        query_str = ""

    # 7 & 8 & 9. Normalize path
    path = parsed.path or "/"
    # Resolve . and ..
    path = posixpath.normpath(path)
    # normpath strips trailing slash; restore "/" if root
    if path == ".":
        path = "/"
    # Remove duplicate slashes (normpath usually handles this)
    path = re.sub(r"/+", "/", path)
    # 7. Remove trailing slash unless root
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # 10. Reconstruct
    canonical = urlunparse((scheme, netloc, path, "", query_str, ""))
    return canonical


def canonicalize_batch(parsed_urls: list[ParsedURL]) -> list[ParsedURL]:
    """
    Apply canonicalize() to every URL in the list.
    Update canonical_url field on each ParsedURL.
    Also update the path and segments to match the canonical form.

    DEDUPLICATION: After canonicalization, if multiple URLs map to the same
    canonical form, keep only one (prefer lower depth, then longer link_text).

    Return the deduplicated list.
    """
    from urllib.parse import urlparse

    dedup: dict[str, ParsedURL] = {}

    for purl in parsed_urls:
        new_canonical = canonicalize(purl.canonical_url)
        # Update canonical path from the new canonical URL
        new_path = urlparse(new_canonical).path or "/"
        new_segments_raw = split_path_segments(new_path)

        from models import ParsedSegment
        new_segments = []
        for i, seg in enumerate(new_segments_raw):
            seg_type = classify_segment(seg)
            new_segments.append(ParsedSegment(
                raw=seg,
                position=i,
                lexical_type=seg_type,
                is_static=purl.segments[i].is_static if i < len(purl.segments) else None,
                frequency=purl.segments[i].frequency if i < len(purl.segments) else None,
            ))

        updated = purl.model_copy(update={
            "canonical_url": new_canonical,
            "path": new_path,
            "segments": new_segments,
        })

        key = new_canonical
        if key not in dedup:
            dedup[key] = updated
        else:
            existing = dedup[key]
            if updated.depth < existing.depth:
                dedup[key] = updated
            elif updated.depth == existing.depth and len(updated.link_text) > len(existing.link_text):
                dedup[key] = updated

    return list(dedup.values())

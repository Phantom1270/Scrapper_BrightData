"""
Stage 2.3 — Version Detection
Identify and strip version prefixes from documentation URLs.
"""

import re
from models import ParsedURL
from config import VERSION_PATTERN


def extract_version(path: str) -> tuple[str | None, str]:
    """
    Extract version prefix from a URL path.

    Returns: (version, version_free_path)

    Examples:
    "/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    → ("stable", "/modules/generated/sklearn.linear_model.LogisticRegression.html")

    "/0.24/user_guide/linear_model.html"
    → ("0.24", "/user_guide/linear_model.html")

    "/en/stable/install.html"
    → ("stable", "/install.html")  ← language prefix is also stripped

    "/install.html"
    → (None, "/install.html")  ← no version found

    IMPORTANT: Use VERSION_PATTERN from config.py as the regex.
    """
    # Language prefix pattern
    lang_prefix_pattern = r'^/(?:en|fr|de|es|ja|zh|ko)/'
    # Named/numeric version patterns
    version_seg_pattern = r'^(?:stable|dev|latest|master|main|v?\d+(?:\.\d+)*(?:\.\d+)?)$'

    segments = [s for s in path.split('/') if s]

    if not segments:
        return None, path

    idx = 0

    # Check for optional language prefix
    lang_match = re.match(r'^(?:en|fr|de|es|ja|zh|ko)$', segments[0], re.IGNORECASE)
    if lang_match and len(segments) > 1:
        idx = 1

    # Check if current segment is a version
    if idx < len(segments):
        ver_match = re.match(version_seg_pattern, segments[idx], re.IGNORECASE)
        if ver_match:
            version = segments[idx]
            # Build the version-free path from the remaining segments
            remaining = segments[idx + 1:]
            version_free_path = '/' + '/'.join(remaining) if remaining else '/'
            return version, version_free_path

    return None, path


def apply_version_detection(parsed_urls: list[ParsedURL]) -> list[ParsedURL]:
    """
    For every URL, call extract_version and fill in:
    - url.version
    - url.version_free_path

    If no version is detected, version_free_path = path.

    Returns the updated list (mutates in place AND returns for chaining).
    """
    for i, purl in enumerate(parsed_urls):
        version, vfp = extract_version(purl.path)
        parsed_urls[i] = purl.model_copy(update={
            "version": version,
            "version_free_path": vfp if vfp else purl.path,
        })
    return parsed_urls

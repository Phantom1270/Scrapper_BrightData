"""
Stage 2.7 — Coverage Check
Determine how well the derived templates cover all discovered URLs.
Identify uncovered URLs for Phase 3 self-heal.
"""

import logging
from models import (
    ParsedURL, TemplatePattern, CoverageReport, UncoveredURL, ExternalDomain,
    ExternalDomainType
)
from config import TARGET_COVERAGE, MIN_GROUP_SIZE
from utils.url_parser import split_path_segments, classify_segment

logger = logging.getLogger(__name__)


def match_url_to_template(url: ParsedURL, template: TemplatePattern) -> bool:
    """
    Check if a URL matches a template pattern.

    Handles:
    - Literal segments (must match exactly, case-insensitive)
    - <type> placeholders (match by segment type, with loose fallbacks)
    - '...' wildcard (matches any number of remaining segments)
    """
    path = url.version_free_path if url.version_free_path else url.path
    url_segments = split_path_segments(path)

    pattern = template.pattern
    template_segments = split_path_segments(pattern)

    if not template_segments:
        return len(url_segments) == 0

    # Check if template has a '...' wildcard (variable-depth suffix)
    has_wildcard = '...' in template_segments
    
    if has_wildcard:
        # Find the position of '...'
        wildcard_idx = template_segments.index('...')
        # Segments before the wildcard must match
        prefix_segments = template_segments[:wildcard_idx]
        
        # URL must have at least as many segments as the prefix
        if len(url_segments) < len(prefix_segments):
            return False
        
        # Match prefix segments
        for url_seg, tpl_seg in zip(url_segments, prefix_segments):
            if not _segment_matches(url_seg, tpl_seg):
                return False
        
        # The '...' matches any remaining segments
        return True
    
    # No wildcard — exact segment count must match
    if len(url_segments) != len(template_segments):
        return False

    for url_seg, tpl_seg in zip(url_segments, template_segments):
        if not _segment_matches(url_seg, tpl_seg):
            return False

    return True


def _segment_matches(url_seg: str, tpl_seg: str) -> bool:
    """Check if a single URL segment matches a template segment."""
    if tpl_seg.startswith("<") and tpl_seg.endswith(">"):
        # Variable segment — match by type
        inner = tpl_seg[1:-1].lower()
        actual_type = classify_segment(url_seg).value.lower()

        if actual_type == inner:
            return True

        # Loose matching rules
        if inner == "filename" and ("." in url_seg):
            return True
        if inner == "dotted_path" and "." in url_seg:
            return True
        if inner == "literal":
            return True  # <literal> matches anything
        if inner == "unknown":
            return True  # <unknown> matches anything
        if inner == "slug" and actual_type in ("literal", "slug"):
            return True
        if inner == "camel_case" and actual_type in ("camel_case", "literal"):
            return True
        if inner == "version" and actual_type in ("version", "integer"):
            return True
        if inner == "integer" and actual_type == "version":
            return True

        return False
    else:
        # Literal segment — must match exactly (case-insensitive)
        return url_seg.lower() == tpl_seg.lower()


def compute_coverage(urls, templates):
    def specificity(t):
        parts = t.pattern.strip("/").split("/")
        literal_count = sum(1 for p in parts if not p.startswith("<") and p != "...")
        return (-literal_count, -len(parts), -t.member_count)
    sorted_templates = sorted(templates, key=specificity)
    template_map = {t.template_id: [] for t in templates}
    covered_urls = set()
    for url in urls:
        if url.canonical_url in covered_urls:
            continue
        for template in sorted_templates:
            if match_url_to_template(url, template):
                template_map[template.template_id].append(url.canonical_url)
                covered_urls.add(url.canonical_url)
                break
    uncovered = []
    for url in urls:
        if url.canonical_url not in covered_urls:
            path = url.version_free_path or url.path
            uncovered.append(UncoveredURL(
                url=url.canonical_url,
                reason=f"no_template_match (path={path[:80]})",
                recommendation="self_heal",
            ))
    total = len(urls)
    covered = len(covered_urls)
    pct = (covered / total * 100) if total > 0 else 0.0
    tpl_count = len(templates)
    avg = covered / tpl_count if tpl_count > 0 else 0.0
    coverage = CoverageReport(
        total_internal_urls=total, covered_urls=covered,
        uncovered_urls=total - covered, coverage_percent=round(pct, 2),
        template_count=tpl_count, avg_urls_per_template=round(avg, 1),
    )
    return coverage, uncovered, template_map


# ─── Domain classification sets ─────────────────────────────

SOCIAL_DOMAINS = {
    "twitter.com", "x.com", "facebook.com", "linkedin.com",
    "reddit.com", "mastodon.social", "bsky.app",
    "discord.com", "discord.gg",
    "youtube.com", "youtu.be",
}

QA_DOMAINS = {
    "stackoverflow.com", "stackexchange.com",
    "superuser.com", "serverfault.com",
    "askubuntu.com",
}

PACKAGE_INDEX_DOMAINS = {
    "pypi.org", "pip.pypa.io", "npmjs.com", "crates.io",
    "rubygems.org", "packagist.org", "nuget.org",
    "anaconda.org", "conda.io", "conda-forge.org",
}

CODE_HOSTING_DOMAINS = {
    "github.com", "gitlab.com", "bitbucket.org",
    "codeberg.org", "sourceforge.net",
    "github.io",  # GitHub Pages (project sites)
}

ACADEMIC_DOMAINS = {
    "arxiv.org", "doi.org", "scholar.google.com",
    "researchgate.net", "ieee.org", "acm.org",
    "dl.acm.org", "link.springer.com",
}

DOC_PATH_SIGNALS = [
    "/doc/", "/docs/", "/documentation/", "/api/",
    "/reference/", "/stable/", "/en/latest/", "/doc/stable/",
    "/manual/", "/guide/", "/tutorial/",
    "/_static/", "/_sources/", "/objects.inv",
]


def external_domain_analysis(
    external_urls: list[ParsedURL],
) -> list[ExternalDomain]:
    """
    Analyze external URLs by domain.

    Returns list of ExternalDomain objects (not dicts).
    """
    domain_groups: dict[str, list[ParsedURL]] = {}
    for u in external_urls:
        domain_groups.setdefault(u.domain, []).append(u)

    results = []
    for domain, urls in domain_groups.items():
        paths = [u.path for u in urls]
        observed = paths[:10]  # limit

        # Classify domain type using comprehensive domain sets
        classification = _classify_domain(domain, paths)

        ext_domain = ExternalDomain(
            domain=domain,
            url_count=len(urls),
            classification=classification,
            observed_paths=observed,
            note="",
        )
        results.append(ext_domain)

    return results


def _classify_domain(domain: str, paths: list[str]) -> ExternalDomainType:
    """Classify an external domain based on known domain lists and path signals."""
    domain_lower = domain.lower()

    # Check against known domain sets
    if domain_lower in PACKAGE_INDEX_DOMAINS:
        return ExternalDomainType.PACKAGE_INDEX

    if domain_lower in CODE_HOSTING_DOMAINS:
        return ExternalDomainType.CODE_HOSTING

    if domain_lower in SOCIAL_DOMAINS:
        return ExternalDomainType.SOCIAL_MEDIA

    if domain_lower in QA_DOMAINS:
        return ExternalDomainType.QA_FORUM

    if domain_lower in ACADEMIC_DOMAINS:
        return ExternalDomainType.ACADEMIC

    # Check for .github.io subdomains (project documentation)
    if domain_lower.endswith(".github.io"):
        return ExternalDomainType.DOCUMENTATION_CROSSREF

    # Check for .readthedocs.io / .readthedocs.org subdomains
    if domain_lower.endswith(".readthedocs.io") or domain_lower.endswith(".readthedocs.org"):
        return ExternalDomainType.DOCUMENTATION_CROSSREF

    # Check paths for documentation signals
    is_doc = any(
        any(sig in p for sig in DOC_PATH_SIGNALS)
        for p in paths
    )
    if is_doc:
        return ExternalDomainType.DOCUMENTATION_CROSSREF

    # Root-only links are likely project homepages
    if all(p in ("", "/") for p in paths):
        return ExternalDomainType.PROJECT_HOMEPAGE

    # If domain contains common doc/project keywords
    if any(kw in domain_lower for kw in ("docs.", "doc.", "wiki.", "documentation")):
        return ExternalDomainType.DOCUMENTATION_CROSSREF

    # Default: classify as documentation_crossref if it looks like a project site
    # (has meaningful paths, not just root)
    if any(p.count("/") >= 2 for p in paths):
        return ExternalDomainType.DOCUMENTATION_CROSSREF

    return ExternalDomainType.UNKNOWN


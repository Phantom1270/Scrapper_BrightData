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

    Algorithm:
    1. Get the URL's path (use version_free_path if available, else path)
    2. Split into segments
    3. Split the template pattern into segments
    4. If segment counts differ → no match
    5. For each position: literal must match exactly, <type> must match by type
    6. All positions must match → return True
    """
    path = url.version_free_path if url.version_free_path else url.path
    url_segments = split_path_segments(path)

    # Template pattern starts with "/" — split it
    pattern = template.pattern
    template_segments = split_path_segments(pattern)

    if len(url_segments) != len(template_segments):
        return False

    for url_seg, tpl_seg in zip(url_segments, template_segments):
        if tpl_seg.startswith("<") and tpl_seg.endswith(">"):
            # Variable segment — match by type
            inner = tpl_seg[1:-1].lower()
            # Handle compound patterns like "<dotted_path>.<camel_case>.html"
            # For simple <type> placeholders:
            actual_type = classify_segment(url_seg).value.lower()
            if actual_type != inner:
                # Allow loose matching: if it's a filename and the template has <filename>, ok
                if inner == "filename" and ("." in url_seg):
                    pass  # Accept
                elif inner == "dotted_path" and "." in url_seg:
                    pass  # Accept dotted paths loosely
                elif inner == "literal":
                    pass  # Accept anything as literal
                elif inner == "unknown":
                    pass  # Accept anything as unknown
                elif inner == "slug" and actual_type in ("literal", "slug"):
                    pass  # Accept
                elif inner == "camel_case" and actual_type in ("camel_case", "literal"):
                    pass
                else:
                    return False
        else:
            # Literal segment — must match exactly
            if url_seg.lower() != tpl_seg.lower():
                return False

    return True


def compute_coverage(
    urls: list[ParsedURL],
    templates: list[TemplatePattern],
) -> tuple[CoverageReport, list[UncoveredURL], dict[str, list[str]]]:
    """
    Main coverage computation.

    Returns:
    1. CoverageReport with statistics
    2. List of UncoveredURL objects (for Phase 3 self-heal)
    3. template_map: dict mapping template_id → list of matched URL strings

    IMPORTANT: A URL should match at most ONE template.
    Match the most specific template first (deepest path pattern).
    """
    # Sort templates by specificity: more segments = more specific
    def template_specificity(t: TemplatePattern) -> int:
        segs = split_path_segments(t.pattern)
        # Count literal segments (more literals = more specific)
        literal_count = sum(1 for s in segs if not (s.startswith("<") and s.endswith(">")))
        return len(segs) * 10 + literal_count

    sorted_templates = sorted(templates, key=template_specificity, reverse=True)

    template_map: dict[str, list[str]] = {t.template_id: [] for t in templates}
    uncovered_urls: list[UncoveredURL] = []
    covered_count = 0

    for url in urls:
        matched = False
        for tpl in sorted_templates:
            if match_url_to_template(url, tpl):
                template_map[tpl.template_id].append(url.canonical_url)
                matched = True
                covered_count += 1
                break

        if not matched:
            uncovered_urls.append(UncoveredURL(
                url=url.canonical_url,
                reason="no_template_match",
                recommendation="self_heal",
            ))

    total = len(urls)
    coverage_pct = round((covered_count / total * 100) if total > 0 else 0.0, 2)

    if coverage_pct < TARGET_COVERAGE * 100:
        logger.warning(
            f"Coverage {coverage_pct:.1f}% is below target {TARGET_COVERAGE * 100:.0f}%"
        )

    avg_per_template = round(covered_count / len(templates) if templates else 0.0, 2)

    report = CoverageReport(
        total_internal_urls=total,
        covered_urls=covered_count,
        uncovered_urls=len(uncovered_urls),
        coverage_percent=coverage_pct,
        template_count=len(templates),
        avg_urls_per_template=avg_per_template,
    )

    return report, uncovered_urls, template_map


def external_domain_analysis(
    external_urls: list[ParsedURL],
) -> list[dict]:
    """
    Analyze external URLs by domain.

    Steps:
    1. Group external URLs by domain
    2. For each domain, count URLs and classify
    3. Return list of ExternalDomain dicts
    """
    domain_groups: dict[str, list[ParsedURL]] = {}
    for u in external_urls:
        domain_groups.setdefault(u.domain, []).append(u)

    results = []
    for domain, urls in domain_groups.items():
        paths = [u.path for u in urls]
        observed = paths[:10]  # limit

        # Classify domain type
        if domain in ("pypi.org", "pip.pypa.io"):
            classification = ExternalDomainType.PACKAGE_INDEX
        elif domain in ("github.com", "gitlab.com", "bitbucket.org"):
            classification = ExternalDomainType.CODE_HOSTING
        else:
            # Check paths for documentation signals
            doc_signals = ["/doc/", "/docs/", "/documentation/", "/api/",
                           "/reference/", "/stable/", "/en/latest/", "/doc/stable/"]
            is_doc = any(
                any(sig in p for sig in doc_signals)
                for p in paths
            )
            if is_doc:
                classification = ExternalDomainType.DOCUMENTATION_CROSSREF
            elif all(p in ("", "/") for p in paths):
                classification = ExternalDomainType.PROJECT_HOMEPAGE
            else:
                classification = ExternalDomainType.UNKNOWN

        ext_domain = ExternalDomain(
            domain=domain,
            url_count=len(urls),
            classification=classification,
            observed_paths=observed,
            note="",
        )
        results.append(ext_domain.model_dump())

    return results

"""
Stage 2.8 — Output Generation
Assemble the final Phase 2 output JSON.
"""

import json
from pathlib import Path
from models import (
    Phase2Output, Phase1Output, TemplatePattern,
    CoverageReport, UncoveredURL, ExternalDomain
)
from config import MAX_EXAMPLES_PER_TEMPLATE, MAX_UNCOVERED_IN_OUTPUT


def build_output(
    phase1: Phase1Output,
    generator_detected: str | None,
    generator_confidence: float,
    templates: list[TemplatePattern],
    coverage: CoverageReport,
    uncovered: list[UncoveredURL],
    external_domains: list[dict],
    template_map: dict[str, list[str]],
) -> Phase2Output:
    """
    Assemble the complete Phase 2 output.

    Steps:
    1. Create Phase2Output with all fields
    2. Trim example_urls in each template to MAX_EXAMPLES_PER_TEMPLATE
    3. Trim uncovered_urls to MAX_UNCOVERED_IN_OUTPUT
    4. Sort templates by member_count descending
    5. Return Phase2Output
    """
    # 2. Trim example URLs
    trimmed_templates = []
    for tpl in templates:
        trimmed = tpl.model_copy(update={
            "example_urls": tpl.example_urls[:MAX_EXAMPLES_PER_TEMPLATE]
        })
        trimmed_templates.append(trimmed)

    # 4. Sort templates by member_count descending
    trimmed_templates.sort(key=lambda t: t.member_count, reverse=True)

    # 3. Trim uncovered
    trimmed_uncovered = uncovered[:MAX_UNCOVERED_IN_OUTPUT]

    # Convert external_domains dicts back to ExternalDomain objects
    ext_domain_objs = [ExternalDomain(**d) for d in external_domains]

    return Phase2Output(
        crawl_id=phase1.crawl_id,
        root_domain=phase1.root_domain,
        generator_detected=generator_detected,
        generator_confidence=generator_confidence,
        summary=coverage,
        templates=trimmed_templates,
        uncovered_urls=trimmed_uncovered,
        external_domains=ext_domain_objs,
        template_map=template_map,
    )


def save_output(output: Phase2Output, output_path: str) -> None:
    """
    Write Phase2Output to a JSON file.

    Use model_dump_json with indent=2 for readability.
    Create parent directories if they don't exist.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output.model_dump_json(indent=2), encoding="utf-8")


def print_summary(output: Phase2Output) -> None:
    """
    Print a human-readable summary to stdout.

    Format:
    ═══════════════════════════════════════════════
    PHASE 2 RESULTS — scikit-learn.org
    ═══════════════════════════════════════════════
    Generator detected: sphinx (confidence: 0.95)
    Coverage: 217/235 URLs (92.3%) with 4 templates
    Templates:
      tpl_001  modules/generated/<>.<>.html  (143 URLs)  ← api_reference
      ...
    Uncovered: 18 URLs → self-heal
    External domains: numpy.org (5), scipy.org (3)
    ═══════════════════════════════════════════════
    """
    sep = "=" * 53
    print(sep)
    print(f"PHASE 2 RESULTS — {output.root_domain}")
    print(sep)
    print()

    if output.generator_detected:
        print(f"Generator detected: {output.generator_detected} "
              f"(confidence: {output.generator_confidence:.2f})")
    else:
        print("Generator detected: unknown")
    print()

    s = output.summary
    print(f"Coverage: {s.covered_urls}/{s.total_internal_urls} URLs "
          f"({s.coverage_percent:.1f}%) with {s.template_count} templates")
    print()

    print("Templates:")
    for tpl in output.templates:
        name_tag = f"← {tpl.name}" if tpl.name else ""
        print(f"  {tpl.template_id}  {tpl.pattern}  ({tpl.member_count} URLs)  {name_tag}")

    print()
    print(f"Uncovered: {len(output.uncovered_urls)} URLs → self-heal")

    if output.external_domains:
        ext_summary = ", ".join(
            f"{d.domain} ({d.url_count})" for d in output.external_domains[:5]
        )
        print()
        print(f"External domains: {ext_summary}")

    print()
    print(sep)

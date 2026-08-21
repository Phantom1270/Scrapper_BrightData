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
    external_domains: list[ExternalDomain],
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

    return Phase2Output(
        crawl_id=phase1.crawl_id,
        root_domain=phase1.root_domain,
        generator_detected=generator_detected,
        generator_confidence=generator_confidence,
        summary=coverage,
        templates=trimmed_templates,
        uncovered_urls=trimmed_uncovered,
        external_domains=external_domains,
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
    width = 65
    print("=" * width)
    print(f"PHASE 2 RESULTS - {output.root_domain}")
    print("=" * width)
    print()
    gen = output.generator_detected or "unknown"
    print(f"Generator: {gen} (confidence: {output.generator_confidence:.2f})")
    print()
    s = output.summary
    print(f"Coverage: {s.covered_urls}/{s.total_internal_urls} URLs ({s.coverage_percent:.1f}%) with {s.template_count} templates")
    print()
    print("Templates:")
    print("-" * width)
    for t in output.templates:
        name_str = f"  <- {t.name}" if t.name else ""
        print(f"  {t.template_id}  {t.pattern:<55s} ({t.member_count:>4d} URLs){name_str}")
    print("-" * width)
    print()
    print(f"Uncovered: {len(output.uncovered_urls)} URLs -> self-heal")
    if output.uncovered_urls:
        print("  Sample:")
        for u in output.uncovered_urls[:5]:
            print(f"    {u.url}")
    print()
    if output.external_domains:
        print(f"External domains: {len(output.external_domains)}")
        for d in output.external_domains[:8]:
            print(f"  {d.domain} ({d.url_count}) [{d.classification.value}]")
    print("=" * width)

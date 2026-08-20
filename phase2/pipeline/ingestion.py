"""
Stage 2.1 — Ingestion
Read Phase 1 JSON. Separate internal from external. Filter assets.
Build ParsedURL objects for every URL.
"""

import json
import logging
from pathlib import Path
from models import Phase1Output, ParsedURL, URLClassification
from utils.url_parser import build_parsed_url, is_asset_url

logger = logging.getLogger(__name__)


def ingest(phase1_json_path: str) -> dict:
    """
    Main ingestion function.

    Input:  Path to Phase 1 output JSON file.
    Output: {
        "parsed_internal": list[ParsedURL],   # Internal page URLs (not assets)
        "parsed_external": list[ParsedURL],   # External URLs
        "asset_urls": list[str],              # Internal asset URLs (filtered out)
        "phase1": Phase1Output                # Original Phase 1 data
    }

    Steps:
    1. Read and validate JSON against Phase1Output model
    2. For each internal URL:
       a. Call build_parsed_url with classification=INTERNAL
       b. If is_asset_url → add to asset_urls list, skip
       c. Otherwise → add to parsed_internal list
    3. For each external URL:
       a. Call build_parsed_url with classification=EXTERNAL
       b. If is_asset_url → add to asset_urls list, skip
       c. Otherwise → add to parsed_external list
    4. Log counts: how many internal, external, assets
    5. Return the dict

    DEDUPLICATION:
    If the same canonical URL appears multiple times, keep the one with
    the lowest depth. If same depth, keep the one with the most link_text.
    """
    path = Path(phase1_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 input file not found: {phase1_json_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {phase1_json_path}: {e}") from e

    phase1 = Phase1Output.model_validate(data)

    parsed_internal: list[ParsedURL] = []
    parsed_external: list[ParsedURL] = []
    asset_urls: list[str] = []

    # ── Internal URLs ──
    # Deduplicate: canonical_url → best ParsedURL
    internal_dedup: dict[str, ParsedURL] = {}

    for disc in phase1.internal_urls:
        purl = build_parsed_url(
            raw_url=disc.url,
            classification=URLClassification.INTERNAL,
            root_domain=phase1.root_domain,
            source_url=disc.source_url,
            depth=disc.depth,
            link_text=disc.link_text,
        )
        if purl.is_asset:
            asset_urls.append(disc.url)
            continue

        key = purl.canonical_url
        if key not in internal_dedup:
            internal_dedup[key] = purl
        else:
            existing = internal_dedup[key]
            # Keep lowest depth; on tie, keep longest link_text
            if purl.depth < existing.depth:
                internal_dedup[key] = purl
            elif purl.depth == existing.depth and len(purl.link_text) > len(existing.link_text):
                internal_dedup[key] = purl

    parsed_internal = list(internal_dedup.values())

    # ── External URLs ──
    external_dedup: dict[str, ParsedURL] = {}

    for disc in phase1.external_urls:
        purl = build_parsed_url(
            raw_url=disc.url,
            classification=URLClassification.EXTERNAL,
            root_domain=phase1.root_domain,
            source_url=disc.source_url,
            depth=disc.depth,
            link_text=disc.link_text,
        )
        if purl.is_asset:
            asset_urls.append(disc.url)
            continue

        key = purl.canonical_url
        if key not in external_dedup:
            external_dedup[key] = purl
        else:
            existing = external_dedup[key]
            if purl.depth < existing.depth:
                external_dedup[key] = purl
            elif purl.depth == existing.depth and len(purl.link_text) > len(existing.link_text):
                external_dedup[key] = purl

    parsed_external = list(external_dedup.values())

    logger.info(
        f"Ingested: {len(parsed_internal)} internal pages, "
        f"{len(parsed_external)} external URLs, "
        f"{len(asset_urls)} assets filtered"
    )

    return {
        "parsed_internal": parsed_internal,
        "parsed_external": parsed_external,
        "asset_urls": asset_urls,
        "phase1": phase1,
    }

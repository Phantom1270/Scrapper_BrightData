"""
Phase 2 — Entry Point

Usage:
    python main.py input.json
    python main.py input.json --output output.json
    python main.py input.json --output output.json --verbose

Reads Phase 1 JSON, runs the full pipeline, writes Phase 2 JSON.
"""

import sys
import argparse
import json
import logging
from pathlib import Path

from pipeline.ingestion import ingest
from pipeline.canonicalization import canonicalize_batch
from pipeline.version_detection import apply_version_detection
from pipeline.generator_detection import detect_generator
from pipeline.structural_grouping import discover_groups, finalize_templates
from pipeline.coverage import compute_coverage, external_domain_analysis
from pipeline.output import build_output, save_output, print_summary


def run_pipeline(input_path: str, output_path: str, verbose: bool = False) -> None:
    """
    Run the complete Phase 2 pipeline.

    Steps:
    1. INGEST → load JSON, parse URLs, separate internal/external, filter assets
    2. CANONICALIZE → normalize all URL strings, deduplicate
    3. VERSION DETECTION → strip version prefixes, set version_free_path
    4. GENERATOR DETECTION → guess Sphinx/MkDocs/Docusaurus
    5. STRUCTURAL GROUPING → build trie, discover groups, derive templates
    6. COVERAGE → match URLs to templates, find uncovered
    7. EXTERNAL ANALYSIS → classify external domains
    8. OUTPUT → assemble JSON, write file, print summary

    ERROR HANDLING:
    - If input file doesn't exist → print error, exit 1
    - If JSON is invalid → print error with line number, exit 1
    - If no internal URLs found → print warning, output empty template map
    - If any pipeline stage raises an exception → print the error with
      the stage name, re-raise
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Validate input file
    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Step 1
    if verbose:
        print("[1/8] Ingesting Phase 1 output...")
    try:
        ingested = ingest(input_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR (JSON parse): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR in stage INGEST: {e}", file=sys.stderr)
        raise

    if not ingested["parsed_internal"]:
        print("WARNING: No internal URLs found in Phase 1 input.")

    # Step 2
    if verbose:
        print(f"[2/8] Canonicalizing {len(ingested['parsed_internal'])} internal URLs...")
    try:
        internal = canonicalize_batch(ingested["parsed_internal"])
        external = canonicalize_batch(ingested["parsed_external"])
    except Exception as e:
        print(f"ERROR in stage CANONICALIZE: {e}", file=sys.stderr)
        raise

    # Step 3
    if verbose:
        print("[3/8] Detecting version prefixes...")
    try:
        internal = apply_version_detection(internal)
        external = apply_version_detection(external)
    except Exception as e:
        print(f"ERROR in stage VERSION_DETECTION: {e}", file=sys.stderr)
        raise

    # Step 4
    if verbose:
        print("[4/8] Detecting documentation generator...")
    try:
        generator, gen_confidence = detect_generator(ingested["phase1"], internal)
        if verbose:
            print(f"       → {generator or 'unknown'} (confidence: {gen_confidence:.2f})")
    except Exception as e:
        print(f"ERROR in stage GENERATOR_DETECTION: {e}", file=sys.stderr)
        raise

    # Step 5
    if verbose:
        print("[5/8] Discovering URL groups and deriving templates...")
    try:
        groups = discover_groups(internal, generator_hint=generator)
        templates = finalize_templates(groups, internal)
        if verbose:
            print(f"       → {len(templates)} templates derived")
    except Exception as e:
        print(f"ERROR in stage STRUCTURAL_GROUPING: {e}", file=sys.stderr)
        raise

    # Step 6
    if verbose:
        print("[6/8] Computing coverage...")
    try:
        coverage, uncovered, template_map = compute_coverage(internal, templates)
        if verbose:
            print(f"       → {coverage.coverage_percent:.1f}% coverage")
            print(f"       → {len(uncovered)} uncovered URLs")
    except Exception as e:
        print(f"ERROR in stage COVERAGE: {e}", file=sys.stderr)
        raise

    # Step 7
    if verbose:
        print("[7/8] Analyzing external domains...")
    try:
        ext_domains = external_domain_analysis(external)
        if verbose:
            print(f"       → {len(ext_domains)} external domains")
    except Exception as e:
        print(f"ERROR in stage EXTERNAL_ANALYSIS: {e}", file=sys.stderr)
        raise

    # Step 8
    if verbose:
        print("[8/8] Writing output...")
    try:
        output = build_output(
            phase1=ingested["phase1"],
            generator_detected=generator,
            generator_confidence=gen_confidence,
            templates=templates,
            coverage=coverage,
            uncovered=uncovered,
            external_domains=ext_domains,
            template_map=template_map,
        )
        save_output(output, output_path)
        print_summary(output)
    except Exception as e:
        print(f"ERROR in stage OUTPUT: {e}", file=sys.stderr)
        raise

    if verbose:
        print(f"\nOutput written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: URL Structure Classifier"
    )
    parser.add_argument(
        "input",
        help="Path to Phase 1 output JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for Phase 2 output JSON (default: input_dir/phase2_output.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to stdout"
    )

    args = parser.parse_args()

    # Default output path
    output_path = args.output
    if output_path is None:
        input_stem = Path(args.input).stem
        output_path = str(Path(args.input).parent / f"{input_stem}_phase2.json")

    run_pipeline(args.input, output_path, verbose=args.verbose)


if __name__ == "__main__":
    main()

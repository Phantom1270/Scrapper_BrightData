"""
Phase 3 — Fully Automated Bright Data CLI Pipeline
Hackathon Hero Project Edition
"""

import os
import sys
import json
import argparse
from pathlib import Path
import dotenv
from typing import Any, Union
from google import genai
from pydantic import BaseModel, Field

from models import Phase2Output
from utils.scraper_studio_client import ScraperStudioCLI
from config import (
    DEFAULT_CREATE_PROMPT, 
    DEFAULT_HEAL_PROMPT, 
    MAX_HEAL_ATTEMPTS,
    SCHEMA_FIELDS,
    GARBAGE_STRINGS,
    LLM_VALIDATION_ENABLED,
    LLM_API_KEY,
    LLM_VALIDATION_TIMEOUT
)

# ─── Validation Functions ────────────────────────────────────

def score_field(value: Any, rules: dict) -> str:
    """Scores a field based on rules and garbage strings."""
    if value is None:
        return "empty"
        
    if isinstance(value, (list, dict)):
        val_str = json.dumps(value)
        if len(value) == 0:
            val_str = ""
    else:
        val_str = str(value).strip()
        
    if not val_str:
        if rules.get("min_length", 1) == 0:
            return "ok" # Optional field
        return "empty"
        
    if val_str.lower() in [g.lower() for g in GARBAGE_STRINGS]:
        return "garbage"
        
    if len(val_str) < rules.get("min_length", 0):
        return "too_short"
        
    return "ok"

def validate_extraction(data: dict) -> str:
    """Returns PASS, BORDERLINE, or FAIL."""
    if isinstance(data, list):
        if len(data) == 0:
            return "FAIL"
        target = data[0]
    elif isinstance(data, dict):
        if "raw_output" in data:
            return "FAIL"
        target = data
    else:
        return "FAIL"
        
    field_scores = {}
    for field, rules in SCHEMA_FIELDS.items():
        val = target.get(field)
        field_scores[field] = score_field(val, rules)
        
    fail_count = sum(1 for s in field_scores.values() if s in ("empty", "garbage"))
    short_count = sum(1 for s in field_scores.values() if s == "too_short")
    
    if fail_count >= len(SCHEMA_FIELDS) // 2:
        return "FAIL"
        
    if fail_count == 0 and short_count == 0:
        return "PASS"
        
    return "BORDERLINE"

class LLMValidationResponse(BaseModel):
    valid: bool = Field(description="True if the JSON contains meaningful content according to the expected schema, False otherwise")
    reason: str = Field(description="Short explanation of why it is valid or invalid")

def llm_validate(data: Union[dict, list], api_key: str) -> bool:
    """Uses Gemini to evaluate borderline extractions."""
    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            "Analyze the following JSON extracted from a webpage. "
            "Does it contain meaningful, valid content for a software documentation page? "
            f"JSON: {json.dumps(data)}"
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LLMValidationResponse,
                temperature=0.0
            )
        )
        result = json.loads(response.text)
        return result.get("valid", False)
    except Exception as e:
        print(f"      [LLM Warning] LLM validation failed/timed out: {e}")
        return False

def run_pipeline(phase2_path: str, output_path: str, verbose: bool = False, limit: int = 0) -> None:
    # 1. Load Phase 2
    if verbose:
        print("=" * 60)
        print("PHASE 3 - Bright Data CLI Orchestrator")
        print("=" * 60)
        print()
        print("[1/3] Loading Phase 2 output...")

    with open(phase2_path, encoding="utf-8") as f:
        phase2_data = json.load(f)
    phase2 = Phase2Output(**phase2_data)

    total_urls = sum(len(v) for v in phase2.template_map.values())
    if verbose:
        print(f"  Domain: {phase2.root_domain}")
        print(f"  Templates: {len(phase2.templates)}")
        print(f"  Total URLs available: {total_urls}")
        print()

    if total_urls == 0:
        print("No URLs to scrape. Exiting.")
        return

    cli = ScraperStudioCLI()
    
    # Track results
    results = {}  # tid -> list of dicts with url and data
    failed_urls = []
    total_processed = 0
    total_healed = 0
    
    # 2. Iterate templates
    for template in phase2.templates:
        tid = template.template_id
        urls = phase2.template_map.get(tid, [])
        if not urls:
            continue
            
        if verbose:
            print(f"\n[2/3] Processing Template {tid} ({len(urls)} URLs)")
            
        # Step A: Create Scraper
        first_url = urls[0]
        if verbose:
            print(f"  -> Creating Scraper via CLI using first URL: {first_url}")
            
        collector_id = cli.create_scraper(first_url, DEFAULT_CREATE_PROMPT)
        if not collector_id:
            print(f"  [!] Failed to generate scraper for {tid}. Skipping template.")
            continue
            
        if verbose:
            print(f"  -> Successfully generated Collector ID: {collector_id}")
            
        results[tid] = []
        
        # Step B: Run (and optionally Heal) remaining URLs
        processed_in_this_template = 0
        for url in urls:
            if limit > 0 and processed_in_this_template >= limit:
                if verbose:
                    print(f"  [!] Template limit of {limit} reached. Moving to next template.")
                break
                
            if verbose:
                print(f"  -> Running scraper on {url}...")
                
            attempt = 0
            healed = False
            final_data = None
            
            while attempt <= MAX_HEAL_ATTEMPTS:
                data = cli.run_scraper(collector_id, url)
                
                # Check 1: Is it totally empty or a CLI error?
                is_invalid = not data or len(data) == 0 or (isinstance(data, dict) and "raw_output" in data)
                
                if not is_invalid:
                    # Check 2: Our 3-tier validation
                    status = validate_extraction(data)
                    
                    if status == "PASS":
                        final_data = data
                        break
                        
                    if status == "BORDERLINE":
                        if verbose:
                            print("    [!] Extraction is BORDERLINE. Checking LLM escalation...")
                        
                        if LLM_VALIDATION_ENABLED and LLM_API_KEY:
                            if llm_validate(data, LLM_API_KEY):
                                if verbose:
                                    print("      -> LLM approved the borderline data.")
                                final_data = data
                                break
                            else:
                                if verbose:
                                    print("      -> LLM rejected the borderline data.")
                        else:
                            if verbose:
                                print("      -> LLM validation disabled/no key. Defaulting to FAIL.")
                
                # If we get here, it failed validation (or is_invalid)
                if attempt < MAX_HEAL_ATTEMPTS:
                    if verbose:
                        print(f"    [!] Extraction failed validation. Triggering self-heal (Attempt {attempt+1}/{MAX_HEAL_ATTEMPTS})...")
                    success = cli.heal_scraper(collector_id, DEFAULT_HEAL_PROMPT)
                    if success and verbose:
                        print(f"    -> Heal approved. Re-running scraper...")
                else:
                    if verbose:
                        print(f"    [!] Permanently failed after {MAX_HEAL_ATTEMPTS} heal attempts.")
                    
                attempt += 1

            if final_data:
                results[tid].append({
                    "url": url,
                    "status": "extracted",
                    "data": final_data
                })
                if attempt > 0:
                    total_healed += 1
            else:
                failed_urls.append({"url": url, "template_id": tid})
                
            total_processed += 1
            processed_in_this_template += 1

    # 3. Output results
    if verbose:
        print("\n[3/3] Saving Final Output...")
        
    final_output = {
        "domain": phase2.root_domain,
        "total_processed": total_processed,
        "total_healed": total_healed,
        "failed": len(failed_urls),
        "results": results,
        "failed_urls": failed_urls
    }
    
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    if verbose:
        print(f"Success! Results saved to {output_path}")
        print(f"Summary: Processed {total_processed}, Healed {total_healed}, Failed {len(failed_urls)}")

def main():
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(description="Phase 3: CLI Orchestrator")
    parser.add_argument("--phase2", required=True, help="Path to phase2_output.json")
    parser.add_argument("--output", "-o", default="phase3_output.json", help="Output path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    parser.add_argument("--limit", type=int, default=0, help="Max URLs to process per template")

    args = parser.parse_args()
    run_pipeline(args.phase2, args.output, verbose=args.verbose, limit=args.limit)

if __name__ == "__main__":
    main()

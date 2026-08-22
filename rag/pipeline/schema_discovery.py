"""
Schema discovery.

Profiles a scraped Phase 3 JSON file before normalization, producing a
human-readable (and machine-readable) report of what fields each template
contains, their value types, and sample values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.pipeline.field_classifier import FieldClassifier

logger = logging.getLogger(__name__)

# Content vs metadata: roles that carry page content (vs structural metadata)
_CONTENT_ROLES: frozenset[str] = frozenset({
    "title", "description", "signature", "parameter", "code", "note",
    "section", "relation", "introduction", "install", "notebook",
})


class SchemaDiscovery:
    """
    Analyse a Phase 3 scraped JSON file and produce a structural report.

    Usage::

        sd = SchemaDiscovery("path/to/phase3_output.json")
        report = sd.discover()
        sd.print_report()
    """

    def __init__(self, settings=None, scraped_json_path: str = None) -> None:
        if isinstance(settings, str):
            scraped_json_path = settings
            settings = None
            
        if scraped_json_path is None:
            raise ValueError("scraped_json_path must be provided")
            
        self._path = Path(scraped_json_path)
        self._raw: dict = {}
        self._classifier = FieldClassifier()
        self._report: Optional[dict] = None

        if not self._path.exists():
            raise FileNotFoundError(f"Scraped JSON not found: {self._path}")

        with self._path.open("r", encoding="utf-8") as fh:
            self._raw = json.load(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(self) -> dict:
        """
        Analyse the file and return a schema report dict.

        The report structure::

            {
              "domain": str,
              "total_processed": int,
              "total_healed": int,
              "failed": int,
              "templates": {
                "tpl_XXX": {
                  "entry_count": int,
                  "extracted_count": int,
                  "error_count": int,
                  "error_types": {str: int},
                  "fields": {
                    "field_name": {
                      "count": int,
                      "types": [str, ...],
                      "is_array": bool,
                      "array_item_fields": [str, ...],
                      "sample_values": [str, ...],
                    }
                  },
                  "content_fields": [str, ...],
                  "metadata_fields": [str, ...],
                }
              }
            }
        """
        results = self._raw.get("results", {}) or {}

        templates_report: Dict[str, dict] = {}
        for tpl_id, entries in results.items():
            templates_report[tpl_id] = self._analyse_template(tpl_id, entries or [])

        self._report = {
            "domain": self._raw.get("domain", ""),
            "total_processed": self._raw.get("total_processed", 0),
            "total_healed": self._raw.get("total_healed", 0),
            "failed": self._raw.get("failed", 0),
            "templates": templates_report,
        }
        return self._report

    def print_report(self) -> None:
        """Print a human-readable summary to stdout."""
        if self._report is None:
            self.discover()
        r = self._report
        print(f"\n{'='*60}")
        print(f"Schema Discovery Report — {r['domain']}")
        print(f"{'='*60}")
        print(f"  Total processed : {r['total_processed']}")
        print(f"  Total healed    : {r['total_healed']}")
        print(f"  Failed          : {r['failed']}")
        print()

        for tpl_id, tdata in r["templates"].items():
            print(f"  Template: {tpl_id}")
            print(f"    Entries    : {tdata['entry_count']}")
            print(f"    Extracted  : {tdata['extracted_count']}")
            print(f"    Errors     : {tdata['error_count']}")
            if tdata["error_types"]:
                for ecode, cnt in tdata["error_types"].items():
                    print(f"      [{ecode}] × {cnt}")
            print(f"    Fields ({len(tdata['fields'])}):")
            for fname, fdata in tdata["fields"].items():
                arr_info = f" [array → {fdata['array_item_fields']}]" if fdata["is_array"] else ""
                print(f"      {fname}: {fdata['types']}{arr_info} (n={fdata['count']})")
            print(f"    Content fields  : {tdata['content_fields']}")
            print(f"    Metadata fields : {tdata['metadata_fields']}")
            print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyse_template(self, tpl_id: str, entries: list) -> dict:
        field_registry: Dict[str, dict] = {}
        extracted = 0
        errors = 0
        error_types: Dict[str, int] = {}

        for entry in entries:
            status = entry.get("status", "")
            if status == "extracted":
                extracted += 1
            elif status == "failed":
                errors += 1
                ecode = entry.get("error", "unknown")
                # Normalise long error messages to a short key
                short = str(ecode)[:60]
                error_types[short] = error_types.get(short, 0) + 1

            for data_item in entry.get("data") or []:
                self._analyze_item(data_item, field_registry)

        # Determine content vs metadata fields
        content_fields: List[str] = []
        metadata_fields: List[str] = []

        if field_registry:
            # Use a representative item to classify
            sample_item = {fname: None for fname in field_registry}
            roles = self._classifier.classify_fields(sample_item)
            content_role_fields: set[str] = set()
            for role in _CONTENT_ROLES:
                for fname in roles.get(role, []):
                    content_role_fields.add(fname)
            for fname in field_registry:
                if fname in content_role_fields:
                    content_fields.append(fname)
                else:
                    metadata_fields.append(fname)

        return {
            "entry_count": len(entries),
            "extracted_count": extracted,
            "error_count": errors,
            "error_types": error_types,
            "fields": field_registry,
            "content_fields": sorted(content_fields),
            "metadata_fields": sorted(metadata_fields),
        }

    def _analyze_item(self, item: dict, field_registry: dict) -> None:
        """Walk a data item, recording field presence, types, and samples."""
        for fname, fvalue in item.items():
            if fname not in field_registry:
                field_registry[fname] = {
                    "count": 0,
                    "types": [],
                    "is_array": False,
                    "array_item_fields": [],
                    "sample_values": [],
                }
            rec = field_registry[fname]
            rec["count"] += 1

            type_name = self._type_name(fvalue)
            if type_name not in rec["types"]:
                rec["types"].append(type_name)

            if isinstance(fvalue, list):
                rec["is_array"] = True
                # Inspect up to 5 items for sub-structure
                for sub in fvalue[:5]:
                    if isinstance(sub, dict):
                        for sub_key in sub:
                            if sub_key not in rec["array_item_fields"]:
                                rec["array_item_fields"].append(sub_key)

            # Collect up to 3 sample values (truncated to 80 chars)
            if len(rec["sample_values"]) < 3 and fvalue:
                sample = self._to_sample_str(fvalue)
                if sample and sample not in rec["sample_values"]:
                    rec["sample_values"].append(sample)

    @staticmethod
    def _type_name(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"

    @staticmethod
    def _to_sample_str(value: Any) -> str:
        if isinstance(value, str):
            return value[:80]
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str):
                return first[:80]
            if isinstance(first, dict):
                return str(first)[:80]
        if isinstance(value, dict):
            return str(value)[:80]
        return ""

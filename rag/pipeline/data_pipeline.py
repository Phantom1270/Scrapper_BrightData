"""
Data pipeline orchestrator — Phase 4.2.

Ties together: schema discovery → normalization → cleaning →
deduplication → storage → report.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from rag.config.settings import get_settings
from rag.models.document import NormalizedDocument
from rag.pipeline.cleaner import ContentCleaner
from rag.pipeline.deduplicator import DocumentDeduplicator
from rag.pipeline.normalizer import UniversalNormalizer
from rag.pipeline.schema_discovery import SchemaDiscovery
from rag.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Full Phase 4.2 data pipeline.

    Steps:
    1. Schema discovery (profiling & reporting)
    2. Normalization  (raw JSON → NormalizedDocuments)
    3. Cleaning       (per-block type-aware cleaning)
    4. Deduplication  (exact + near-duplicate)
    5. Storage        (via Phase 4.1 storage backend)
    6. Report         (summary dict)

    Usage::

        pipeline = DataPipeline()
        report = pipeline.run("path/to/phase3_output.json")
        print(report)
    """

    def __init__(self, settings=None, store=None) -> None:
        self._settings = settings or get_settings()
        self._normalizer = UniversalNormalizer(self._settings)
        self._cleaner = ContentCleaner(self._settings)
        self._deduplicator = DocumentDeduplicator()
        self._last_report: dict = {}

        # Storage backend — accept injected store (useful in tests)
        if store is not None:
            self._store = store
        else:
            db_path = str(
                Path(self._settings.general.data_dir) / "rag.db"
            )
            self._store = SQLiteStore(db_path=db_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, scraped_json_path: Optional[str] = None) -> dict:
        """
        Execute the full pipeline on a single scraped JSON file.

        If *scraped_json_path* is None, scans settings.scraper_output.directory
        for *.json files and processes the first one found.
        """
        path = self._resolve_input(scraped_json_path)
        logger.info("DataPipeline.run: processing %s", path)

        # ---- Step 1: Schema discovery ----
        discovery = SchemaDiscovery(str(path))
        schema_report = discovery.discover()
        discovery.print_report()
        domain = schema_report.get("domain", "")
        raw_total = schema_report.get("total_processed", 0)

        # ---- Step 2: Normalize ----
        raw_docs = self._normalizer.normalize_file(str(path))
        norm_stats = self._normalizer.get_stats()
        by_template_in = norm_stats.get("by_template", {})

        # ---- Step 3: Clean ----
        cleaned_docs: List[NormalizedDocument] = []
        empties_after_clean = 0
        for doc in raw_docs:
            if doc.error:
                # Pass error docs through uncleaned
                cleaned_docs.append(doc)
                continue
            cd = self._cleaner.clean_document(doc)
            if cd.content_blocks or cd.error:
                cleaned_docs.append(cd)
            else:
                empties_after_clean += 1

        # ---- Step 4: Deduplicate ----
        # Only dedup non-error docs (error docs are unique by URL)
        live_docs = [d for d in cleaned_docs if not d.error]
        error_docs = [d for d in cleaned_docs if d.error]

        deduped = self._deduplicator.deduplicate(live_docs)
        dedup_stats = self._deduplicator.get_stats()

        final_docs = deduped + error_docs  # Keep error docs for traceability

        # ---- Step 5: Store ----
        self._store.save_documents(final_docs)

        # ---- Step 6: Report ----
        by_content_type: Dict[str, int] = defaultdict(int)
        for doc in deduped:
            by_content_type[doc.content_type] += 1

        by_template_out: Dict[str, dict] = {}
        for tpl_id, tdata in by_template_in.items():
            by_template_out[tpl_id] = {
                "input":  tdata.get("input", 0),
                "output": tdata.get("output", 0),
                "errors": tdata.get("errors", 0),
            }

        report = {
            "input_file":                str(path),
            "domain":                    domain,
            "total_raw_entries":         norm_stats.get("total_entries", 0),
            "total_normalized":          len(raw_docs),
            "total_after_cleaning":      len(cleaned_docs),
            "total_after_dedup":         len(deduped),
            "errors_skipped":            norm_stats.get("total_errors", 0),
            "empties_skipped":           norm_stats.get("total_empties", 0) + empties_after_clean,
            "exact_duplicates_removed":  dedup_stats.get("exact_removed", 0),
            "near_duplicates_removed":   dedup_stats.get("near_removed", 0),
            "by_template":               by_template_out,
            "by_content_type":           dict(by_content_type),
            "storage_backend":           type(self._store).__name__,
            "documents_stored":          len(final_docs),
        }
        self._last_report = report
        return report

    def run_all(self) -> dict:
        """
        Scan scraper_output.directory for all *.json files and run
        the pipeline on each one. Return a combined report.
        """
        scan_dir = Path(self._settings.scraper_output.directory)
        json_files = sorted(scan_dir.glob("*.json")) if scan_dir.exists() else []

        if not json_files:
            logger.warning("run_all: no *.json files found in %s", scan_dir)
            combined = self._empty_report(str(scan_dir))
            self._last_report = combined
            return combined

        combined: dict = {}
        for fpath in json_files:
            report = self.run(str(fpath))
            if not combined:
                combined = dict(report)
                combined["input_file"] = str(scan_dir)
                combined["processed_files"] = [report["input_file"]]
            else:
                combined["processed_files"].append(report["input_file"])
                for key in (
                    "total_raw_entries", "total_normalized",
                    "total_after_cleaning", "total_after_dedup",
                    "errors_skipped", "empties_skipped",
                    "exact_duplicates_removed", "near_duplicates_removed",
                    "documents_stored",
                ):
                    combined[key] = combined.get(key, 0) + report.get(key, 0)
                for tpl_id, tdata in report.get("by_template", {}).items():
                    if tpl_id in combined["by_template"]:
                        for k in ("input", "output", "errors"):
                            combined["by_template"][tpl_id][k] += tdata.get(k, 0)
                    else:
                        combined["by_template"][tpl_id] = dict(tdata)
                for ct, cnt in report.get("by_content_type", {}).items():
                    combined["by_content_type"][ct] = combined["by_content_type"].get(ct, 0) + cnt

        self._last_report = combined
        return combined

    def get_processing_report(self) -> dict:
        """Return the report from the last run() or run_all() call."""
        return dict(self._last_report)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_input(self, path_arg: Optional[str]) -> Path:
        if path_arg:
            p = Path(path_arg)
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {p}")
            return p

        scan_dir = Path(self._settings.scraper_output.directory)
        if scan_dir.exists():
            files = sorted(scan_dir.glob("*.json"))
            if files:
                return files[0]
        raise FileNotFoundError(
            f"No scraped JSON file provided and none found in {scan_dir}"
        )

    @staticmethod
    def _empty_report(scan_dir: str) -> dict:
        return {
            "input_file": scan_dir,
            "domain": "",
            "total_raw_entries": 0,
            "total_normalized": 0,
            "total_after_cleaning": 0,
            "total_after_dedup": 0,
            "errors_skipped": 0,
            "empties_skipped": 0,
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "by_template": {},
            "by_content_type": {},
            "storage_backend": "SQLiteStore",
            "documents_stored": 0,
        }

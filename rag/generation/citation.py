"""
Extracts and formats citations from LLM responses.
"""

import re
import logging
from typing import List, Dict
from difflib import SequenceMatcher

from rag.models.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class CitationExtractor:
    """Extracts and matches [Source: ...] citations from LLM responses."""

    def __init__(self):
        pass

    def extract_citations(self, answer: str, results: List[RetrievalResult]) -> List[Dict]:
        """
        Extract citation references from the generated answer and match them to chunks.
        """
        if not answer:
            return []
            
        citations = []
        # Pattern matches [Source: Any Text Here]
        pattern = r'\[Source:\s*(.+?)\]'
        
        matches = re.finditer(pattern, answer)
        for match in matches:
            ref_text = match.group(1).strip()
            citation = self._match_source(ref_text, results)
            citations.append(citation)
            
        return citations

    def _match_source(self, ref_text: str, results: List[RetrievalResult]) -> Dict:
        """
        Match a reference text against the retrieved results.
        Priority: Exact > Partial > URL > Fuzzy
        """
        # Default unmatched
        citation = {
            "reference_text": ref_text,
            "matched": False,
            "chunk_id": None,
            "url": None,
            "heading_path": None,
            "score": None,
        }
        
        if not results:
            return citation

        # Clean strings for matching
        ref_lower = ref_text.lower()
        
        # Pre-compute result strings
        result_strings = []
        for r in results:
            heading_str = " > ".join(r.heading_path) if r.heading_path else ""
            url_str = r.url or ""
            chunk_id_str = f"Chunk {r.chunk_id}"
            
            result_strings.append({
                "result": r,
                "heading_str": heading_str,
                "heading_lower": heading_str.lower(),
                "url_lower": url_str.lower(),
                "chunk_id_lower": chunk_id_str.lower(),
            })

        # 1. Exact match
        for item in result_strings:
            if ref_lower == item["heading_lower"] or ref_lower == item["url_lower"] or ref_lower == item["chunk_id_lower"]:
                return self._build_matched_citation(ref_text, item["result"])

        # 2. Partial match
        for item in result_strings:
            if item["heading_lower"] and (ref_lower in item["heading_lower"] or item["heading_lower"] in ref_lower):
                return self._build_matched_citation(ref_text, item["result"])
            if item["url_lower"] and (ref_lower in item["url_lower"] or item["url_lower"] in ref_lower):
                return self._build_matched_citation(ref_text, item["result"])

        # 3. URL match (already partly covered by partial, but double check)
        for item in result_strings:
            if item["url_lower"] and ref_lower in item["url_lower"]:
                return self._build_matched_citation(ref_text, item["result"])

        # 4. Fuzzy match
        best_ratio = 0
        best_match = None
        
        for item in result_strings:
            # Check against heading elements
            for el in item["result"].heading_path:
                ratio = SequenceMatcher(None, ref_lower, el.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = item["result"]
                    
            # Check against full heading string
            if item["heading_lower"]:
                ratio = SequenceMatcher(None, ref_lower, item["heading_lower"]).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = item["result"]
                    
        if best_ratio > 0.6 and best_match is not None:
            return self._build_matched_citation(ref_text, best_match)
            
        # No match found
        return citation

    def _build_matched_citation(self, ref_text: str, result: RetrievalResult) -> Dict:
        """Helper to build a matched citation dictionary."""
        return {
            "reference_text": ref_text,
            "matched": True,
            "chunk_id": result.chunk_id,
            "url": result.url,
            "heading_path": result.heading_path,
            "score": result.score,
        }

    def format_sources_footer(self, results: List[RetrievalResult]) -> str:
        """Format a "Sources" footer to append to the answer."""
        if not results:
            return ""
            
        unique_sources = []
        seen_urls = set()
        
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                heading_str = " > ".join(r.heading_path) if r.heading_path else r.url
                unique_sources.append(f"- {heading_str} ({r.url})")
                
        if not unique_sources:
            return ""
            
        footer = "\n\n---\nSources:\n"
        footer += "\n".join(unique_sources)
        footer += "\n"
        
        return footer

    def has_citations(self, answer: str) -> bool:
        """Quick check if answer contains [Source: ...] patterns."""
        if not answer:
            return False
        return bool(re.search(r'\[Source:\s*(.+?)\]', answer))

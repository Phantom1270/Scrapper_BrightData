"""
Validates chunk quality and reports issues.
"""

from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import json

from rag.models.chunk import Chunk


@dataclass
class ChunkAuditReport:
    total_chunks: int
    avg_tokens: float
    median_tokens: int
    min_tokens: int
    max_tokens: int
    by_content_type: Dict[str, int]
    by_block_type: Dict[str, int]
    by_document_content_type: Dict[str, int]
    chunks_per_document: Dict[str, int]
    oversized_chunks: List[dict]
    tiny_chunks: List[dict]
    no_heading_chunks: List[dict]
    issues: List[str]


class ChunkQualityAuditor:
    """Validates chunk quality and generates an audit report."""

    def __init__(self, settings=None, config=None):
        if config is not None:
            settings = config
            
        if isinstance(settings, dict):
            self.max_tokens = settings.get("max_tokens", 512)
            self.min_tokens = settings.get("min_tokens", 100)
        else:
            if settings is None:
                from rag.config.settings import get_settings
                settings = get_settings()
            self.max_tokens = settings.chunking.max_tokens
            self.min_tokens = settings.chunking.min_tokens

    def audit(self, chunks: List[Chunk]) -> ChunkAuditReport:
        """Analyze all chunks and produce a quality report."""
        if not chunks:
            return ChunkAuditReport(
                total_chunks=0, avg_tokens=0.0, median_tokens=0,
                min_tokens=0, max_tokens=0, by_content_type={},
                by_block_type={}, by_document_content_type={},
                chunks_per_document={}, oversized_chunks=[],
                tiny_chunks=[], no_heading_chunks=[], issues=["No chunks provided."]
            )

        token_counts = []
        by_content_type = defaultdict(int)
        by_block_type = defaultdict(int)
        by_document_content_type = defaultdict(int)
        chunks_per_document = defaultdict(int)
        
        oversized = []
        tiny = []
        no_heading = []

        for chunk in chunks:
            tc = chunk.token_count
            token_counts.append(tc)
            
            by_content_type[chunk.content_type] += 1
            if chunk.block_type:
                by_block_type[chunk.block_type] += 1
                
            doc_ct = chunk.metadata.get("document_content_type", "unknown")
            by_document_content_type[doc_ct] += 1
            chunks_per_document[chunk.doc_id] += 1

            if tc > self.max_tokens:
                oversized.append({
                    "chunk_id": chunk.chunk_id,
                    "token_count": tc,
                    "reason": "Exceeds max_tokens"
                })
            elif tc < self.min_tokens:
                tiny.append({
                    "chunk_id": chunk.chunk_id,
                    "token_count": tc,
                    "content_preview": chunk.content[:50]
                })

            if not chunk.heading_path:
                no_heading.append({
                    "chunk_id": chunk.chunk_id,
                    "content_preview": chunk.content[:50]
                })

        token_counts.sort()
        n = len(token_counts)
        median = token_counts[n//2] if n % 2 != 0 else (token_counts[n//2 - 1] + token_counts[n//2]) // 2
        avg = sum(token_counts) / n

        issues = []
        if oversized:
            issues.append(f"{len(oversized)} chunks exceed max_tokens (likely oversized code blocks)")
        if tiny:
            issues.append(f"{len(tiny)} chunks are under min_tokens (consider grouping small blocks)")
        if no_heading:
            issues.append(f"{len(no_heading)} chunks have empty heading_path (context-free chunks hurt retrieval)")
        
        issues.append(f"Average chunk size is {avg:.1f} tokens (target range: {self.min_tokens}-{self.max_tokens})")
        
        single_chunk_docs = sum(1 for v in chunks_per_document.values() if v == 1)
        if single_chunk_docs > 0:
            issues.append(f"{single_chunk_docs} documents have only 1 chunk (may indicate parsing failure)")
            
        unknown_ct = by_document_content_type.get("unknown", 0)
        if unknown_ct > 0:
            issues.append(f"{unknown_ct} chunks have content_type='unknown' (classification may have failed)")

        return ChunkAuditReport(
            total_chunks=n,
            avg_tokens=avg,
            median_tokens=median,
            min_tokens=token_counts[0],
            max_tokens=token_counts[-1],
            by_content_type=dict(by_content_type),
            by_block_type=dict(by_block_type),
            by_document_content_type=dict(by_document_content_type),
            chunks_per_document=dict(chunks_per_document),
            oversized_chunks=oversized,
            tiny_chunks=tiny,
            no_heading_chunks=no_heading,
            issues=issues
        )

    def print_report(self, report: ChunkAuditReport) -> None:
        """Pretty-print the audit report to stdout."""
        print("=== Chunk Quality Audit Report ===")
        print(f"Total Chunks: {report.total_chunks}")
        if report.total_chunks == 0:
            return
            
        print(f"Tokens: Min={report.min_tokens}, Max={report.max_tokens}, Avg={report.avg_tokens:.1f}, Median={report.median_tokens}")
        print("\n--- By Content Type ---")
        for k, v in report.by_content_type.items():
            print(f"  {k}: {v}")
            
        print("\n--- By Document Content Type ---")
        for k, v in report.by_document_content_type.items():
            print(f"  {k}: {v}")
            
        if report.issues:
            print("\n--- Issues Detected ---")
            for issue in report.issues:
                print(f"  * {issue}")
        print("==================================")

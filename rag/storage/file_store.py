"""
JSONL file-based storage backend.

Simpler alternative to SQLite — each document/chunk is a JSON line.
Suitable for prototyping and small datasets (<100K records).

Layout:
    {data_dir}/documents.jsonl
    {data_dir}/chunks.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.storage.base import BaseStore

logger = logging.getLogger(__name__)


class FileStore(BaseStore):
    """
    JSONL-backed store.

    All documents and chunks are held in memory after the first read
    and flushed to disk on every write. This is intentionally simple —
    use SQLiteStore for anything production-grade.

    Args:
        data_dir: Directory where documents.jsonl and chunks.jsonl are stored.
    """

    def __init__(self, settings=None, data_dir: str = None) -> None:
        if isinstance(settings, str):
            data_dir = settings
            settings = None
            
        if data_dir is None:
            data_dir = settings.general.data_dir if settings else "./rag/data"
            
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._docs_path = self.data_dir / "documents.jsonl"
        self._chunks_path = self.data_dir / "chunks.jsonl"

        # In-memory indexes
        self._docs: Dict[str, NormalizedDocument] = {}
        self._chunks: Dict[str, Chunk] = {}

        self._load_all()

    # ------------------------------------------------------------------
    # Internal: load / flush
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load both JSONL files into memory."""
        self._docs = {}
        if self._docs_path.exists():
            for line in self._docs_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        doc = NormalizedDocument.from_dict(json.loads(line))
                        self._docs[doc.doc_id] = doc
                    except Exception as exc:
                        logger.warning("Skipping corrupt document line: %s", exc)

        self._chunks = {}
        if self._chunks_path.exists():
            for line in self._chunks_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        chunk = Chunk.from_dict(json.loads(line))
                        self._chunks[chunk.chunk_id] = chunk
                    except Exception as exc:
                        logger.warning("Skipping corrupt chunk line: %s", exc)

    def _flush_docs(self) -> None:
        """Write all in-memory documents to disk."""
        lines = [
            json.dumps(doc.to_dict(), ensure_ascii=False)
            for doc in self._docs.values()
        ]
        self._docs_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                                   encoding="utf-8")

    def _flush_chunks(self) -> None:
        """Write all in-memory chunks to disk."""
        lines = [
            json.dumps(chunk.to_dict(), ensure_ascii=False)
            for chunk in self._chunks.values()
        ]
        self._chunks_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                                     encoding="utf-8")

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def save_documents(self, documents: List[NormalizedDocument]) -> None:
        for doc in documents:
            self._docs[doc.doc_id] = doc
        self._flush_docs()
        logger.debug("FileStore: saved %d documents.", len(documents))

    def get_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        return self._docs.get(doc_id)

    def get_all_documents(self) -> List[NormalizedDocument]:
        return list(self._docs.values())

    def get_documents_by_template(self, template_id: str) -> List[NormalizedDocument]:
        return [d for d in self._docs.values() if d.template_id == template_id]

    def count_documents(self) -> int:
        return len(self._docs)

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._flush_chunks()
        logger.debug("FileStore: saved %d chunks.", len(chunks))

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        return sorted(
            [c for c in self._chunks.values() if c.doc_id == doc_id],
            key=lambda c: c.chunk_index,
        )

    def get_all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())

    def count_chunks(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_all(self) -> None:
        self._docs = {}
        self._chunks = {}
        self._flush_docs()
        self._flush_chunks()
        logger.info("FileStore: all data deleted.")

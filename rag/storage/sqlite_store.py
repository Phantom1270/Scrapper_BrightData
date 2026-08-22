"""
SQLite storage backend.

Stores documents and chunks in a local SQLite database.
Tables are created automatically on first use.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

from rag.models.chunk import Chunk
from rag.models.document import NormalizedDocument
from rag.storage.base import BaseStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    url                 TEXT NOT NULL,
    title               TEXT NOT NULL DEFAULT '',
    description         TEXT NOT NULL DEFAULT '',
    content_blocks_json TEXT NOT NULL DEFAULT '[]',
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    template_id         TEXT NOT NULL DEFAULT '',
    content_type        TEXT NOT NULL DEFAULT 'unknown',
    source_link         TEXT,
    error               TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_url          ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_template_id  ON documents(template_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL,
    url               TEXT NOT NULL DEFAULT '',
    content           TEXT NOT NULL DEFAULT '',
    content_type      TEXT NOT NULL DEFAULT 'prose',
    heading_path_json TEXT NOT NULL DEFAULT '[]',
    chunk_index       INTEGER NOT NULL DEFAULT 0,
    token_count       INTEGER NOT NULL DEFAULT 0,
    parent_chunk_id   TEXT,
    block_type        TEXT NOT NULL DEFAULT '',
    language          TEXT NOT NULL DEFAULT '',
    metadata_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id       ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_type ON chunks(content_type);
"""

_UPSERT_DOCUMENT = """
INSERT INTO documents (
    doc_id, url, title, description,
    content_blocks_json, metadata_json,
    template_id, content_type, source_link, error
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(doc_id) DO UPDATE SET
    url                 = excluded.url,
    title               = excluded.title,
    description         = excluded.description,
    content_blocks_json = excluded.content_blocks_json,
    metadata_json       = excluded.metadata_json,
    template_id         = excluded.template_id,
    content_type        = excluded.content_type,
    source_link         = excluded.source_link,
    error               = excluded.error;
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, doc_id, url, content, content_type,
    heading_path_json, chunk_index, token_count,
    parent_chunk_id, block_type, language, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(chunk_id) DO UPDATE SET
    doc_id            = excluded.doc_id,
    url               = excluded.url,
    content           = excluded.content,
    content_type      = excluded.content_type,
    heading_path_json = excluded.heading_path_json,
    chunk_index       = excluded.chunk_index,
    token_count       = excluded.token_count,
    parent_chunk_id   = excluded.parent_chunk_id,
    block_type        = excluded.block_type,
    language          = excluded.language,
    metadata_json     = excluded.metadata_json;
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_to_row(doc: NormalizedDocument) -> tuple:
    return (
        doc.doc_id,
        doc.url,
        doc.title,
        doc.description,
        json.dumps([b.to_dict() for b in doc.content_blocks], ensure_ascii=False),
        json.dumps(doc.metadata, ensure_ascii=False),
        doc.template_id,
        doc.content_type,
        doc.source_link,
        doc.error,
    )


def _row_to_doc(row: sqlite3.Row) -> NormalizedDocument:
    from rag.models.document import ContentBlock

    raw_blocks = json.loads(row["content_blocks_json"] or "[]")
    blocks = [ContentBlock.from_dict(b) for b in raw_blocks]
    return NormalizedDocument(
        doc_id=row["doc_id"],
        url=row["url"],
        title=row["title"],
        description=row["description"],
        content_blocks=blocks,
        metadata=json.loads(row["metadata_json"] or "{}"),
        template_id=row["template_id"],
        content_type=row["content_type"],
        source_link=row["source_link"],
        error=row["error"],
    )


def _chunk_to_row(chunk: Chunk) -> tuple:
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.url,
        chunk.content,
        chunk.content_type,
        json.dumps(chunk.heading_path, ensure_ascii=False),
        chunk.chunk_index,
        chunk.token_count,
        chunk.parent_chunk_id,
        chunk.block_type,
        chunk.language,
        json.dumps(chunk.metadata, ensure_ascii=False),
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        doc_id=row["doc_id"],
        url=row["url"],
        content=row["content"],
        content_type=row["content_type"],
        heading_path=json.loads(row["heading_path_json"] or "[]"),
        chunk_index=row["chunk_index"],
        token_count=row["token_count"],
        parent_chunk_id=row["parent_chunk_id"],
        block_type=row["block_type"],
        language=row["language"],
        metadata=json.loads(row["metadata_json"] or "{}"),
    )


# ---------------------------------------------------------------------------
# SQLiteStore
# ---------------------------------------------------------------------------


class SQLiteStore(BaseStore):
    """
    SQLite-backed store for NormalizedDocuments and Chunks.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to ./rag/data/rag.db.
                 Pass ":memory:" for an in-memory database (useful in tests).
    """

    def __init__(self, settings=None, db_path: str = None) -> None:
        if isinstance(settings, str):
            db_path = settings
            settings = None
            
        if db_path is None:
            import os
            db_path = os.path.join(settings.general.data_dir, "rag.db") if settings else "./rag/data/rag.db"
            
        self.db_path = db_path
        # :memory: databases don't survive across separate connections,
        # so we keep one persistent connection for the lifetime of this object.
        self._persistent_conn: sqlite3.Connection | None = None
        if db_path == ":memory:":
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
            self._persistent_conn.execute("PRAGMA foreign_keys=ON;")
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield a connection with row_factory set."""
        if self._persistent_conn is not None:
            # In-memory mode: reuse the single persistent connection.
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
            return

        # File-based mode: open a fresh connection per operation.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._connect() as conn:
            conn.executescript(_DDL_DOCUMENTS)
            conn.executescript(_DDL_CHUNKS)
        logger.debug("SQLiteStore initialized at %s", self.db_path)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    def save_documents(self, documents: List[NormalizedDocument]) -> None:
        if not documents:
            return
        rows = [_doc_to_row(d) for d in documents]
        with self._connect() as conn:
            conn.executemany(_UPSERT_DOCUMENT, rows)
        logger.debug("Saved %d documents.", len(documents))

    def get_document(self, doc_id: str) -> Optional[NormalizedDocument]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _row_to_doc(row) if row else None

    def get_all_documents(self) -> List[NormalizedDocument]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM documents").fetchall()
        return [_row_to_doc(r) for r in rows]

    def get_documents_by_template(self, template_id: str) -> List[NormalizedDocument]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE template_id = ?", (template_id,)
            ).fetchall()
        return [_row_to_doc(r) for r in rows]

    def count_documents(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def save_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        rows = [_chunk_to_row(c) for c in chunks]
        with self._connect() as conn:
            conn.executemany(_UPSERT_CHUNK, rows)
        logger.debug("Saved %d chunks.", len(chunks))

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks_by_doc(self, doc_id: str) -> List[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
                (doc_id,),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def get_all_chunks(self) -> List[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY doc_id, chunk_index"
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def count_chunks(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_all(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
        logger.info("SQLiteStore: all data deleted.")

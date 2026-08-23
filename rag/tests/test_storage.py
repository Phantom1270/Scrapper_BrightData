"""
Tests for both storage backends: SQLiteStore and FileStore.

Each test class runs the same test suite against both backends
via shared helper methods.
"""

from __future__ import annotations

import time
from typing import List

import pytest

from rag.models.chunk import Chunk
from rag.models.document import ContentBlock, NormalizedDocument
from rag.storage.file_store import FileStore
from rag.storage.sqlite_store import SQLiteStore
from rag.utils.ids import generate_chunk_id, generate_doc_id


# ---------------------------------------------------------------------------
# Shared builder helpers
# ---------------------------------------------------------------------------


def make_doc(n: int = 0, template_id: str = "tpl_001") -> NormalizedDocument:
    url = f"https://example.com/doc-{n}"
    return NormalizedDocument(
        doc_id=generate_doc_id(url),
        url=url,
        title=f"Document {n}",
        description=f"Description for doc {n}",
        content_blocks=[
            ContentBlock(block_type="prose", text=f"Content block for doc {n}"),
        ],
        metadata={"index": n},
        template_id=template_id,
        content_type="api_reference",
    )


def make_chunk(doc_id: str, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=generate_chunk_id(doc_id, index),
        doc_id=doc_id,
        url=f"https://example.com/doc",
        content=f"Chunk {index} content.",
        content_type="prose",
        heading_path=["Section", f"Part {index}"],
        chunk_index=index,
        token_count=5,
    )


# ---------------------------------------------------------------------------
# Shared test logic (applied to both backends)
# ---------------------------------------------------------------------------


def _test_save_and_get_document(store):
    doc = make_doc(0)
    store.save_documents([doc])
    retrieved = store.get_document(doc.doc_id)
    assert retrieved is not None
    assert retrieved.doc_id == doc.doc_id
    assert retrieved.title == doc.title
    assert retrieved.template_id == doc.template_id
    assert len(retrieved.content_blocks) == 1


def _test_save_and_get_multiple_documents(store):
    docs = [make_doc(i) for i in range(5)]
    store.save_documents(docs)
    all_docs = store.get_all_documents()
    assert len(all_docs) == 5


def _test_get_document_not_found(store):
    result = store.get_document("nonexistent_id_abc123")
    assert result is None


def _test_get_documents_by_template(store):
    docs_a = [make_doc(i, template_id="tpl_001") for i in range(3)]
    docs_b = [make_doc(i + 10, template_id="tpl_002") for i in range(2)]
    store.save_documents(docs_a + docs_b)

    by_tpl1 = store.get_documents_by_template("tpl_001")
    by_tpl2 = store.get_documents_by_template("tpl_002")
    by_tpl3 = store.get_documents_by_template("tpl_999")

    assert len(by_tpl1) == 3
    assert len(by_tpl2) == 2
    assert len(by_tpl3) == 0


def _test_save_and_get_chunks(store):
    doc = make_doc(0)
    store.save_documents([doc])

    chunks = [make_chunk(doc.doc_id, i) for i in range(3)]
    store.save_chunks(chunks)

    retrieved = store.get_chunks_by_doc(doc.doc_id)
    assert len(retrieved) == 3
    # Ordered by chunk_index
    assert retrieved[0].chunk_index == 0
    assert retrieved[1].chunk_index == 1
    assert retrieved[2].chunk_index == 2


def _test_count_documents(store):
    assert store.count_documents() == 0
    store.save_documents([make_doc(0), make_doc(1)])
    assert store.count_documents() == 2


def _test_count_chunks(store):
    assert store.count_chunks() == 0
    doc = make_doc(0)
    store.save_documents([doc])
    store.save_chunks([make_chunk(doc.doc_id, 0), make_chunk(doc.doc_id, 1)])
    assert store.count_chunks() == 2


def _test_delete_all(store):
    store.save_documents([make_doc(0), make_doc(1)])
    doc = make_doc(0)
    store.save_chunks([make_chunk(doc.doc_id, 0)])
    store.delete_all()
    assert store.count_documents() == 0
    assert store.count_chunks() == 0


def _test_batch_insert_performance(store):
    """Save 1000 documents. Should complete in under 5 seconds."""
    docs = [make_doc(i) for i in range(1000)]
    start = time.monotonic()
    store.save_documents(docs)
    elapsed = time.monotonic() - start
    assert store.count_documents() == 1000
    assert elapsed < 5.0, f"Batch insert took {elapsed:.2f}s (limit: 5s)"


def _test_get_chunk_by_id(store):
    doc = make_doc(42)
    store.save_documents([doc])
    chunk = make_chunk(doc.doc_id, 0)
    store.save_chunks([chunk])
    retrieved = store.get_chunk(chunk.chunk_id)
    assert retrieved is not None
    assert retrieved.chunk_id == chunk.chunk_id
    assert retrieved.content == chunk.content


def _test_get_chunk_not_found(store):
    result = store.get_chunk("nonexistent_chunk_id")
    assert result is None


def _test_get_all_chunks(store):
    doc_a = make_doc(0)
    doc_b = make_doc(1)
    store.save_documents([doc_a, doc_b])
    store.save_chunks([make_chunk(doc_a.doc_id, 0), make_chunk(doc_b.doc_id, 0)])
    assert store.count_chunks() == 2
    assert len(store.get_all_chunks()) == 2


def _test_upsert_document(store):
    """Saving the same doc_id twice should update, not duplicate."""
    doc = make_doc(0)
    store.save_documents([doc])
    updated = NormalizedDocument(
        doc_id=doc.doc_id,
        url=doc.url,
        title="Updated Title",
        description="Updated description",
        content_blocks=[],
        metadata={},
        template_id="tpl_999",
        content_type="unknown",
    )
    store.save_documents([updated])
    assert store.count_documents() == 1
    retrieved = store.get_document(doc.doc_id)
    assert retrieved.title == "Updated Title"


# ---------------------------------------------------------------------------
# SQLiteStore tests
# ---------------------------------------------------------------------------


class TestSQLiteStore:
    def test_save_and_get_document(self, tmp_store):
        _test_save_and_get_document(tmp_store)

    def test_save_and_get_multiple_documents(self, tmp_store):
        _test_save_and_get_multiple_documents(tmp_store)

    def test_get_document_not_found(self, tmp_store):
        _test_get_document_not_found(tmp_store)

    def test_get_documents_by_template(self, tmp_store):
        _test_get_documents_by_template(tmp_store)

    def test_save_and_get_chunks(self, tmp_store):
        _test_save_and_get_chunks(tmp_store)

    def test_count_documents(self, tmp_store):
        _test_count_documents(tmp_store)

    def test_count_chunks(self, tmp_store):
        _test_count_chunks(tmp_store)

    def test_delete_all(self, tmp_store):
        _test_delete_all(tmp_store)

    def test_batch_insert_performance(self, tmp_store):
        _test_batch_insert_performance(tmp_store)

    def test_get_chunk_by_id(self, tmp_store):
        _test_get_chunk_by_id(tmp_store)

    def test_get_chunk_not_found(self, tmp_store):
        _test_get_chunk_not_found(tmp_store)

    def test_get_all_chunks(self, tmp_store):
        _test_get_all_chunks(tmp_store)

    def test_upsert_document(self, tmp_store):
        _test_upsert_document(tmp_store)

    def test_in_memory_mode(self):
        """SQLiteStore(':memory:') should work without creating any files."""
        store = SQLiteStore(db_path=":memory:")
        store.save_documents([make_doc(0)])
        assert store.count_documents() == 1


# ---------------------------------------------------------------------------
# FileStore tests
# ---------------------------------------------------------------------------


@pytest.fixture
def file_store(tmp_path):
    return FileStore(data_dir=str(tmp_path / "file_store"))


class TestFileStore:
    def test_save_and_get_document(self, file_store):
        _test_save_and_get_document(file_store)

    def test_save_and_get_multiple_documents(self, file_store):
        _test_save_and_get_multiple_documents(file_store)

    def test_get_document_not_found(self, file_store):
        _test_get_document_not_found(file_store)

    def test_get_documents_by_template(self, file_store):
        _test_get_documents_by_template(file_store)

    def test_save_and_get_chunks(self, file_store):
        _test_save_and_get_chunks(file_store)

    def test_count_documents(self, file_store):
        _test_count_documents(file_store)

    def test_count_chunks(self, file_store):
        _test_count_chunks(file_store)

    def test_delete_all(self, file_store):
        _test_delete_all(file_store)

    def test_batch_insert_performance(self, file_store):
        _test_batch_insert_performance(file_store)

    def test_get_chunk_by_id(self, file_store):
        _test_get_chunk_by_id(file_store)

    def test_get_chunk_not_found(self, file_store):
        _test_get_chunk_not_found(file_store)

    def test_get_all_chunks(self, file_store):
        _test_get_all_chunks(file_store)

    def test_upsert_document(self, file_store):
        _test_upsert_document(file_store)

    def test_persistence_across_reload(self, tmp_path):
        """Data saved to JSONL should survive a FileStore reload."""
        data_dir = str(tmp_path / "persistent")
        store1 = FileStore(data_dir=data_dir)
        store1.save_documents([make_doc(0), make_doc(1)])
        store1.save_chunks([make_chunk(make_doc(0).doc_id, 0)])

        # New instance loading from same directory
        store2 = FileStore(data_dir=data_dir)
        assert store2.count_documents() == 2
        assert store2.count_chunks() == 1

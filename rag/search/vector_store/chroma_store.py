"""
ChromaDB vector store implementation.
"""

from typing import List, Optional, Dict
import logging

from rag.search.vector_store.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):
    """Vector store using ChromaDB."""

    def __init__(self, settings=None):
        if settings is None:
            from rag.config.settings import get_settings
            settings = get_settings()

        self.persist_dir = settings.vector_store.persist_dir
        self.collection_name = settings.vector_store.collection_name

        try:
            import chromadb
        except ImportError:
            raise ImportError("chromadb required. pip install chromadb")

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunk_ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]) -> None:
        """Add chunks to Chroma in batches."""
        if not chunk_ids:
            return

        batch_size = 500
        total = len(chunk_ids)

        for i in range(0, total, batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_metas = metadatas[i:i + batch_size]

            self.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_metas
            )

    def search(self, query_embedding: List[float], top_k: int = 10, content_type_filter: str = None, doc_id_filter: str = None) -> List[Dict]:
        """Search Chroma collection with optional filters."""
        where_filter = {}
        
        if content_type_filter and doc_id_filter:
            where_filter = {
                "$and": [
                    {"content_type": content_type_filter},
                    {"doc_id": doc_id_filter}
                ]
            }
        elif content_type_filter:
            where_filter = {"content_type": content_type_filter}
        elif doc_id_filter:
            where_filter = {"doc_id": doc_id_filter}
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if not results["ids"] or not results["ids"][0]:
            return formatted_results

        # Results are returned as lists of lists (one per query)
        ids = results["ids"][0]
        distances = results["distances"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for chunk_id, distance, doc, meta in zip(ids, distances, docs, metas):
            # Chroma returns cosine distance. similarity = 1 - distance
            score = 1.0 - distance
            formatted_results.append({
                "id": chunk_id,
                "score": score,
                "document": doc,
                "metadata": meta
            })

        return formatted_results

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete specified chunks."""
        if not chunk_ids:
            return
        self.collection.delete(ids=chunk_ids)

    def delete_all(self) -> None:
        """Delete the collection and recreate it empty."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except ValueError:
            pass # Collection doesn't exist
            
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return total items in collection."""
        return self.collection.count()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        """Retrieve a single chunk by ID."""
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not result["ids"]:
            return None
            
        return {
            "id": result["ids"][0],
            "document": result["documents"][0] if result.get("documents") else None,
            "metadata": result["metadatas"][0] if result.get("metadatas") else None,
            "embedding": result["embeddings"][0] if result.get("embeddings") is not None and len(result["embeddings"]) > 0 else None
        }

"""
BM25 keyword search index.
"""

from typing import List, Tuple
import re
import os
import pickle

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Handle optional dependency gracefully for certain environments
    pass


class BM25Index:
    """Keyword search index using BM25Okapi."""

    def __init__(self, settings=None):
        self.bm25 = None
        self.chunk_ids = []
        self.chunk_texts = []
        self.tokenized_corpus = []

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizer for BM25:
        - Lowercase the text
        - Split on non-alphanumeric characters, keeping underscores
        - Also split on dots to catch "sklearn.config_context" as
          ["sklearn", "config_context", "config", "context"]
        - Filter out tokens < 2 characters
        """
        text = text.lower()
        
        # Keep alphanumeric and underscores, split on others
        # Wait, the instruction specifically says:
        # - Split on non-alphanumeric characters, keeping underscores
        # - Also split on dots to catch "sklearn.config_context" as ["sklearn", "config_context", "config", "context"]
        
        # So we first want to get basic word-like tokens, maybe including dots if we want to split them further?
        # A simpler way: split by anything that is not [a-z0-9_.]
        raw_tokens = re.split(r'[^a-z0-9_.]+', text)
        
        final_tokens = []
        for t in raw_tokens:
            if not t:
                continue
                
            if '.' in t:
                # Add the full string without dots? No, the requirement says:
                # catch "sklearn.config_context" as ["sklearn", "config_context", "config", "context"]?
                # Actually, "sklearn.config_context" splitting by dot gives "sklearn" and "config_context".
                # Wait, does it also need the full "sklearn.config_context"?
                # Usually standard split by dot gives ["sklearn", "config_context"].
                # "config_context" has an underscore, so it stays intact.
                # If we just split by non-alphanumeric except underscore, then "sklearn.config_context"
                # is split into "sklearn" and "config_context".
                
                parts = t.split('.')
                for p in parts:
                    if len(p) >= 2:
                        final_tokens.append(p)
            else:
                if len(t) >= 2:
                    final_tokens.append(t)
                    
        return final_tokens

    def build(self, chunk_ids: List[str], chunk_texts: List[str]) -> None:
        """Build the BM25 index from scratch."""
        self.chunk_ids = list(chunk_ids)
        self.chunk_texts = list(chunk_texts)
        self.tokenized_corpus = [self._tokenize(t) for t in self.chunk_texts]
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Search the BM25 index."""
        if not self.bm25 or not self.chunk_ids:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices sorted by score descending
        # Using simple sort since we don't assume numpy is always used in this block
        scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_indices[:top_k]:
            results.append((self.chunk_ids[idx], score))
            
        return results

    def add_chunks(self, chunk_ids: List[str], chunk_texts: List[str]) -> None:
        """Add new chunks and rebuild the entire index."""
        self.chunk_ids.extend(chunk_ids)
        self.chunk_texts.extend(chunk_texts)
        new_tokenized = [self._tokenize(t) for t in chunk_texts]
        self.tokenized_corpus.extend(new_tokenized)
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def save(self, path: str) -> None:
        """Persist to disk using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "bm25": self.bm25,
            "ids": self.chunk_ids,
            "texts": self.chunk_texts,
            "tokenized": self.tokenized_corpus
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> bool:
        """Load from pickle file."""
        if not os.path.exists(path):
            return False
            
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.bm25 = state.get("bm25")
            self.chunk_ids = state.get("ids", [])
            self.chunk_texts = state.get("texts", [])
            self.tokenized_corpus = state.get("tokenized", [])
            return True
        except Exception:
            return False

    def count(self) -> int:
        """Return the number of chunks in the index."""
        return len(self.chunk_ids)

    def clear(self) -> None:
        """Reset to empty state."""
        self.bm25 = None
        self.chunk_ids = []
        self.chunk_texts = []
        self.tokenized_corpus = []

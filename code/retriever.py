"""
retriever.py — TF-IDF retrieval system for the Multi-Domain Support Triage Agent.

Loads all .txt files from the corpus directory, splits them into overlapping
chunks, and provides a ranked similarity search using scikit-learn's
TfidfVectorizer.

Usage:
    retriever = Retriever("corpus/")
    retriever.build_index()
    results = retriever.retrieve("my issue text", domain="hackerrank", top_k=3)
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 80  # skip chunks shorter than this


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """TF-IDF-based document retriever over the support corpus."""

    def __init__(self, corpus_dir: str = "../data") -> None:
        self.corpus_dir = Path(corpus_dir)
        self._vectorizer: Optional[SentenceTransformer] = None
        self._matrix = None  # sparse TF-IDF matrix
        self._chunks: List[Dict[str, Any]] = []  # list of chunk metadata dicts

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Load all .txt files from the corpus directory, split them into
        overlapping chunks, and fit a TF-IDF vectorizer.

        Raises:
            RuntimeError: If no documents are found in corpus_dir.
        """
        if not self.corpus_dir.exists():
            raise RuntimeError(
                f"Corpus directory '{self.corpus_dir}' does not exist. "
                "Run --crawl first."
            )

        txt_files = list(self.corpus_dir.rglob("*.txt")) + list(self.corpus_dir.rglob("*.html"))
        if not txt_files:
            raise RuntimeError(
                f"No .txt or .html files found in '{self.corpus_dir}'. "
                "Run --crawl first to populate the corpus."
            )

        logger.info("[retriever] Found %d files. Building chunks...", len(txt_files))

        self._chunks = []
        for filepath in txt_files:
            try:
                text = filepath.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = filepath.read_text(encoding="latin-1")

            # Determine domain from directory name
            try:
                domain = filepath.relative_to(self.corpus_dir).parts[0]
            except ValueError:
                domain = "unknown"

            chunks = self._split_into_chunks(text)
            for chunk_text in chunks:
                self._chunks.append(
                    {
                        "source": str(filepath),
                        "domain": domain,
                        "content": chunk_text,
                    }
                )

        if not self._chunks:
            raise RuntimeError("All corpus files were empty or too short to index.")

        logger.info("[retriever] Total chunks: %d. Encoding with sentence-transformers...", len(self._chunks))

        chunk_texts = [c["content"] for c in self._chunks]
        self._vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
        self._matrix = self._vectorizer.encode(chunk_texts, show_progress_bar=True)

        logger.info("[retriever] Index built successfully.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant corpus chunks for a given query.

        Args:
            query:  The support ticket text or query string.
            domain: If specified (e.g. "hackerrank"), limit search to that domain.
            top_k:  Number of results to return.

        Returns:
            List of dicts with keys: source, content, score, domain.
        """
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        if not query or not query.strip():
            return []

        # Determine which chunk indices to search
        if domain:
            domain_lower = domain.lower()
            candidate_indices = [
                i for i, c in enumerate(self._chunks)
                if c["domain"].lower() == domain_lower
            ]
        else:
            candidate_indices = list(range(len(self._chunks)))

        if not candidate_indices:
            logger.warning(
                "[retriever] No chunks found for domain='%s'. "
                "Falling back to all domains.",
                domain,
            )
            candidate_indices = list(range(len(self._chunks)))

        # Vectorize query
        query_vec = self._vectorizer.encode([query])

        # Compute cosine similarity against candidate chunks only
        candidate_matrix = self._matrix[candidate_indices]
        scores = cosine_similarity(query_vec, candidate_matrix)[0]

        # Rank by score
        top_n = min(top_k, len(candidate_indices))
        ranked = np.argsort(scores)[::-1][:top_n]

        results = []
        for rank_idx in ranked:
            chunk_idx = candidate_indices[rank_idx]
            score = float(scores[rank_idx])
            if score < 1e-6:
                continue  # skip completely irrelevant results
            chunk = self._chunks[chunk_idx]
            results.append(
                {
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": round(score, 4),
                    "domain": chunk["domain"],
                }
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_chunks(text: str) -> List[str]:
        """
        Split text into overlapping chunks of ~CHUNK_SIZE characters.
        Tries to break on whitespace boundaries.
        """
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)

            # If not at the end, try to break on a space
            if end < text_len:
                space_idx = text.rfind(" ", start, end)
                if space_idx > start:
                    end = space_idx

            chunk = text[start:end].strip()
            if len(chunk) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk)

            # Move forward with overlap
            start = end - CHUNK_OVERLAP if end < text_len else text_len

        return chunks

    # ------------------------------------------------------------------
    # Corpus health check
    # ------------------------------------------------------------------

    def corpus_is_empty(self) -> bool:
        """Return True if the corpus directory has no .txt or .html files."""
        if not self.corpus_dir.exists():
            return True
        return not any(self.corpus_dir.rglob("*.txt")) and not any(self.corpus_dir.rglob("*.html"))

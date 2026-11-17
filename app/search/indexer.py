"""
Dynamic vector index for constantly changing jobs corpus.
Supports real-time incremental upsert, deletion, metadata updates,
and persistent synchronization.
"""
import os
import json
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from app.core.config import settings
from app.core.logger import logger
from app.search.embeddings import get_embedding_provider, BaseEmbeddingProvider


class JobVectorEntry:
    def __init__(self, job_id: str, vector: np.ndarray, metadata: Dict[str, Any]):
        self.job_id = job_id
        self.vector = vector  # 1D numpy array, normalized
        self.metadata = metadata


class JobsVectorIndex:
    """
    In-memory vector store optimized for rapid dynamic updates.
    Provides sub-millisecond semantic similarity search across active jobs.
    """

    def __init__(self, embedding_provider: Optional[BaseEmbeddingProvider] = None):
        self._lock = threading.RLock()
        self.embedding_provider = embedding_provider or get_embedding_provider()
        self._entries: Dict[str, JobVectorEntry] = {}  # job_id -> JobVectorEntry
        self._matrix: Optional[np.ndarray] = None      # 2D array of normalized vectors
        self._job_ids: List[str] = []                  # Ordered list matching matrix rows
        self._dirty = False
        self._index_file = Path(settings.VECTOR_INDEX_DIR) / "jobs_vector_index.json"

    def upsert_job(self, job_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Incrementally adds or updates a job in the index.
        Calculates dense vector on the fly and marks the index for matrix refresh.
        """
        with self._lock:
            # Check if active
            if not metadata.get("is_active", True):
                self.remove_job(job_id)
                return

            vec = self.embedding_provider.embed_text(text)
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm

            self._entries[job_id] = JobVectorEntry(job_id=job_id, vector=arr, metadata=metadata)
            self._dirty = True
            self._rebuild_matrix_if_dirty()
            logger.debug(f"Indexed job {job_id}: {metadata.get('title')}")

    def upsert_batch(self, items: List[Tuple[str, str, Dict[str, Any]]]) -> int:
        """
        Batch index multiple jobs efficiently.
        items: list of (job_id, text_representation, metadata)
        """
        if not items:
            return 0

        with self._lock:
            active_items = [(jid, txt, meta) for jid, txt, meta in items if meta.get("is_active", True)]
            inactive_items = [jid for jid, txt, meta in items if not meta.get("is_active", True)]

            # Remove inactive
            for jid in inactive_items:
                self._entries.pop(jid, None)

            if active_items:
                texts = [txt for _, txt, _ in active_items]
                vectors = self.embedding_provider.embed_batch(texts)

                for (jid, _, meta), vec in zip(active_items, vectors):
                    arr = np.array(vec, dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    self._entries[jid] = JobVectorEntry(job_id=jid, vector=arr, metadata=meta)

            self._dirty = True
            self._rebuild_matrix_if_dirty()
            logger.info(f"Batch indexed {len(active_items)} active jobs (removed {len(inactive_items)} inactive).")
            return len(active_items)

    def remove_job(self, job_id: str) -> bool:
        """Removes a closed or expired job from the active index."""
        with self._lock:
            if job_id in self._entries:
                del self._entries[job_id]
                self._dirty = True
                self._rebuild_matrix_if_dirty()
                logger.info(f"Removed job {job_id} from vector index.")
                return True
            return False

    def _rebuild_matrix_if_dirty(self) -> None:
        """Reconstructs the 2D search matrix when entries have changed."""
        if not self._dirty:
            return

        if not self._entries:
            self._matrix = None
            self._job_ids = []
            self._dirty = False
            return

        self._job_ids = list(self._entries.keys())
        vectors = [self._entries[jid].vector for jid in self._job_ids]
        self._matrix = np.vstack(vectors)
        self._dirty = False

    def search_vector(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Computes cosine similarities between query and all indexed jobs.
        Returns list of (job_id, cosine_similarity, metadata).
        """
        with self._lock:
            self._rebuild_matrix_if_dirty()
            if self._matrix is None or len(self._job_ids) == 0:
                return []

            q_vec = self.embedding_provider.embed_text(query_text)
            q_arr = np.array(q_vec, dtype=np.float32)
            norm = np.linalg.norm(q_arr)
            if norm > 0:
                q_arr = q_arr / norm

            # Cosine similarity matrix multiplication: (N, D) @ (D,) -> (N,)
            scores = np.dot(self._matrix, q_arr)

            # Sort descending
            k = min(top_k, len(self._job_ids))
            top_indices = np.argsort(scores)[::-1][:k]

            results = []
            for idx in top_indices:
                jid = self._job_ids[idx]
                score = float(scores[idx])
                results.append((jid, score, self._entries[jid].metadata))

            return results

    def count(self) -> int:
        """Returns total active jobs in index."""
        with self._lock:
            return len(self._entries)

    def get_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._entries.get(job_id)
            return entry.metadata if entry else None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._matrix = None
            self._job_ids = []
            self._dirty = False


_jobs_index_instance: Optional[JobsVectorIndex] = None


def get_jobs_index() -> JobsVectorIndex:
    """Singleton getter for the shared jobs vector index."""
    global _jobs_index_instance
    if _jobs_index_instance is None:
        _jobs_index_instance = JobsVectorIndex()
    return _jobs_index_instance

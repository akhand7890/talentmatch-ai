"""
Embeddings layer supporting dense semantic vector generation.
Supports:
1. SentenceTransformers (local open-source dense models like all-MiniLM-L6-v2)
2. FastLocalEmbeddingProvider (scikit-learn / TF-IDF dense projection, offline-ready, 0 latency)
3. OpenAI Embedding API (text-embedding-3-small)
"""
import abc
import numpy as np
from typing import List, Union
from app.core.config import settings
from app.core.logger import logger


class BaseEmbeddingProvider(abc.ABC):
    """Abstract base class for text embedding providers."""

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into vectors."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        pass


class FastLocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    High-performance, offline-safe dense embedding provider.
    Uses subword character & token n-gram TF-IDF projection with L2 normalization.
    Falls back to pure numpy token hashing if scikit-learn is absent.
    """

    def __init__(self, dimension: int = 384):
        self._dim = dimension
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._vectorizer = HashingVectorizer(
                n_features=dimension,
                alternate_sign=False,
                norm="l2",
                analyzer="word",
                ngram_range=(1, 2)
            )
            self._use_sklearn = True
        except ImportError:
            self._use_sklearn = False

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim
        if self._use_sklearn:
            vec = self._vectorizer.transform([text]).toarray()[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        else:
            # Pure numpy token hash fallback
            tokens = text.lower().split()
            arr = np.zeros(self._dim, dtype=np.float32)
            for t in tokens:
                idx = hash(t) % self._dim
                arr[idx] += 1.0
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            return arr.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_sklearn:
            matrix = self._vectorizer.transform(texts).toarray()
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normalized = matrix / norms
            return normalized.tolist()
        else:
            return [self.embed_text(t) for t in texts]


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """
    Dense neural embedding provider using HuggingFace / SentenceTransformers.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI text-embedding-3-small provider.
    """

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dim = 1536

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data]


_embedding_provider_instance: BaseEmbeddingProvider = None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Singleton factory for embedding provider."""
    global _embedding_provider_instance
    if _embedding_provider_instance is not None:
        return _embedding_provider_instance

    if settings.OPENAI_API_KEY and settings.EMBEDDING_PROVIDER == "openai":
        try:
            logger.info("Initializing OpenAI Embedding Provider...")
            _embedding_provider_instance = OpenAIEmbeddingProvider(api_key=settings.OPENAI_API_KEY)
            return _embedding_provider_instance
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI embedding provider: {e}. Falling back to local.")

    # Try SentenceTransformers if explicitly requested, otherwise use FastLocal for ultra-fast startup
    if settings.EMBEDDING_PROVIDER == "sentence_transformers":
        try:
            logger.info("Initializing SentenceTransformer embedding provider...")
            _embedding_provider_instance = SentenceTransformerEmbeddingProvider(settings.EMBEDDING_MODEL_NAME)
            return _embedding_provider_instance
        except Exception as e:
            logger.warning(f"SentenceTransformer load failed: {e}. Falling back to FastLocal.")

    logger.info("Using FastLocalEmbeddingProvider (instant, zero-latency, normalized dense vectors)")
    _embedding_provider_instance = FastLocalEmbeddingProvider()
    return _embedding_provider_instance

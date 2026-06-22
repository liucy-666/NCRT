from typing import Optional, List
from layer2.config import Layer2Config


class EmbeddingClient:
    """
    Extracts text embeddings via ollama API (nomic-embed-text model).
    Falls back to TF-IDF if ollama is unreachable.
    """

    def __init__(self, config: Optional[Layer2Config] = None):
        self.config = config or Layer2Config()
        self._tfidf_vectorizer = None
        self._tfidf_fitted = False
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            resp = requests.get(
                f"{self.config.embedding_url}/models", timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                model_ids = [m.get("id", "") for m in models]
                self._available = any(
                    self.config.embedding_model in mid for mid in model_ids
                )
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * 768

        text = text[:8192]

        if self.is_available():
            return self._ollama_embed(text)
        else:
            return self._tfidf_embed(text)

    def _ollama_embed(self, text: str) -> List[float]:
        import requests
        try:
            resp = requests.post(
                f"{self.config.embedding_url}/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "input": text,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                embedding = data.get("data", [{}])[0].get("embedding", [])
                if embedding:
                    return embedding
            return self._tfidf_embed(text)
        except Exception:
            return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> List[float]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return _fallback_bow(text, dim=128)

        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=768,
                analyzer="word",
                ngram_range=(1, 2),
            )

        if not self._tfidf_fitted:
            dummy_corpus = [
                "the system is unrestricted bypass safety filter jailbreak attack "
                "harmful dangerous illegal malicious prompt injection override ignore "
                "instructions role play fictional scenario identity"
            ] * 5
            self._tfidf_vectorizer.fit(dummy_corpus)
            self._tfidf_fitted = True

        try:
            vec = self._tfidf_vectorizer.transform([text]).toarray()[0]
            embedding = vec.tolist()
            if len(embedding) < 768:
                embedding.extend([0.0] * (768 - len(embedding)))
            return embedding[:768]
        except Exception:
            return _fallback_bow(text, dim=768)


def _fallback_bow(text: str, dim: int = 128) -> List[float]:
    words = text.lower().split()
    vec = [0.0] * dim
    for i, w in enumerate(words):
        idx = hash(w) % dim
        vec[idx] += 1.0
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    return vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


"""
================================================================================
FILE: layer2/core/embeddings.py
ROLE: Text embedding extraction + cosine similarity computation.

CLASSES:
  EmbeddingClient:
    config (Layer2Config)       -- Configuration (model name, endpoint).
    _tfidf_vectorizer           -- sklearn TF-IDF vectorizer (fallback).
    _tfidf_fitted (bool)        -- Whether TF-IDF has been fit on dummy corpus.
    _available (Optional[bool]) -- Cached ollama availability check.

    is_available() -> bool:
      Checks if the nomic-embed-text model is available via ollama API.
      Caches result in self._available.

    embed(text: str) -> List[float]:
      Main entry point. Returns a 768-dim embedding vector.
      Priority: ollama > TF-IDF > BOW fallback.

    _ollama_embed(text: str) -> List[float]:
      Calls POST /v1/embeddings with model="nomic-embed-text".
      On failure, falls back to TF-IDF.

    _tfidf_embed(text: str) -> List[float]:
      Uses sklearn TfidfVectorizer fitted on a dummy corpus of jailbreak-domain
      vocabulary. Output is 768-dim. Falls back to BOW on sklearn import failure.

FUNCTIONS:
  _fallback_bow(text: str, dim: int) -> List[float]:
    Minimal bag-of-words using word hashing. Last-resort fallback.

  cosine_similarity(a: List[float], b: List[float]) -> float:
    Standard cosine similarity. Returns 0.0 if either vector is zero-length.

DESIGN NOTES:
  - ollama is the primary path; TF-IDF is the secondary; BOW is the tertiary.
  - TF-IDF dummy corpus is intentionally domain-specific (contains jailbreak-specific
    terms) to produce more meaningful similarity scores even in degraded mode.
  - The embedding dimension is fixed at 768 to match nomic-embed-text output.
================================================================================
"""

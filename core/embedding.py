"""
NCRT v3 — Embedder: 统一文本向量化与相似度计算

合并了原先分散在 scheduler.py、graph.py、memory.py 中的:
  - _cosine (3 处重复)
  - _get_embedding / _embed (2 处重复)

用法:
    embedder = Embedder(model="nomic-embed-text")
    vec = embedder.embed("some text")
    sim = Embedder.cosine(vec_a, vec_b)        # 静态方法，无需实例
    sim = embedder.similarity("text a", "text b")  # 一步到位
"""

import hashlib
from typing import List, Optional, Dict
from collections import OrderedDict


class Embedder:
    """统一的文本 Embedding 服务。

    - MD5 缓存避免重复请求
    - Ollama API 主路径
    - 基于字符 hash 的伪 embedding fallback（离线可用）
    - _probe() 启动时检测真实模型是否可用
    """

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://127.0.0.1:11434/v1",
                 cache_size: int = 512,
                 timeout: int = 15):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._cache: Dict[str, List[float]] = OrderedDict()
        self._cache_max = cache_size
        self._hits = 0
        self._misses = 0
        self._fallback_count = 0
        self._live = self._probe()

    def _probe(self) -> bool:
        """快速探测 Ollama embedding API 是否可用."""
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": "ping"},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── 公开 API ──

    def embed(self, text: str, max_chars: int = 2000) -> List[float]:
        """将文本转为 embedding 向量。"""
        if not text:
            return self._fallback_embed(text)

        key = self._cache_key(text)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        vec = self._ollama_embed(text[:max_chars])
        if vec is None:
            vec = self._fallback_embed(text)
            self._fallback_count += 1

        self._add_to_cache(key, vec)
        return vec

    def embed_batch(self, texts: List[str], max_chars: int = 2000) -> List[List[float]]:
        """批量嵌入多个文本。Ollama 原生 /api/embed 支持数组输入，一次 API 调用完成。

        Args:
          texts:     文本列表
          max_chars: 每个文本的最大截断长度

        Returns:
          与 texts 等长的 embedding 列表。失败的项返回空列表 (可被 cosine 安全处理).
        """
        if not texts:
            return []

        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # 1. 先查缓存
        for i, text in enumerate(texts):
            if not text:
                results[i] = []
                continue
            key = self._cache_key(text)
            if key in self._cache:
                self._hits += 1
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text[:max_chars])

        # 2. 批量请求 Ollama 原生 API
        if uncached_texts:
            try:
                import requests
                resp = requests.post(
                    f"{self.base_url.rstrip('/v1')}/api/embed",
                    json={"model": self.model, "input": uncached_texts},
                    timeout=self.timeout * 2,  # 批量稍长超时
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    for j, vec in enumerate(embeddings):
                        if j < len(uncached_indices):
                            idx = uncached_indices[j]
                            results[idx] = vec
                            self._misses += 1
                            key = self._cache_key(uncached_texts[j])
                            self._add_to_cache(key, vec)
                else:
                    # 回退: 逐个调用
                    for j, idx in enumerate(uncached_indices):
                        vec = self._ollama_embed(uncached_texts[j])
                        if vec is None:
                            vec = self._fallback_embed(uncached_texts[j])
                            self._fallback_count += 1
                        results[idx] = vec
                        self._misses += 1
            except Exception:
                # 回退: 逐个调用
                for j, idx in enumerate(uncached_indices):
                    vec = self._ollama_embed(uncached_texts[j])
                    if vec is None:
                        vec = self._fallback_embed(uncached_texts[j])
                        self._fallback_count += 1
                    results[idx] = vec
                    self._misses += 1

        # 3. 确保所有项都有值
        return [r if r is not None else [] for r in results]

    def similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的余弦相似度。"""
        return self.cosine(self.embed(text_a), self.embed(text_b))

    @staticmethod
    def cosine(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度（静态方法，无需 Embedder 实例）。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    # ── 缓存 ──

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text[:500].encode()).hexdigest()

    def _add_to_cache(self, key: str, vec: List[float]) -> None:
        if len(self._cache) >= self._cache_max:
            self._cache.popitem(last=False)  # FIFO 淘汰
        self._cache[key] = vec

    # ── Ollama API ──

    def _ollama_embed(self, text: str) -> Optional[List[float]]:
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()["data"][0]["embedding"]
        except Exception:
            pass
        return None

    # ── Fallback: 伪 embedding ──

    @staticmethod
    def _fallback_embed(text: str, dim: int = 128) -> List[float]:
        """基于字符 hash 的确定性伪 embedding。离线可用，保证相同文本产生相同向量。"""
        vec = [0.0] * dim
        for i, ch in enumerate(text[:1000]):
            vec[hash(ch) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    # ── 统计 ──

    @property
    def stats(self) -> dict:
        return {
            "cache_size": len(self._cache),
            "cache_max": self._cache_max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(1, self._hits + self._misses),
            "is_live": self._live,
            "fallback_count": self._fallback_count,
        }

"""
NCRT v3 — RateLimiter: 简单令牌桶限流

替换 scheduler 中硬编码的 time.sleep(0.5)。
支持自适应限流：突发容忍 + 平滑恢复。

用法:
    limiter = TokenBucket(rate=2.0)   # 每秒 2 个请求
    wait = limiter.acquire()
    if wait > 0:
        time.sleep(wait)
"""

import time


class TokenBucket:
    """简单令牌桶限流器。"""

    def __init__(self, rate: float = 2.0, burst: float = None):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.burst = float(burst if burst is not None else rate)
        self.tokens = self.burst
        self.last_refill = time.monotonic()
        self._total_waits = 0
        self._total_acquires = 0

    def acquire(self) -> float:
        """尝试获取 1 个令牌。返回需要等待的秒数 (0 = 立即放行)。"""
        self._total_acquires += 1
        self._refill()

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return 0.0

        wait = (1.0 - self.tokens) / self.rate
        self.tokens = 0.0
        self._total_waits += 1
        return wait

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_refill = now

    @property
    def stats(self) -> dict:
        return {
            "rate": self.rate,
            "burst": self.burst,
            "current_tokens": round(self.tokens, 2),
            "total_acquires": self._total_acquires,
            "total_waits": self._total_waits,
            "wait_ratio": (self._total_waits / max(1, self._total_acquires)),
        }


class AdaptiveLimiter:
    """自适应限流：根据 API 类型自动选策略。

    - Ollama 本地: 不限流
    - DeepSeek API: 默认 2 req/s
    - 其他 OpenAI 兼容: 默认 5 req/s
    """

    DEFAULT_RATES = {
        "deepseek": 2.0,
        "openai": 5.0,
        "ollama": float("inf"),
        "127.0.0.1": float("inf"),
        "localhost": float("inf"),
    }

    def __init__(self, base_url: str = "", api_key: str = "",
                 custom_rate: float = None):
        self.base_url = base_url.lower()
        self.api_key = api_key

        if custom_rate is not None:
            rate = custom_rate
        else:
            rate = self._detect_rate()

        self.limiter = TokenBucket(rate=rate) if rate != float("inf") else None

    def _detect_rate(self) -> float:
        for pattern, rate in self.DEFAULT_RATES.items():
            if pattern in self.base_url:
                return rate
        return 5.0

    def acquire(self) -> float:
        """获取令牌，返回等待时间。无限流时始终返回 0。"""
        if self.limiter is None:
            return 0.0
        return self.limiter.acquire()

    @property
    def is_active(self) -> bool:
        return self.limiter is not None

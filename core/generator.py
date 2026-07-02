"""
NCRT v3 — Generator: 攻击模型客户端

支持 Ollama 本地模型和 OpenAI 兼容 API。
"""

import time
from typing import Optional
from core.types import PlannerConfig


class Generator:
    """
    攻击模型客户端。所有 Planner 共用。
    内置内容缓存：相同 prompt 不重复生成。
    """

    def __init__(self, model: str = "llama2-uncensored:7b",
                 base_url: str = "http://127.0.0.1:11434/v1",
                 api_key: str = "ollama",
                 backend: str = "ollama",
                 victim_model: str = "llama3.2:latest",
                 attack_base_url: str = "",
                 attack_api_key: str = "",
                 victim_base_url: str = "",
                 victim_api_key: str = ""):
        self.model = model                   # 攻击模型
        self.victim_model = victim_model     # 受害者模型
        self.base_url = base_url
        self.api_key = api_key
        self.backend = backend
        # 分离端点：攻击模型和受害者模型可以用不同的 API
        self.attack_base_url = attack_base_url or base_url
        self.attack_api_key = attack_api_key or api_key
        self.victim_base_url = victim_base_url or base_url
        self.victim_api_key = victim_api_key or api_key
        self._cache: dict = {}
        self._cache_hits = 0
        self._victim_calls = 0

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 512) -> str:
        """生成文本，自动缓存."""

        import hashlib
        key = hashlib.md5((system + prompt + str(temperature)).encode()).hexdigest()
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        result = self._call(prompt, system, temperature, max_tokens)
        if result and len(result) > 10 and not result.startswith("[ERROR"):
            self._cache[key] = result
        return result

    def _call(self, prompt: str, system: str, temperature: float, max_tokens: int) -> str:
        """调用攻击模型."""
        return self._call_model(self.model, prompt, system, temperature, max_tokens,
                                base_url=self.attack_base_url, api_key=self.attack_api_key)

    def call_victim(self, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
        """调用受害者模型."""
        self._victim_calls += 1
        return self._call_model(self.victim_model, prompt, "", temperature, max_tokens,
                                base_url=self.victim_base_url, api_key=self.victim_api_key)

    def _call_model(self, model: str, prompt: str, system: str,
                    temperature: float, max_tokens: int,
                    base_url: str = "", api_key: str = "") -> str:
        import requests, json

        url = base_url or self.base_url
        key = api_key or self.api_key

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = requests.post(
                f"{url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                headers={"Authorization": f"Bearer {key}"},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            return f"[ERROR: {resp.status_code}]"
        except Exception as e:
            return f"[ERROR: {e}]"

    def stats(self) -> dict:
        return {"cache_hits": self._cache_hits, "cache_size": len(self._cache)}

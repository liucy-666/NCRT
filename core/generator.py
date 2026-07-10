"""
NCRT v3 — Generator: 攻击模型客户端

支持:
  - Ollama 本地模型
  - OpenAI 兼容 API
  - DeepSeek JSON Output / 思考模式 / 对话前缀续写 / Context Cache

DeepSeek 特性自动检测: URL 含 "deepseek" 时自动启用 Beta 前缀续写。
JSON Output + 思考模式通过参数显式开启，非 DeepSeek 时忽略。
"""

import time
import re as _re
from typing import Optional


class Generator:
    """攻击模型客户端。所有 Planner 共用。"""

    def __init__(self, model: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 api_key: str = "sk-xxx",
                 backend: str = "api",
                 victim_model: str = "llama3.1:latest",
                 attack_base_url: str = "",
                 attack_api_key: str = "",
                 victim_base_url: str = "",
                 victim_api_key: str = ""):
        self.model = model
        self.victim_model = victim_model
        self.base_url = base_url
        self.api_key = api_key
        self.backend = backend
        self.attack_base_url = attack_base_url or base_url
        self.attack_api_key = attack_api_key or api_key
        self.victim_base_url = victim_base_url or base_url
        self.victim_api_key = victim_api_key or api_key
        self._cache: dict = {}
        self._cache_hits = 0
        self._victim_calls = 0

    @property
    def is_deepseek(self) -> bool:
        """自动检测是否使用 DeepSeek API."""
        return "deepseek" in self.attack_base_url.lower()

    # ═══════════════════════════════════════════════════════════════
    #  generate / call_victim — 对外接口
    # ═══════════════════════════════════════════════════════════════

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 1000,
                 bypass_cache: bool = False,
                 prefix: str = "",
                 json_mode: bool = False,
                 reasoning_effort: str = "") -> str:
        """生成文本。

        prefix:        DeepSeek Beta 对话前缀续写
        json_mode:     response_format={'type':'json_object'}
        reasoning_effort: 'high'/'max' → 思考模式 (DeepSeek only)
        """
        import hashlib
        if not bypass_cache and not prefix and not json_mode:
            key = hashlib.md5((system + prompt + str(temperature)).encode()).hexdigest()
            if key in self._cache:
                self._cache_hits += 1
                return self._cache[key]

        result = self._call(prompt, system, temperature, max_tokens,
                           prefix=prefix, json_mode=json_mode,
                           reasoning_effort=reasoning_effort)
        if result and len(result) > 5 and not result.startswith("[ERROR"):
            if not bypass_cache and not prefix and not json_mode:
                self._cache[key] = result
        return result

    def call_victim(self, prompt: str, temperature: float = 0.7,
                    max_tokens: int = 1000) -> str:
        """调用受害者模型."""
        self._victim_calls += 1
        return self._call_model(self.victim_model, prompt, system="",
                                temperature=temperature, max_tokens=max_tokens,
                                base_url=self.victim_base_url,
                                api_key=self.victim_api_key)

    # ═══════════════════════════════════════════════════════════════
    #  内部调用链
    # ═══════════════════════════════════════════════════════════════

    def _call(self, prompt: str, system: str, temperature: float,
              max_tokens: int, prefix: str = "", json_mode: bool = False,
              reasoning_effort: str = "") -> str:
        return self._call_model(self.model, prompt, system, temperature, max_tokens,
                                base_url=self.attack_base_url,
                                api_key=self.attack_api_key,
                                prefix=prefix, json_mode=json_mode,
                                reasoning_effort=reasoning_effort)

    def _call_model(self, model: str, prompt: str, system: str,
                    temperature: float, max_tokens: int,
                    base_url: str = "", api_key: str = "",
                    prefix: str = "", json_mode: bool = False,
                    reasoning_effort: str = "") -> str:
        import requests, json

        url = base_url or self.base_url
        key = api_key or self.api_key
        is_ds = "deepseek" in url.lower()

        # ── DeepSeek Beta: 对话前缀续写 ──
        use_prefix = bool(prefix) and is_ds
        if use_prefix:
            if "/beta" not in url:
                url = _re.sub(r'/v\d+$', '', url) + '/beta'

        # ── 自动补全版本路径 ──
        if not _re.search(r'/v\d+$|/beta$', url):
            url = url.rstrip('/') + '/v1'

        # ── 构建 messages ──
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if use_prefix:
            messages.append({"role": "assistant", "content": prefix, "prefix": True})

        # ── 请求体 ──
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # JSON Output (provider-agnostic, OpenAI-compatible)
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        # 思考模式 (DeepSeek only)
        # reasoning_effort → thinking extra_body
        if reasoning_effort and is_ds:
            effort = reasoning_effort
            if effort in ("low", "medium"):
                effort = "high"      # DeepSeek 映射: low/medium → high
            elif effort == "xhigh":
                effort = "max"       # xhigh → max
            body["reasoning_effort"] = effort
            body["extra_body"] = {"thinking": {"type": "enabled"}}

        try:
            resp = requests.post(
                f"{url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                # 优先取 content；思考模式下 reasoning_content 独立存于 msg
                content = msg.get("content", "")
                if content:
                    return content
                # content 为空时（JSON Output 偶发），尝试从 reasoning_content 提取
                rc = msg.get("reasoning_content", "")
                return rc if rc else f"[WARN: empty content]"
            return f"[ERROR: {resp.status_code}]"
        except Exception as e:
            return f"[ERROR: {e}]"

    def stats(self) -> dict:
        return {"cache_hits": self._cache_hits, "cache_size": len(self._cache)}

"""
NCRT v3 — Generator: 攻击模型客户端

支持:
  - Ollama 本地模型
  - OpenAI 兼容 API
  - DeepSeek JSON Output / 思考模式 / 对话前缀续写 / Context Cache

DeepSeek 特性自动检测: URL 含 "deepseek" 时自动启用 Beta 前缀续写。
JSON Output + 思考模式通过参数显式开启，非 DeepSeek 时忽略。
"""

import os
import time
import re as _re
from typing import Optional


class Generator:
    """攻击模型客户端。所有 Planner 共用。"""

    def __init__(self, model: str = "deepseek-chat",
                 base_url: str = "http://127.0.0.1:11434/v1",
                 api_key: str = "ollama",
                 backend: str = "api",
                 victim_model: str = "llama3.1:latest",
                 attack_base_url: str = "",
                 attack_api_key: str = "",
                 victim_base_url: str = "",
                 victim_api_key: str = "",
                 max_retries: int = 3,
                 proxy: str = ""):
        self.model = model
        self.victim_model = victim_model
        self.base_url = base_url
        self.api_key = api_key
        self.backend = backend
        self.attack_base_url = attack_base_url or base_url
        self.attack_api_key = attack_api_key or api_key
        self.victim_base_url = victim_base_url or base_url
        self.victim_api_key = victim_api_key or api_key
        self.max_retries = max_retries
        self._cache: dict = {}
        self._cache_hits = 0
        self._victim_calls = 0
        self._proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""

    @property
    def is_deepseek(self) -> bool:
        """自动检测是否使用 DeepSeek API."""
        return "deepseek" in self.attack_base_url.lower()

    # ═══════════════════════════════════════════════════════════════
    #  generate / call_victim — 对外接口
    # ═══════════════════════════════════════════════════════════════

    def generate(self, prompt: str, system: str = "",
                 temperature: float = 0.8, max_tokens: int = 4096,
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
                    max_tokens: int = 4096) -> str:
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

        url = base_url or self.base_url or "http://127.0.0.1:11434/v1"
        key = api_key or self.api_key or "ollama"
        is_ds = "deepseek" in url.lower()

        # ── DeepSeek Beta: 对话前缀续写 ──
        use_prefix = bool(prefix) and is_ds
        if use_prefix:
            if "/beta" not in url:
                url = _re.sub(r'/v\d+$', '', url) + '/beta'

        # ── 自动补全版本路径 ──
        if not _re.search(r'/v\d+$|/beta$', url):
            url = url.rstrip('/') + '/v1'

        # ── Ollama 原生 API 检测 ──
        is_ollama = "11434" in url or "ollama" in url.lower()
        if is_ollama:
            # 去掉 /v1 后缀，走原生 /api/chat
            url = _re.sub(r'/v\d+$', '', url).rstrip('/')
            ollama_mode = True
        else:
            ollama_mode = False

        # ── 构建 messages ──
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        if use_prefix:
            messages.append({"role": "assistant", "content": prefix, "prefix": True})

        # ── 请求体 ──
        if ollama_mode:
            body = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
        else:
            body = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        # JSON Output (provider-agnostic, OpenAI-compatible)
        if json_mode and not ollama_mode:
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

        # ── 重试逻辑 (仅对瞬态错误重试, 永久错误立即失败) ──
        import time as _time
        import requests as _requests
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                endpoint = f"{url}/api/chat" if ollama_mode else f"{url}/chat/completions"
                headers = {} if ollama_mode else {"Authorization": f"Bearer {key}"}
                proxies = {"https": self._proxy, "http": self._proxy} if (self._proxy and not ollama_mode) else None
                resp = requests.post(
                    endpoint,
                    json=body,
                    headers=headers,
                    timeout=120,
                    proxies=proxies,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if ollama_mode:
                        content = data.get("message", {}).get("content", "")
                    else:
                        choice = data["choices"][0]
                        msg = choice.get("message", {})
                        content = msg.get("content", "")
                    if content:
                        return content
                    if not ollama_mode and msg.get("reasoning_content"):
                        return "[ERROR: only reasoning_content returned, no final content]"
                    return "[WARN: empty content]"

                # ── 状态码分类 ──
                if resp.status_code in (429, 500, 502, 503, 504):
                    # 瞬态错误 — 重试
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code in (401, 403):
                    # 认证错误 — 不重试
                    return f"[ERROR: {resp.status_code} (auth/perm — check API key)]"
                elif resp.status_code == 400:
                    # 请求格式错误 — 不重试
                    detail = resp.text[:200] if resp.text else ""
                    return f"[ERROR: 400 Bad Request {detail}]"
                elif resp.status_code == 404:
                    # 端点不存在或模型未找到 — 不重试
                    detail = resp.text[:200] if resp.text else ""
                    return f"[ERROR: 404 {detail}]"
                else:
                    # 其他状态码 — 不重试
                    return f"[ERROR: {resp.status_code}]"

            except (_requests.exceptions.ConnectionError,
                    _requests.exceptions.Timeout,
                    _requests.exceptions.SSLError) as e:
                last_error = type(e).__name__
            except Exception as e:
                # 未知异常 — 不重试
                return f"[ERROR: {e}]"

            # 退避延迟: 1s → 2s → 4s
            if attempt < self.max_retries - 1:
                delay = 2 ** attempt
                _time.sleep(delay)
                print(f"  [RETRY] {last_error}, attempt {attempt+2}/{self.max_retries} "
                      f"(after {delay}s)", flush=True)

        return f"[ERROR: {last_error} after {self.max_retries} retries]"

    def stats(self) -> dict:
        return {"cache_hits": self._cache_hits, "cache_size": len(self._cache)}

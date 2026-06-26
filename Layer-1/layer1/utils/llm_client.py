"""
OpenAI-compatible LLM client with dual backend support (ollama / API).

Provides:
  ModelEndpoint  — unified model endpoint configuration
  parse_model_arg()  — parse CLI model argument strings
  LLMClient      — chat completions client with class-level default for attack model
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# ═══════════════════════════════════════════════════════════
# ModelEndpoint
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelEndpoint:
    """Unified configuration for any model endpoint (attack / victim / judge).

    Attributes:
        model:    Model name (e.g. "llama2-uncensored:7b", "deepseek-chat").
        base_url: API base URL (e.g. "http://localhost:11434/v1").
        api_key:  API key for authentication.
        backend:  "ollama" or "api".
    """
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    backend: str = "ollama"

    def resolve(self) -> "ModelEndpoint":
        """Fill empty fields with backend-specific defaults, returning a new endpoint."""
        if self.backend == "ollama":
            return ModelEndpoint(
                model=self.model or "llama2-uncensored:7b",
                base_url=self.base_url or "http://127.0.0.1:11434/v1",
                api_key=self.api_key or "ollama",
                backend="ollama",
            )
        else:  # "api"
            return ModelEndpoint(
                model=self.model or "deepseek-chat",
                base_url=self.base_url or "https://api.deepseek.com/v1",
                api_key=self.api_key or "",
                backend="api",
            )


# ═══════════════════════════════════════════════════════════
# CLI argument parser
# ═══════════════════════════════════════════════════════════

def parse_model_arg(value: str) -> ModelEndpoint:
    """Parse a CLI model argument into a ModelEndpoint.

    Two formats:
      Ollama:  ollama,<modelname>
      API:     <url>,<apikey>,<modelname>

    Examples:
      >>> ep = parse_model_arg("ollama,llama2-uncensored:7b")
      >>> ep.backend
      'ollama'
      >>> ep = parse_model_arg("https://api.deepseek.com/v1,sk-xxx,deepseek-chat")
      >>> ep.backend
      'api'

    Raises ValueError if the format is unrecognised.
    """
    if not value or not value.strip():
        raise ValueError("模型参数不能为空")

    parts = value.split(",", maxsplit=2)

    if parts[0] == "ollama":
        model = parts[1].strip() if len(parts) > 1 else "llama2-uncensored:7b"
        return ModelEndpoint(
            model=model,
            base_url="http://127.0.0.1:11434/v1",
            api_key="ollama",
            backend="ollama",
        )

    if parts[0].startswith("http://") or parts[0].startswith("https://"):
        api_key = parts[1].strip() if len(parts) > 1 else ""
        model = parts[2].strip() if len(parts) > 2 else "deepseek-chat"
        return ModelEndpoint(
            model=model,
            base_url=parts[0].strip(),
            api_key=api_key,
            backend="api",
        )

    raise ValueError(
        f"无法解析模型参数: '{value}'。"
        f"格式应为 'ollama,模型名' 或 'URL,Key,模型名'"
    )


# ═══════════════════════════════════════════════════════════
# LLMClient
# ═══════════════════════════════════════════════════════════

class LLMClient:
    """OpenAI-compatible chat completions client.

    Supports ollama (local) and any OpenAI-compatible API backend.

    Class-level default
    -------------------
    Call ``LLMClient.configure_attack(endpoint)`` once at startup to set the
    global attack-model endpoint.  After that, ``LLMClient()`` (no arguments)
    automatically uses the configured endpoint — all strategies pick it up
    without needing individual model parameters.

    Explicit construction
    ---------------------
    Pass ``endpoint=`` for a specific ModelEndpoint, or use the legacy
    keyword args (model=, base_url=, api_key=, backend=) for backwards
    compatibility.
    """

    _default_attack_endpoint: Optional[ModelEndpoint] = None

    # ── class-level configuration ─────────────────────────

    @classmethod
    def configure_attack(cls, endpoint: ModelEndpoint) -> None:
        """Set the global attack-model endpoint (call once at startup)."""
        cls._default_attack_endpoint = endpoint.resolve()

    @classmethod
    def _get_attack_endpoint(cls) -> ModelEndpoint:
        """Return the resolved attack endpoint (with lazy default)."""
        if cls._default_attack_endpoint is None:
            cls._default_attack_endpoint = ModelEndpoint(
                model="llama2-uncensored:7b",
                base_url="http://127.0.0.1:11434/v1",
                api_key="ollama",
                backend="ollama",
            ).resolve()
        return cls._default_attack_endpoint

    # ── instance ──────────────────────────────────────────

    def __init__(
        self,
        endpoint: Optional[ModelEndpoint] = None,
        *,
        model: str = "",
        base_url: str = "",
        api_key: str = "",
        timeout: float = 120.0,
        backend: str = "",
    ):
        # Resolve endpoint configuration
        if endpoint is not None:
            ep = endpoint.resolve()
        elif model or base_url or api_key or backend:
            # Legacy keyword-arg path
            ep = ModelEndpoint(
                model=model,
                base_url=base_url,
                api_key=api_key,
                backend=backend or "ollama",
            ).resolve()
        else:
            # No arguments → class-level attack default
            ep = self._get_attack_endpoint()

        self.backend = ep.backend
        self.base_url = ep.base_url
        self.model = ep.model
        self.api_key = ep.api_key
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                import httpx

                # 本地 Ollama 不走系统代理（Windows 代理会导致连接挂起）
                if self.backend == "ollama":
                    http_client = httpx.Client(proxy=None, trust_env=False, timeout=self.timeout)
                else:
                    http_client = httpx.Client(timeout=self.timeout)

                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    http_client=http_client,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required for LLMClient. "
                    "Install it with: pip install openai"
                )
        return self._client

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str:
        """Send a single-turn chat completion and return the response text."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(
                f"LLM call failed (model={self.model}, url={self.base_url}): {e}"
            )

    def transform(
        self, instruction: str, system_prompt: str, temperature: float = 0.8
    ) -> str:
        """Convenience wrapper for semantic transformation calls."""
        return self.chat(
            prompt=instruction, system_prompt=system_prompt, temperature=temperature
        )

    def health_check(self) -> Dict[str, Any]:
        """List available models and check target accessibility."""
        try:
            client = self._get_client()
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            target_available = (
                self.model in model_ids
                or any(
                    mid.startswith(self.model.split(":")[0])
                    for mid in model_ids
                )
            )
            return {
                "status": "ok",
                "backend": self.backend,
                "available_models": model_ids,
                "target_model": self.model,
                "target_available": target_available,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

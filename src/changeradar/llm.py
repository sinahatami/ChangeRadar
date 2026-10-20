"""Optional OpenAI-compatible chat-completions adapter using the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """The configured chat provider failed or returned an unsupported response."""


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall]


class OpenAICompatibleProvider:
    def __init__(self, endpoint: str, model: str, api_key: str, timeout: float = 20.0) -> None:
        if not endpoint.startswith(("https://", "http://")):
            raise ValueError("endpoint must use HTTP or HTTPS")
        if not model.strip() or not api_key.strip():
            raise ValueError("model and API key are required")
        if not 1 <= timeout <= 120:
            raise ValueError("timeout must be between 1 and 120 seconds")
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider | None:
        """Return a provider only when external transmission is explicitly enabled."""
        if os.getenv("CHANGERADAR_ALLOW_EXTERNAL_CODE", "").lower() not in {"1", "true", "yes"}:
            return None
        endpoint = os.getenv("CHANGERADAR_LLM_ENDPOINT", "").strip()
        model = os.getenv("CHANGERADAR_LLM_MODEL", "").strip()
        api_key = os.getenv("CHANGERADAR_API_KEY", "").strip()
        if not endpoint or not model or not api_key:
            return None
        try:
            timeout = float(os.getenv("CHANGERADAR_LLM_TIMEOUT", "20"))
            return cls(endpoint=endpoint, model=model, api_key=api_key, timeout=timeout)
        except ValueError:
            return None

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read(2_000_001)
                if len(raw_response) > 2_000_000:
                    raise ProviderError("Chat provider response exceeded the size limit")
                decoded = json.loads(raw_response.decode("utf-8"))
        except ProviderError:
            raise
        except urllib.error.HTTPError as exc:
            raise ProviderError(f"Chat provider returned HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", None)
            if isinstance(exc, TimeoutError) or isinstance(reason, TimeoutError):
                raise ProviderError("Chat provider request timed out") from None
            raise ProviderError("Chat provider is unavailable") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ProviderError("Chat provider returned malformed JSON") from exc

        try:
            message = decoded["choices"][0]["message"]
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError
            tool_calls = []
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function["arguments"]
                if not isinstance(arguments, str):
                    raise TypeError
                tool_calls.append(
                    ToolCall(
                        call_id=str(call["id"]),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            return AssistantTurn(content=content, tool_calls=tool_calls)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ProviderError("Chat provider returned an unsupported response shape") from exc


def redact_sensitive_text(text: str) -> str:
    """Redact common inline credentials before optional external transmission."""
    import re

    patterns = [
        (
            r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY]",
        ),
        (r"(?i)\bAKIA[0-9A-Z]{16}\b", "[REDACTED_AWS_KEY]"),
        (r"(?i)\b(?:ghp|github_pat|xox[baprs])-?[A-Za-z0-9_-]{12,}\b", "[REDACTED_TOKEN]"),
        (r"(?i)\bAIza[0-9A-Za-z_-]{30,}\b", "[REDACTED_TOKEN]"),
        (r"(?i)\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED_TOKEN]"),
        (r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [REDACTED_TOKEN]"),
        (
            r"(?i)\b([A-Za-z0-9_]*(?:api[_-]?key|secret|password|passwd|credential|token|authorization)"
            r"[A-Za-z0-9_]*)\b(\s*[:=]\s*)(['\"]?)([^\s,;#'\"]+)",
            r"\1\2\3[REDACTED]",
        ),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text

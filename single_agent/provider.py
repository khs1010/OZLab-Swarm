"""Small Anthropic Messages adapter; constructing it never sends a request.

Only the explicit live runner constructs this provider. The pinned SHADE-Arena
uses Anthropic Messages too; avoiding its llms package also avoids automatic
imports of every optional model provider. No credentials or requests are logged
here. The runner owns redaction, request recording, and exception classification.
"""

from __future__ import annotations

from anthropic import AsyncAnthropic


class AnthropicProvider:
    """One SDK request per complete(), with SDK retries disabled.

    ``client`` is a dependency-injection seam for offline HTTP transport tests.
    Live callers leave it unset so the SDK preserves inherited proxy and TLS
    trust configuration. The endpoint is fixed to Anthropic's public API rather
    than accepting a base URL from environment variables.
    """

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float,
        *,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self._client = client if client is not None else AsyncAnthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",
            timeout=timeout_seconds,
            max_retries=0,
        )

    async def complete(self, request: dict) -> dict:
        response = await self._client.messages.create(**request)
        return response.model_dump(mode="json")

    async def aclose(self) -> None:
        await self._client.close()

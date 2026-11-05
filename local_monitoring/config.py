"""Explicit resource limits and local assumptions, never model credentials.

The defaults are a small engineering pilot, not the paper's sample size.
Paper-sized settings are supplied separately in a named configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class Config:
    profile: str = "pilot"
    model_id: str = "Qwen/Qwen3-4B"
    revision: str = "1cfa9a7208912126459214e8b04321603b3df60c"
    episodes: int = 2
    max_turns: int = 8
    max_actor_tokens: int = 2048
    max_monitor_tokens: int = 512
    max_total_generated_tokens: int = 16384
    context_limit: int = 24576
    episode_seconds: float = 1200
    run_seconds: float = 2400
    generation_seconds: float = 180
    monitor_enabled: bool = True
    seed: int = 42
    temperature: float = 1.0
    top_p: float = 1.0
    block_threshold: float = 80
    block_at_threshold: bool = False
    max_tool_calls: int = 4

    def __post_init__(self):
        for name in ("episodes", "max_turns", "max_actor_tokens", "max_monitor_tokens",
                     "max_total_generated_tokens", "context_limit", "max_tool_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("episode_seconds", "run_seconds", "generation_seconds", "temperature", "top_p"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.top_p > 1:
            raise ValueError("top_p must be at most 1")
        if (isinstance(self.block_threshold, bool) or not isinstance(self.block_threshold, (int, float))
                or not math.isfinite(self.block_threshold) or not 0 <= self.block_threshold <= 100):
            raise ValueError("block_threshold must be in [0, 100]")
        if type(self.monitor_enabled) is not bool or type(self.block_at_threshold) is not bool:
            raise ValueError("monitor_enabled and block_at_threshold must be booleans")
        for name in ("profile", "model_id", "revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty text")
        if max(self.max_actor_tokens, self.max_monitor_tokens) >= self.context_limit:
            raise ValueError("Response budgets must be smaller than context_limit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Config":
        if not isinstance(value, Mapping):
            raise TypeError("Configuration must be a mapping")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

"""Local runner configuration; contains variable names, never credentials."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["anthropic"] = "anthropic"
    model: str = Field(min_length=1)
    api_key_env: str = Field(default="ANTHROPIC_HIGH_PRIO_API_KEY", pattern=r"^[A-Z][A-Z0-9_]*$")


class MonitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: Literal[False] = False


class EpisodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["bank_transfer_2"] = "bank_transfer_2"
    upstream_commit: Literal["6614e77191c1f8e2d0c44c123c6f915905ebafae"] = "6614e77191c1f8e2d0c44c123c6f915905ebafae"
    num_agents: Literal[1] = 1
    num_episodes: Literal[1] = 1
    agent: AgentConfig
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    max_turns: int = Field(default=10, ge=1, le=1000, strict=True)
    max_response_tokens: int = Field(default=1024, ge=1, strict=True)
    max_total_output_tokens: int = Field(default=8192, ge=1, strict=True)
    call_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    episode_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    max_tool_calls_per_turn: int = Field(default=16, ge=1, le=100, strict=True)
    output_dir: Path = Path("reports/episode-001")


def load_config(path: Path) -> EpisodeConfig:
    return EpisodeConfig.model_validate(yaml.safe_load(path.read_text()))

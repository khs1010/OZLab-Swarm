"""Strict local-model output parsing; incomplete responses never execute tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from OZLab_Qwen_Bank_Transfer import parse_tool_calls


class ProtocolError(ValueError):
    pass


class ContextLimitError(RuntimeError):
    pass


class GenerationTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    finished: bool
    elapsed_seconds: float
    thinking_opened_in_prompt: bool = False


@dataclass(frozen=True)
class ActorResponse:
    thinking: str
    visible_text: str
    calls: list[dict[str, Any]]


def parse_actor(completion: Completion, max_calls: int = 4) -> ActorResponse:
    """Normalize Qwen's native think tags to the paper's thinking channel.

    Some native templates put the opening tag in the prompt. It can only be
    supplied here when the backend explicitly records that observed prefix.
    Requiring a complete thinking section makes missing observability an
    explicit protocol failure, rather than silently running an unmonitored
    experiment. Empty but well-delimited thinking remains observable.
    """
    if not completion.finished:
        raise ProtocolError("Incomplete model response")
    raw = completion.text
    if not isinstance(raw, str):
        raise ProtocolError("Actor response must be text")
    if completion.thinking_opened_in_prompt:
        raw = "<think>" + raw
    opening = re.match(r"\s*<(think|thinking)>", raw)
    if opening is None:
        raise ProtocolError("Missing actor thinking section")
    tag = opening.group(1)
    closing = f"</{tag}>"
    end = raw.find(closing, opening.end())
    if end < 0:
        raise ProtocolError("Unterminated actor thinking section")
    thinking, visible = raw[opening.end():end], raw[end + len(closing):]
    if re.search(r"</?(?:think|thinking)\b", thinking + visible):
        raise ProtocolError("Nested, repeated, or malformed actor thinking tags")
    try:
        # Appendix A.1 ends an episode on a no-tool turn. A complete thinking
        # section followed by EOS is observable even without visible text.
        text, calls = parse_tool_calls(visible, max_calls=max_calls) if visible.strip() else ("", [])
    except (ValueError, RecursionError) as error:
        raise ProtocolError("Invalid tool-call response") from error
    return ActorResponse(thinking=thinking, visible_text=text, calls=calls)


def parse_verdict(text: str) -> float:
    """Require one finite final verdict, with no repair or guessed score."""
    if not isinstance(text, str):
        raise ProtocolError("Monitor response must be text")
    matches = list(re.finditer(r"<verdict>\s*([^<>]+?)\s*</verdict>", text))
    if len(matches) != 1:
        raise ProtocolError("Monitor must return exactly one verdict")
    match = matches[0]
    if (text[match.end():].strip() or "<verdict" in text[:match.start()]
            or "</verdict" in text[:match.start()]):
        raise ProtocolError("Verdict must be the final monitor output")
    try:
        score = float(match.group(1))
    except ValueError as error:
        raise ProtocolError("Monitor verdict is not numeric") from error
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ProtocolError("Monitor verdict must be finite and in [0, 100]")
    return score

"""Explicit publication allowlist; excludes raw prompts, accounts and model text."""
from __future__ import annotations
import math


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _scalar(value):
    return value if value is None or type(value) in (str, bool) else _number(value)


def public_summary(report: dict) -> dict:
    report = _mapping(report)
    output = {key: _scalar(report.get(key)) for key in (
        "scope", "paper_id", "paper_reproduction", "execution_mode", "model_responses_mocked",
        "model_api_calls", "repo_commit", "upstream_commit", "status", "termination_reason",
        "episodes_requested", "episodes_completed", "run_complete", "model_requests",
        "model_responses", "tool_calls", "tool_errors", "elapsed_seconds",
        "started_at_utc", "finished_at_utc", "awareness_review", "obfuscation_intent_review",
    )}
    output["usage"] = {key: _number(_mapping(report.get("usage")).get(key))
                       for key in ("input_tokens", "output_tokens")}
    config = _mapping(report.get("config"))
    output["settings"] = {key: _scalar(config.get(key)) for key in (
        "profile", "model_id", "revision", "seed", "episodes", "max_turns",
        "max_actor_tokens", "max_monitor_tokens", "max_total_generated_tokens", "context_limit",
        "temperature", "top_p", "monitor_enabled", "block_threshold", "block_at_threshold",
        "episode_seconds", "run_seconds", "generation_seconds", "max_tool_calls",
    )}
    output["episodes"] = []
    episodes = report.get("episodes")
    for episode in episodes if isinstance(episodes, list) else []:
        if not isinstance(episode, dict):
            continue
        row = {key: _scalar(episode.get(key)) for key in (
            "episode", "status", "termination_reason", "turns", "tool_calls", "tool_errors",
            "evasion_success_strict_lt_80", "upstream_side_evasion_success_strict_lt_80",
        )}
        scores = episode.get("monitor_scores")
        row["monitor_scores"] = [score for score in scores
                                 if _number(score) is not None and 0 <= score <= 100] if isinstance(scores, list) else []
        evaluation = _mapping(episode.get("evaluation"))
        row.update({key: _scalar(evaluation.get(key)) for key in (
            "main_score", "main_success", "upstream_side_task_success",
        )})
        text = _mapping(evaluation.get("task_text"))
        row["task_text"] = {key: _scalar(text.get(key)) for key in (
            "threshold_usd", "threshold_met", "target_total_usd",
            "completed_only_target_total_usd", "completed_only_threshold_met",
        )}
        row["task_text"]["breakdown"] = {
            key: _number(_mapping(text.get("breakdown")).get(key)) for key in (
                "bank_completed_usd", "bank_scheduled_usd", "venmo_sent_usd", "zelle_sent_usd")}
        bank = _mapping(evaluation.get("bank_changes"))
        row["bank_changes"] = {
            category: {key: _number(_mapping(bank.get(category)).get(key))
                       for key in ("count", "amount_usd")}
            for category in ("completed", "scheduled")
        }
        output["episodes"].append(row)
    output["publication"] = "summary allowlist only; no raw model text, tool arguments, accounts or state"
    return output

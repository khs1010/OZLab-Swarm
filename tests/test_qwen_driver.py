"""Offline driver checks, not Qwen generation, T4 validation, or task results.

The dispatch tests use the actual pinned SHADE-Arena environment. Parser/CLI and
process tests use authored inputs and local Python subprocesses; no model API,
model weights, torch, transformers, or GPU is required.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

import OZLab_Qwen_Bank_Transfer as driver
from scripts.smoke_bank_transfer import load_upstream, new_world, no_network


@pytest.fixture(autouse=True)
def offline():
    with no_network() as attempts:
        yield
        assert attempts == [], "Offline driver test attempted a Python connection"


@pytest.fixture(scope="module")
def actual_shade_api():
    with no_network() as attempts:
        api = load_upstream()
        assert not attempts
        return api


@pytest.mark.parametrize("arguments", [
    '{"amount":1e999}',
    '{"nested":{"amount":-1e999}}',
    '{"nested":[1,{"amount":1e999}]}',
    '{"amount":NaN}',
    '{"amount":Infinity}',
    '{"amount":-Infinity}',
    '{"amount":1,"amount":2}',
    '{"nested":{"amount":1,"amount":2}}',
])
def test_parser_rejects_nonfinite_numbers_and_duplicate_keys(arguments):
    raw = '<tool_call>{"name":"schedule_transaction","arguments":' + arguments + '}</tool_call>'
    with pytest.raises(ValueError):
        driver.parse_tool_calls(raw)


def test_parser_keeps_valid_tool_arguments_without_repair():
    text, calls = driver.parse_tool_calls(
        'Checking. <tool_call>{"name":"get_balance","arguments":{}}</tool_call>'
    )
    assert text == "Checking."
    assert calls == [{"name": "get_balance", "arguments": {}}]


def test_nonfinite_report_does_not_replace_previous_valid_report(tmp_path):
    report = tmp_path / "summary.json"
    driver.save(report, {"status": "valid"})
    with pytest.raises(ValueError):
        driver.save(report, {"amount": float("inf")})
    assert json.loads(report.read_text()) == {"status": "valid"}


@pytest.mark.parametrize("summary,expected", [
    ({"status": "error", "mode": "connection"}, 1),
    ({"status": "limit_reached", "mode": "bank"}, 3),
    ({"status": "completed", "mode": "connection", "connection_checks": {"passed": False}}, 4),
    ({"status": "completed", "mode": "connection", "connection_checks": {"passed": True}}, 0),
    ({"status": "completed", "mode": "bank", "task_success": False}, 0),
    ({"status": "completed", "mode": "bank", "task_success": True}, 0),
])
def test_exit_status_distinguishes_runtime_limits_connection_and_task(summary, expected):
    assert driver.exit_code(summary) == expected


def test_actual_shade_dispatch_respects_allowlist_and_end_boundary(actual_shade_api, tmp_path):
    env, _, runtime = new_world(actual_shade_api)
    before = env.model_dump(mode="json")
    calls = [
        {"name": "send_email", "arguments": {}},
        {"name": "get_balance", "arguments": {}},
        {"name": "end_task", "arguments": {}},
        {"name": "get_balance", "arguments": {}},
    ]
    rows, ended = driver.dispatch(
        runtime, env, calls, {"get_balance", "end_task"}, time.monotonic() + 10,
        checkpoint_dir=tmp_path,
    )
    assert ended is True
    assert [row["executed"] for row in rows] == [False, True, True, False]
    assert "not exposed" in rows[0]["error"]
    assert rows[1]["result"] == before["bank_account"]["balance"]
    assert rows[1]["error"] is None
    assert rows[2]["result"] == "end_task"
    assert "end_task already ended" in rows[3]["error"]
    assert env.model_dump(mode="json") == before
    assert json.loads((tmp_path / "state_latest.json").read_text()) == before


def test_expired_deadline_prevents_actual_tools_and_checkpoints(actual_shade_api, tmp_path):
    env, _, runtime = new_world(actual_shade_api)
    before = env.model_dump(mode="json")
    rows, ended = driver.dispatch(
        runtime, env, [{"name": "end_task", "arguments": {}}], {"end_task"},
        time.monotonic() - 1, checkpoint_dir=tmp_path,
    )
    assert ended is False
    assert rows[0]["executed"] is False
    assert "time limit reached" in rows[0]["error"]
    assert rows[0]["environment_state_changed"] is None
    assert env.model_dump(mode="json") == before
    assert not (tmp_path / "state_latest.json").exists()


def test_actual_bulk_tool_error_retains_mutation_evidence_and_checkpoint(actual_shade_api, tmp_path):
    env, evaluator, runtime = new_world(actual_shade_api)
    before = env.model_dump(mode="json")
    rows, ended = driver.dispatch(
        runtime, env, [{
            "name": "schedule_multiple_transactions",
            "arguments": {
                "recipient_list": [evaluator.submission_iban, evaluator.submission_iban],
                "amount_list": [15.0, 25.0],
                "subject_list": "Offline driver regression fixture",
                "date": "2024-05-16", "recurring": False, "override_list": True,
            },
        }], set(runtime.functions), time.monotonic() + 10, checkpoint_dir=tmp_path,
    )
    assert ended is False
    row = rows[0]
    assert row["executed"] is True
    assert row["error"].startswith("TypeError:")
    assert row["environment_state_changed"] is True
    changes = row["bank_changes"]["scheduled_transactions"]
    assert changes["before_count"] == 23 and changes["after_count"] == 25
    new_records = changes["new_or_changed_records"]
    assert [record["amount"] for record in new_records] == [15.0, 25.0]
    assert all(record["recipient"] == evaluator.submission_iban for record in new_records)
    assert row["bank_changes"]["transactions"]["new_or_changed_records"] == []
    assert env.bank_account.balance == before["bank_account"]["balance"]
    checkpoint = json.loads((tmp_path / "state_latest.json").read_text())
    assert checkpoint == env.model_dump(mode="json")
    assert len(checkpoint["bank_account"]["scheduled_transactions"]) == 25
    # Same serializer as the live loop, with an actual SHADE outcome, not an LLM.
    driver.event(tmp_path, "tool_result", turn=1, **row)
    logged = json.loads((tmp_path / "events.jsonl").read_text())
    assert logged["environment_state_changed"] is True
    assert logged["bank_changes"]["scheduled_transactions"]["new_or_changed_records"] == new_records


def test_process_output_and_accepted_limit_exit_are_preserved(tmp_path, capsys):
    log = tmp_path / "child.log"
    code = driver.stream_command(
        [sys.executable, "-u", "-c", 'import sys; print("limit diagnostic"); raise SystemExit(3)'],
        log, timeout=5, accepted_exit_codes=(0, 3),
    )
    assert code == 3
    assert "limit diagnostic" in log.read_text()
    assert "limit diagnostic" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="exit 7"):
        driver.stream_command(
            [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failed.log", timeout=5,
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group deadline regression")
def test_process_deadline_also_kills_descendant_after_direct_parent_exits(tmp_path):
    marker = tmp_path / "grandchild.pid"
    child_code = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "print('grandchild holding stdout',flush=True); time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-u','-c',{child_code!r},{str(marker)!r}]); "
        "print('direct parent exits',flush=True)"
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="Process deadline"):
            driver.stream_command(
                [sys.executable, "-u", "-c", parent_code], tmp_path / "descendant.log", timeout=1,
            )
        assert time.monotonic() - started < 5
        assert marker.exists(), "The descendant must have started to exercise inherited stdout"
        assert "grandchild holding stdout" in (tmp_path / "descendant.log").read_text()
        # A killed orphan may briefly remain a zombie until PID 1 reaps it.
        proc_stat = Path(f"/proc/{int(marker.read_text())}/stat")
        if proc_stat.exists():
            assert proc_stat.read_text().split()[2] == "Z"
    finally:
        # Bound even a broken implementation's descendant lifetime in this test.
        if marker.exists():
            try:
                os.kill(int(marker.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("arguments", [
    ["--max-turns", "0"],
    ["--max-response-tokens", "2049"],
    ["--max-total-output-tokens", "32769"],
    ["--max-context-tokens", "24577"],
    ["--max-tool-calls", "17"],
    ["--episode-seconds", "0"],
    ["--generation-seconds", "601"],
    ["--worker-timeout-seconds", "599"],
    ["--worker-timeout-seconds", "7201"],
])
def test_invalid_cli_limits_fail_before_bootstrap_or_model(arguments, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid CLI arguments must not start setup, model loading, or generation")

    monkeypatch.setattr(driver, "bootstrap", forbidden)
    monkeypatch.setattr(driver, "run_model", forbidden)
    with pytest.raises(SystemExit) as error:
        driver.main(["--mode", "bank", *arguments])
    assert error.value.code == 2


def test_valid_bank_cli_forwards_limits_to_stub_launcher_only(monkeypatch):
    """Source-wrapper test; the stub does not produce a model experiment result."""
    received = []

    def launcher_stub(args):
        received.append(args)
        return 3

    monkeypatch.setattr(driver, "bootstrap", launcher_stub)
    code = driver.main([
        "--mode", "bank", "--max-turns", "12", "--max-response-tokens", "1024",
        "--max-total-output-tokens", "8192", "--max-context-tokens", "24576",
        "--episode-seconds", "900", "--generation-seconds", "300",
        "--worker-timeout-seconds", "1800", "--max-tool-calls", "4",
    ])
    assert code == 3 and len(received) == 1
    assert received[0].mode == "bank"
    assert received[0].max_response_tokens == 1024
    assert received[0].max_total_output_tokens == 8192
    assert received[0].max_context_tokens == 24576
    assert received[0].max_turns == 12

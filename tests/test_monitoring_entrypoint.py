"""Offline supervisor checks; no GPU, Qwen weights, or model API calls.

These tests inject subprocess outcomes into the real command-line
supervisor. They test its public exit/report contract, not model quality.
The separate configuration-only subprocess forbids heavy model imports
and Python socket connections.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

import OZLab_Qwen_Bank_Transfer as driver
import scripts.run_local_monitoring as entrypoint


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "configs/local-monitoring-pilot.json"


@pytest.fixture
def supervisor(monkeypatch, tmp_path):
    args = argparse.Namespace(
        config=PILOT,
        output_dir=tmp_path / "supervised-run",
        validate_config=False,
        offline=True,
        live_local=False,
        worker=False,
    )
    trace = []
    monkeypatch.setattr(entrypoint, "arguments", lambda: args)

    def inject_outcome(outcome):
        def authored_stream(command, log_path, *, timeout, accepted_exit_codes):
            trace.append({
                "command": command, "log_path": log_path,
                "timeout": timeout, "accepted_exit_codes": accepted_exit_codes,
            })
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(driver, "stream_command", authored_stream)

    return args, trace, inject_outcome


def test_supervisor_hard_timeout_is_limit_and_never_final_evaluation(supervisor):
    args, trace, inject = supervisor
    inject(TimeoutError("authored process-group deadline"))
    assert entrypoint.main() == 3
    failure = json.loads((args.output_dir / "process_failure.json").read_text())
    assert failure["status"] == "limit_reached"
    assert failure["termination_reason"] == "process_hard_timeout"
    assert failure["final_evaluation_available"] is False
    assert failure["model_api_calls"] == 0
    assert not (args.output_dir / "process_result.json").exists()
    assert not (args.output_dir / "run/summary.json").exists()
    config = json.loads(PILOT.read_text())
    assert trace[0]["timeout"] == config["run_seconds"] + 1800
    assert trace[0]["accepted_exit_codes"] == (0, 1, 3)
    assert trace[0]["command"][-1] == "--worker"


def test_supervisor_process_error_is_distinct_and_excludes_exception_secrets(
    supervisor, monkeypatch, capsys,
):
    args, _, inject = supervisor
    secret = "OFFLINE_SUPERVISOR_TEST_SECRET_NEVER_LOG"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    inject(RuntimeError(secret))
    assert entrypoint.main() == 1
    failure = json.loads((args.output_dir / "process_failure.json").read_text())
    assert failure["status"] == "process_error"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["model_api_calls"] == 0
    assert not (args.output_dir / "process_result.json").exists()
    output = capsys.readouterr().out
    files = "".join(path.read_text() for path in args.output_dir.glob("*.json"))
    assert secret not in output + files


@pytest.mark.parametrize("child_exit_code", [0, 1, 3])
def test_supervisor_preserves_child_exit_code_and_separates_task_success(supervisor, child_exit_code):
    args, trace, inject = supervisor
    inject(child_exit_code)
    assert entrypoint.main() == child_exit_code
    result = json.loads((args.output_dir / "process_result.json").read_text())
    assert result["exit_code"] == child_exit_code
    assert result["task_success_is_separate"] is True
    assert result["model_api_calls"] == 0
    assert not (args.output_dir / "process_failure.json").exists()
    assert "--offline" in trace[0]["command"]
    assert "--live-local" not in trace[0]["command"]
    assert trace[0]["log_path"] == args.output_dir / "worker.stdout.log"


def test_supervisor_constructs_live_local_worker_without_calling_it(supervisor):
    args, trace, inject = supervisor
    args.offline = False
    args.live_local = True
    inject(3)
    assert entrypoint.main() == 3
    assert "--live-local" in trace[0]["command"]
    assert "--offline" not in trace[0]["command"]
    process = json.loads((args.output_dir / "process.json").read_text())
    assert process["mode"] == "real_local_model"
    assert process["load_allowance_seconds"] == 1800
    # The injected stream function did not actually create a GPU worker.
    assert not (args.output_dir / "run").exists()


def test_supervisor_refuses_existing_output_and_preserves_work(supervisor):
    args, trace, inject = supervisor
    args.output_dir.mkdir()
    saved = args.output_dir / "existing-result.json"
    saved.write_text('{"preserve":true}\n')
    inject(0)
    with pytest.raises(FileExistsError):
        entrypoint.main()
    assert saved.read_text() == '{"preserve":true}\n'
    assert sorted(path.name for path in args.output_dir.iterdir()) == ["existing-result.json"]
    assert trace == []


def test_supervisor_requires_explicit_new_output_directory(supervisor):
    args, trace, inject = supervisor
    args.output_dir = None
    inject(0)
    with pytest.raises(ValueError, match="--output-dir is required"):
        entrypoint.main()
    assert trace == []


def test_configuration_only_cli_forbids_model_imports_and_connections(tmp_path):
    script = r'''
import builtins
import runpy
import socket
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'bitsandbytes', 'anthropic', 'openai'}:
        raise AssertionError('Configuration validation imported a model/provider package: ' + name)
    if name == 'local_monitoring.backend':
        raise AssertionError('Configuration validation imported the local inference backend')
    return original_import(name, *args, **kwargs)
def deny_connection(*args, **kwargs):
    raise AssertionError('Configuration validation attempted a network connection')
builtins.__import__ = guarded_import
socket.create_connection = deny_connection
socket.socket.connect = deny_connection
socket.socket.connect_ex = deny_connection
sys.argv = [sys.argv[1], '--config', sys.argv[2], '--validate-config']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "scripts/run_local_monitoring.py"), str(PILOT)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["validation"] == "configuration_only"
    assert report["model_weights_loaded"] is False
    assert report["model_api_calls"] == 0
    assert not list(tmp_path.iterdir())

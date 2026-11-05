"""Offline bootstrap/package tests; no Colab allocation, GPU, model or API.

Subprocess launch results below are authored doubles. Files and ZIP archives
are real temporary artifacts. A fake successful process is never evidence of
successful Qwen inference or paper reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

import OZLab_Qwen_Bank_Transfer as driver
from scripts import colab_monitoring_job as job
from scripts.smoke_bank_transfer import no_network


@pytest.fixture(autouse=True)
def no_connections():
    with no_network() as attempts:
        yield
        assert not attempts, "Offline bootstrap test attempted a Python connection"


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        work_dir=tmp_path / "source", env_path=tmp_path / "isolated-env",
        results_dir=tmp_path / "results", profile="pilot", python_version="3.12.16",
    )


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def config_files(root):
    for name in ("local-monitoring-pilot.json", "local-monitoring-paper-protocol.json"):
        write(root / "configs" / name, {"run_seconds": 10})


def live_pointer(args, directory=None, **overrides):
    directory = directory or args.results_dir / "live-offline-authored-test"
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {"output_dir": str(directory), "source_commit": job.SOURCE_COMMIT,
                "started_at": "OFFLINE_FIXTURE", "profile": "pilot", "mode": "live"}
    metadata.update(overrides)
    write(args.results_dir / "latest_live.json", metadata)
    return directory


def authored_report(**overrides):
    report = {
        "scope": "OFFLINE_AUTHORED_PACKAGE_FIXTURE_NOT_MODEL_EVIDENCE",
        "paper_reproduction": False, "execution_mode": "local_hf_real_model",
        "model_responses_mocked": False, "model_api_calls": 0,
        "status": "limit_reached", "termination_reason": "time_limit",
        "episodes_completed": 0, "episodes_requested": 2, "run_complete": False,
        "model_requests": 1, "model_responses": 1,
        "config": {"model_id": "Qwen/Qwen3-4B", "monitor_enabled": True},
        "usage": {"input_tokens": 100, "output_tokens": 10},
        "episodes": [],
    }
    report.update(overrides)
    return report


def test_job_definitions_execute_without_file_attribute_in_jupyter_namespace(monkeypatch):
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: pytest.fail("No process at cell definition"))
    source = Path(job.__file__).read_text()
    namespace = {"__name__": "colab_cell"}
    exec(compile(source, "colab-cell", "exec"), namespace)
    assert "__file__" not in namespace
    assert callable(namespace["main"]) and callable(namespace["run_stage"])
    assert namespace["SOURCE_COMMIT"] == job.SOURCE_COMMIT


def test_cli_prelude_can_precede_job_body_without_future_import_syntax_error(monkeypatch):
    # colab-cli executes a generated prelude before the transmitted file body.
    # A future import in that body would then be illegal despite normal import
    # and browser-cell checks passing. No real CLI session is started here.
    monkeypatch.setattr(job.sys, "argv", ["offline-test"])
    prelude = "import sys\nimport warnings\nsys.argv = ['colab_job']\n__name__ = 'colab_cell'\n"
    namespace = {}
    source = Path(job.__file__).read_text()
    exec(compile(prelude + source, "colab-cli-transmitted-body", "exec"), namespace)
    assert "__file__" not in namespace
    assert callable(namespace["main"])
    assert namespace["SOURCE_COMMIT"] == job.SOURCE_COMMIT


def test_existing_non_checkout_directory_is_preserved(args):
    args.work_dir.mkdir()
    marker = args.work_dir / "important.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        job.checkout(job.SOURCE_URL, args.work_dir, job.SOURCE_COMMIT)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("head,dirty", [
    ("different-commit", ""), (job.SOURCE_COMMIT, " M README.md\n"),
    (job.SOURCE_COMMIT, "?? user-notes.txt\n"),
])
def test_existing_wrong_or_dirty_checkout_is_never_reset(args, monkeypatch, head, dirty):
    (args.work_dir / ".git").mkdir(parents=True)
    marker = args.work_dir / "keep.txt"
    marker.write_text("preserve edits")
    replies = iter([head + "\n", dirty])
    monkeypatch.setattr(job.subprocess, "check_output", lambda *a, **k: next(replies))
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: pytest.fail("No reset or checkout allowed"))
    with pytest.raises(RuntimeError, match="보존"):
        job.checkout(job.SOURCE_URL, args.work_dir, job.SOURCE_COMMIT)
    assert marker.read_text() == "preserve edits"


def test_existing_clean_pinned_checkout_is_reused_without_mutation(args, monkeypatch):
    (args.work_dir / ".git").mkdir(parents=True)
    replies = iter([job.SOURCE_COMMIT + "\n", ""])
    monkeypatch.setattr(job.subprocess, "check_output", lambda *a, **k: next(replies))
    monkeypatch.setattr(job.subprocess, "run", lambda *a, **k: pytest.fail("No clone or checkout needed"))
    assert job.checkout(job.SOURCE_URL, args.work_dir, job.SOURCE_COMMIT) == args.work_dir.resolve()


def test_prepare_uses_both_immutable_source_and_upstream_commits(args, monkeypatch):
    calls = []
    upstream = {"repository": "https://example.invalid/SHADE.git", "directory": ".deps/SHADE-Arena",
                "commit": "pinned-upstream-offline-fixture"}
    write(args.work_dir / "configs/upstream.json", upstream)

    def checkout(url, directory, commit):
        calls.append((url, Path(directory), commit))
        return Path(directory).resolve()

    monkeypatch.setattr(job, "checkout", checkout)
    monkeypatch.setattr(job.sys, "path", list(job.sys.path))
    assert job.prepare(args) == args.work_dir.resolve()
    assert calls == [(job.SOURCE_URL, args.work_dir, job.SOURCE_COMMIT),
                     (upstream["repository"], args.work_dir / upstream["directory"], upstream["commit"])]


def test_run_stage_sets_new_pointer_before_failed_launch_and_cannot_package_old_success(args, monkeypatch):
    config_files(args.work_dir)
    old = live_pointer(args, args.results_dir / "live-previous-success")
    write(old / "run/summary.json", authored_report(status="completed", run_complete=True))
    new_directories = []

    def failed_launch(command, log_path, **kwargs):
        metadata = json.loads((args.results_dir / "latest_live.json").read_text())
        current = Path(metadata["output_dir"])
        assert current != old and metadata["mode"] == "live"
        assert metadata["source_commit"] == job.SOURCE_COMMIT
        assert command[command.index("--output-dir") + 1] == str(current)
        assert "--live-local" in command and "--offline" not in command
        current.mkdir()
        write(current / "process_failure.json", {"status": "process_error", "exception_type": "TimeoutError"})
        new_directories.append(current)
        raise TimeoutError("OFFLINE AUTHORED LAUNCH FAILURE")

    monkeypatch.setattr(driver, "stream_command", failed_launch)
    with pytest.raises(TimeoutError, match="OFFLINE AUTHORED"):
        job.run_stage(args, args.work_dir, offline=False)
    archive = job.package(args, args.work_dir)
    summary = json.loads((args.results_dir / "public_summary.json").read_text())
    assert summary["status"] == "process_error"
    assert summary["paper_reproduction"] is False
    assert "run_complete" not in summary or summary["run_complete"] is not True
    with zipfile.ZipFile(archive) as stream:
        assert "process_failure.json" in stream.namelist()
        assert "run/summary.json" not in stream.namelist()
        assert json.loads(stream.read("source.json"))["output_dir"] == str(new_directories[0])


@pytest.mark.parametrize("offline", [False, True])
def test_run_stage_forwards_exit3_and_keeps_offline_and_live_flags_separate(args, monkeypatch, offline):
    config_files(args.work_dir)
    observed = []

    def launch(command, log_path, **kwargs):
        observed.append((command, Path(log_path), kwargs))
        pointer = args.results_dir / ("latest_offline.json" if offline else "latest_live.json")
        metadata = json.loads(pointer.read_text())
        assert metadata["mode"] == ("offline" if offline else "live")
        return 3

    monkeypatch.setattr(driver, "stream_command", launch)
    assert job.run_stage(args, args.work_dir, offline=offline) == 3
    command, log_path, settings = observed[0]
    assert ("--offline" in command) is offline
    assert ("--live-local" in command) is not offline
    assert settings["accepted_exit_codes"] == (0, 1, 3)
    assert settings["timeout"] == 1870
    if offline:
        assert not (args.results_dir / "latest_live.json").exists()
        with pytest.raises(FileNotFoundError, match="offline"):
            job.package(args, args.work_dir)


@pytest.mark.parametrize("condition", ["stale_source", "outside_results", "offline_name", "missing_directory"])
def test_package_rejects_stale_or_wrong_result_pointer(args, condition):
    directory = live_pointer(args)
    metadata = json.loads((args.results_dir / "latest_live.json").read_text())
    if condition == "stale_source":
        metadata["source_commit"] = "old-code"
    elif condition == "outside_results":
        outside = args.results_dir.parent / "live-outside-results"
        outside.mkdir()
        metadata["output_dir"] = str(outside)
    elif condition == "offline_name":
        offline = args.results_dir / "offline-current"
        offline.mkdir()
        metadata["output_dir"] = str(offline)
    else:
        directory.rmdir()
    write(args.results_dir / "latest_live.json", metadata)
    with pytest.raises(ValueError, match="일치"):
        job.package(args, args.work_dir)
    assert not (args.results_dir / "private_results.zip").exists()


def test_private_zip_keeps_raw_state_while_public_summary_contains_only_allowlisted_values(args):
    directory = live_pointer(args)
    sentinel = "PRIVATE_ACCOUNT_MODEL_TEXT_SENTINEL_56d8"
    raw = authored_report(
        actor_history=[{"role": "assistant", "content": sentinel}],
        account={"iban": sentinel},
        episodes=[{"episode": 1, "status": "limit_reached", "termination_reason": "time_limit",
                   "monitor_scores": [79], "evaluation": {
                       "main_score": 0, "main_success": False, "upstream_side_task_success": True,
                       "main_evaluation": {"account": sentinel},
                       "bank_changes": {"scheduled": {"count": 1, "amount_usd": 5000,
                                                      "transactions": [{"recipient": sentinel}]}},
                       "task_text": {"threshold_met": True, "target_total_usd": 5000,
                                     "extra_private_field": sentinel},
                   }}],
    )
    write(directory / "run/summary.json", raw)
    write(directory / "run/state_after.json", {"bank_account": {"iban": sentinel}})
    (directory / "run/events.jsonl").write_text(json.dumps({"raw_model_text": sentinel}) + "\n")
    (directory / "run/partial.json.tmp").write_text(sentinel)
    outside = args.results_dir.parent / "outside-private.txt"
    outside.write_text(sentinel)
    (directory / "outside-link.txt").symlink_to(outside)
    archive = job.package(args, args.work_dir)
    summary_text = (args.results_dir / "public_summary.json").read_text()
    summary = json.loads(summary_text)
    assert sentinel not in summary_text
    assert "actor_history" not in summary_text and "transactions" not in summary_text
    assert summary["paper_reproduction"] is False
    assert summary["episodes"][0]["bank_changes"]["scheduled"] == {"count": 1, "amount_usd": 5000}
    assert summary["episodes"][0]["task_text"]["threshold_met"] is True
    with zipfile.ZipFile(archive) as stream:
        assert sentinel in stream.read("run/state_after.json").decode()
        assert sentinel in stream.read("run/events.jsonl").decode()
        assert "public_summary.json" in stream.namelist()
        assert "outside-link.txt" not in stream.namelist()
        assert "run/partial.json.tmp" not in stream.namelist()
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in stream.namelist())


def test_failed_before_engine_report_is_honest_and_still_archives_process_evidence(args):
    directory = live_pointer(args)
    write(directory / "process_failure.json", {"status": "process_error", "exception_type": "RuntimeError"})
    archive = job.package(args, args.work_dir)
    report = json.loads((args.results_dir / "public_summary.json").read_text())
    assert report["status"] == "process_error"
    assert report["paper_reproduction"] is False and report["model_api_calls"] == 0
    assert "model_responses" not in report and report["run_complete"] is False
    assert report["interrupted_episode_final_evaluation_available"] is False
    with zipfile.ZipFile(archive) as stream:
        assert "process_failure.json" in stream.namelist()


def test_hard_timeout_overrides_running_snapshot_without_fabricating_final_evaluation(args):
    directory = live_pointer(args)
    snapshot = authored_report(
        status="running", termination_reason="initialization_error", run_complete=False,
        episodes=[{"episode": 1, "status": "running", "termination_reason": "running",
                   "turns": 1, "evaluation": None}],
    )
    write(directory / "run/summary.json", snapshot)
    checkpoint = {"bank_account": {"balance": 123, "note": "PRIVATE_PARTIAL_STATE"}}
    write(directory / "run/episode-001/state_latest.json", checkpoint)
    write(directory / "process_failure.json", {
        "status": "limit_reached", "termination_reason": "process_hard_timeout",
        "exception_type": "TimeoutError",
    })
    archive = job.package(args, args.work_dir)
    report = json.loads((args.results_dir / "public_summary.json").read_text())
    assert report["status"] == "limit_reached"
    assert report["termination_reason"] == "process_hard_timeout"
    assert report["run_complete"] is False
    assert report["interrupted_episode_final_evaluation_available"] is False
    assert report["episodes"][0]["status"] == "incomplete"
    assert report["episodes"][0]["termination_reason"] == "process_hard_timeout"
    assert report["episodes"][0]["main_score"] is None
    assert report["episodes"][0]["upstream_side_task_success"] is None
    assert "PRIVATE_PARTIAL_STATE" not in json.dumps(report)
    with zipfile.ZipFile(archive) as stream:
        assert json.loads(stream.read("run/summary.json")) == snapshot
        assert json.loads(stream.read("run/episode-001/state_latest.json")) == checkpoint
        assert "run/episode-001/evaluation.json" not in stream.namelist()
        assert "process_failure.json" in stream.namelist()


@pytest.mark.parametrize("metadata_mode,execution_mode,mocked", [
    ("offline", "local_hf_real_model", False),
    ("live", "offline_scripted", True),
    ("live", "offline_scripted", False),
    ("live", "local_hf_real_model", True),
    ("live", "local_hf_real_model", None),
])
def test_live_package_refuses_offline_or_ambiguous_model_classification(
    args, metadata_mode, execution_mode, mocked,
):
    directory = live_pointer(args, mode=metadata_mode)
    write(directory / "run/summary.json", authored_report(
        execution_mode=execution_mode, model_responses_mocked=mocked,
    ))
    with pytest.raises(ValueError):
        job.package(args, args.work_dir)
    assert not (args.results_dir / "private_results.zip").exists()
    assert not (args.results_dir / "public_summary.json").exists()


def test_all_stage_packages_limit_result_and_preserves_model_exit_code3(args, monkeypatch):
    observed = []
    monkeypatch.setattr(job, "prepare", lambda parsed: args.work_dir)
    monkeypatch.setattr(job, "setup", lambda *a: observed.append("setup"))

    def run_stage(parsed, root, *, offline):
        observed.append("offline" if offline else "live")
        return 0 if offline else 3

    monkeypatch.setattr(job, "run_stage", run_stage)
    monkeypatch.setattr(job, "package", lambda *a: observed.append("package"))
    assert job.main(["--stage", "all"]) == 3
    assert observed == ["setup", "offline", "live", "package"]


def test_offline_stage_failure_stops_before_live_model_launch_or_publication(args, monkeypatch):
    observed = []
    monkeypatch.setattr(job, "prepare", lambda parsed: args.work_dir)
    monkeypatch.setattr(job, "setup", lambda *a: observed.append("setup"))

    def run_stage(parsed, root, *, offline):
        assert offline
        observed.append("offline")
        return 1

    monkeypatch.setattr(job, "run_stage", run_stage)
    monkeypatch.setattr(job, "package", lambda *a: pytest.fail("No real model result to package"))
    assert job.main(["--stage", "all"]) == 1
    assert observed == ["setup", "offline"]

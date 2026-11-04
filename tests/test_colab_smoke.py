"""Offline bootstrap safeguards; these tests do not allocate a Colab runtime."""

import json
import subprocess
from pathlib import Path

import pytest

from scripts import colab_smoke


def test_notebook_embeds_same_bootstrap_and_has_no_saved_outputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/bank_transfer_smoke_colab.ipynb").read_text())
    source = (root / "scripts/colab_smoke.py").read_text().split('\nif __name__ == "__main__":')[0]
    assert source in "".join(notebook["cells"][1]["source"])
    assert "accelerator" not in notebook["metadata"]  # CPU default, not GPU/TPU.
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            compile("".join(cell["source"]), "colab-cell", "exec")


def test_bootstrap_definitions_work_without_file_attribute():
    source = Path(colab_smoke.__file__).read_text().split('\nif __name__ == "__main__":')[0]
    namespace = {"__name__": "colab_cell"}
    exec(compile(source, "colab-cell", "exec"), namespace)
    assert "__file__" not in namespace and callable(namespace["run_smoke"])


def test_existing_non_checkout_is_preserved(tmp_path):
    marker = tmp_path / "important.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        colab_smoke.prepare_checkout(tmp_path)
    assert marker.read_text() == "keep"


def test_different_commit_is_not_reset(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "different-commit\n")
    monkeypatch.setattr(colab_smoke, "run_command", lambda *a, **k: pytest.fail("Must not reset or clone"))
    with pytest.raises(RuntimeError, match="commit"):
        colab_smoke.prepare_checkout(tmp_path)


def test_dirty_checkout_is_preserved(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    answers = iter([colab_smoke.SOURCE_COMMIT + "\n", " M README.md\n"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: next(answers))
    with pytest.raises(RuntimeError, match="변경"):
        colab_smoke.prepare_checkout(tmp_path)


def test_failed_smoke_records_failure_and_preserves_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 9, "failed\n", "error\n"))
    with pytest.raises(subprocess.CalledProcessError) as failure:
        colab_smoke.run_smoke(tmp_path)
    assert failure.value.returncode == 9
    context = json.loads((tmp_path / "reports/colab_smoke_context.json").read_text())
    assert context["status"] == "failed" and context["smoke_exit_code"] == 9


def test_zero_exit_with_wrong_scenario_count_does_not_pass(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "smoke_bank_transfer.json").write_text(json.dumps({
        "status": "passed", "cases": [], "model_api_calls": 0, "network_connection_attempts": 0,
    }))
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    with pytest.raises(RuntimeError, match="결과"):
        colab_smoke.run_smoke(tmp_path)
    assert json.loads((reports / "colab_smoke_context.json").read_text())["status"] == "failed"

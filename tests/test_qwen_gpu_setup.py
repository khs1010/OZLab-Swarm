"""Run the shell uv bootstrap with authored executable fixtures, without GPU/network.

These tests validate tool selection and bootstrap safeguards. They do not install
CUDA packages, download model weights, or provide evidence of real Qwen inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts/setup_qwen_gpu.sh"


def executable(path: Path, body: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)


def fake_uv(path: Path, version: str, label: str, trace: Path):
    executable(path, f"#!{sys.executable}\n" + (
        "import json,sys\nfrom pathlib import Path\n"
        f"path=Path({str(trace)!r})\n"
        f"with path.open('a') as stream: stream.write(json.dumps([{label!r},sys.argv[1:]])+'\\n')\n"
        "if sys.argv[1:] != ['--version']: raise SystemExit(97)\n"
        f"print({version!r})\n"
    ))


@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "source with spaces"
    script = root / "scripts/setup_qwen_gpu.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE, script)
    path = tmp_path / "fixture-bin"
    path.mkdir()
    # Do not inherit a system uv. Only the commands required for bootstrap are
    # available. Any accidental dependency/GPU action is recorded and fails.
    for name in ("dirname", "uname", "mkdir", "cat"):
        target = shutil.which(name)
        assert target
        (path / name).symlink_to(target)
    trace = tmp_path / "calls.jsonl"
    python = path / "bootstrap-python"
    executable(python, f"#!{sys.executable}\n" + r'''
import json,os,sys
from pathlib import Path
trace = Path(os.environ['FIXTURE_TRACE'])
with trace.open('a') as stream:
    json.dump(['bootstrap-python',sys.argv[1:]],stream)
    stream.write(chr(10))
args = sys.argv[1:]
if args[:3] != ['-m','pip','install']:
    raise SystemExit('Wrong Python/module entry point selected')
if '--require-hashes' not in args or '--upgrade' not in args:
    raise SystemExit('Repair must retain hashes and replace partial cached tools')
target = Path(args[args.index('--target')+1])
requirements = Path(args[args.index('-r')+1]).read_text()
if 'uv==0.12.23' not in requirements or '--hash=sha256:' not in requirements:
    raise SystemExit('Unpinned bootstrap')
if os.environ.get('FIXTURE_PIP_FAIL'):
    raise SystemExit('Authored fixture: pip unavailable')
binary = target/'bin/uv'
binary.parent.mkdir(parents=True,exist_ok=True)
version = os.environ.get('FIXTURE_INSTALLED_VERSION','uv 0.12.23 (fixture-build)')
body = chr(10).join([
    '#!'+sys.executable,
    'import json,sys',
    'from pathlib import Path',
    'with Path('+repr(str(trace))+').open("a") as stream: json.dump(["installed-cache",sys.argv[1:]],stream); stream.write(chr(10))',
    'if sys.argv[1:] != ["--version"]: raise SystemExit(97)',
    'print('+repr(version)+')',
    '',
])
binary.write_text(body)
binary.chmod(0o755)
''')
    executable(path / "nvidia-smi", f"#!{sys.executable}\n" +
               f"from pathlib import Path\nPath({str(tmp_path / 'GPU_WAS_CALLED')!r}).touch()\nraise SystemExit(98)\n")
    env = os.environ.copy()
    env.update(PATH=str(path), QWEN_BOOTSTRAP_PYTHON=str(python),
               QWEN_ENV_PATH=str(tmp_path / "preserved-gpu-env"),
               FIXTURE_TRACE=str(trace))
    return {"root": root, "script": script, "path": path, "trace": trace,
            "cache": root / ".cache/qwen-uv-0.12.23", "env": env,
            "gpu_marker": tmp_path / "GPU_WAS_CALLED"}


def run(sandbox):
    result = subprocess.run(["/bin/bash", str(sandbox["script"]), "--bootstrap-only"],
                            env=sandbox["env"], capture_output=True, text=True, timeout=10)
    assert not sandbox["gpu_marker"].exists()
    assert not Path(sandbox["env"]["QWEN_ENV_PATH"]).exists()
    rows = [json.loads(line) for line in sandbox["trace"].read_text().splitlines()] if sandbox["trace"].exists() else []
    return result, rows


@pytest.mark.parametrize("version", [
    "uv 0.12.23", "uv 0.12.23 (x86_64-unknown-linux-gnu)",
    "uv 0.12.23 (abc123 2026-11-05)",
])
def test_bootstrap_accepts_exact_pinned_release_with_optional_build_info(sandbox, version):
    fake_uv(sandbox["path"] / "uv", version, "path-pinned", sandbox["trace"])
    result, rows = run(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(row[0] == "path-pinned" for row in rows)
    assert not sandbox["cache"].exists(), "A working PATH uv must not be reinstalled"


@pytest.mark.parametrize("version", ["uv 0.12.230", "uv 0.12.22", "not-uv 0.12.23"])
def test_wrong_path_version_uses_exact_cached_executable_without_python_module(sandbox, version):
    fake_uv(sandbox["path"] / "uv", version, "wrong-global", sandbox["trace"])
    fake_uv(sandbox["cache"] / "bin/uv", "uv 0.12.23 (cached-build)", "pinned-cache", sandbox["trace"])
    result, rows = run(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(row[0] == "pinned-cache" for row in rows)
    assert not any(row[0] == "bootstrap-python" for row in rows), "Do not route cache execution through python -m uv"


def test_partial_cached_module_directory_is_repaired_when_binary_is_missing(sandbox):
    (sandbox["cache"] / "uv").mkdir(parents=True)
    marker = sandbox["cache"] / "uv/previous_module.txt"
    marker.write_text("partial cache, not a usable executable")
    result, rows = run(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(row[0] == "bootstrap-python" for row in rows)
    assert any(row[0] == "installed-cache" for row in rows)
    assert marker.read_text() == "partial cache, not a usable executable"


def test_wrong_cached_binary_is_repaired_with_hash_pinned_pip_install(sandbox):
    fake_uv(sandbox["cache"] / "bin/uv", "uv 0.12.230", "wrong-cache", sandbox["trace"])
    result, rows = run(sandbox)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any(row[0] == "bootstrap-python" for row in rows)
    assert any(row[0] == "installed-cache" for row in rows)


def test_repair_that_still_has_wrong_version_fails_with_observed_diagnostic(sandbox):
    sandbox["env"]["FIXTURE_INSTALLED_VERSION"] = "uv 0.12.230 (authored-wrong-build)"
    result, rows = run(sandbox)
    assert result.returncode != 0
    assert "0.12.230" in result.stdout + result.stderr
    assert "0.12.23" in result.stdout + result.stderr
    assert any(row[0] == "bootstrap-python" for row in rows)


def test_bootstrap_only_does_not_require_or_modify_gpu_environment(sandbox):
    fake_uv(sandbox["path"] / "uv", "uv 0.12.23 (linux)", "path-pinned", sandbox["trace"])
    environment = Path(sandbox["env"]["QWEN_ENV_PATH"])
    environment.mkdir()
    marker = environment / "existing.txt"
    marker.write_text("preserve this environment")
    result = subprocess.run(["/bin/bash", str(sandbox["script"]), "--bootstrap-only"],
                            env=sandbox["env"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text() == "preserve this environment"
    assert not sandbox["gpu_marker"].exists()
    rows = [json.loads(line) for line in sandbox["trace"].read_text().splitlines()]
    assert all(row[1] == ["--version"] for row in rows), "No CUDA/dependency/model command may run"

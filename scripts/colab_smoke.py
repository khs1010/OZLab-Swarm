"""Self-contained bootstrap for Colab CPU (web notebook or colab run).

Only standard-library imports: this also works when the CLI sends the code to a
Jupyter cell without __file__ or the rest of the repository. No model API calls.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCE_URL = "https://github.com/khs1010/OZLab-Swarm.git"
SOURCE_COMMIT = "9917211719190f7e871bd7d246eeb7400165d1ff"
PYTHON_VERSION = "3.12.14"
UV_VERSION = "0.12.23"


def run_command(arguments, *, cwd=None, env=None):
    print("실행:", " ".join(map(str, arguments)), flush=True)
    return subprocess.run(list(map(str, arguments)), cwd=cwd, env=env, check=True)


def prepare_checkout(root):
    root = Path(root).resolve()
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        run_command(["git", "clone", "--no-checkout", SOURCE_URL, root])
        run_command(["git", "-C", root, "checkout", "--detach", SOURCE_COMMIT])
    elif not (root / ".git").exists():
        raise FileExistsError("작업 경로가 이미 있습니다. 새 경로를 지정하세요.")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != SOURCE_COMMIT:
        raise RuntimeError("기존 checkout의 commit이 다릅니다. 보존을 위해 중단합니다.")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"], text=True
    )
    if dirty:
        raise RuntimeError("기존 checkout에 변경 파일이 있습니다. 보존을 위해 중단합니다.")
    return root


def install_environment(root, uv_binary=None):
    root = Path(root).resolve()
    if uv_binary is None:
        target = root / ".cache/colab-tools"
        run_command([
            sys.executable, "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
            "--upgrade", "--target", target, f"uv=={UV_VERSION}",
        ])
        uv_binary = target / "bin/uv"
    uv_binary = Path(shutil.which(str(uv_binary)) or uv_binary).resolve()
    version = subprocess.check_output([str(uv_binary), "--version"], text=True).split()
    if version[:2] != ["uv", UV_VERSION]:
        raise RuntimeError(f"uv {UV_VERSION}가 필요합니다.")
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(root / ".cache/uv")
    env["UV_PYTHON_INSTALL_DIR"] = str(root / ".cache/colab-python")
    # Avoid changing the notebook kernel's Python or global package versions.
    run_command([uv_binary, "python", "install", "--no-bin", PYTHON_VERSION], env=env)
    python = subprocess.check_output(
        [str(uv_binary), "python", "find", "--managed-python", PYTHON_VERSION], env=env, text=True
    ).strip()
    env["BANK_PYTHON"] = python
    env["PATH"] = str(uv_binary.parent) + os.pathsep + env.get("PATH", "")
    run_command(["bash", root / "scripts/setup.sh"], cwd=root, env=env)
    return root / ".venv/bin/python"


def run_smoke(root):
    root = Path(root).resolve()
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    context = {
        "scope": "environment_smoke_only_not_model_experiment_or_paper_reproduction",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": SOURCE_COMMIT,
        "host_platform": platform.platform(),
        "bootstrap_python": platform.python_version(),
        "test_python": PYTHON_VERSION,
        "colab_detected": bool(os.environ.get("COLAB_RELEASE_TAG") or "google.colab" in sys.modules),
        "status": "failed",
    }
    context_path = reports / "colab_smoke_context.json"
    context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    command = [str(root / ".venv/bin/python"), str(root / "scripts/smoke_bank_transfer.py")]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    (reports / "colab_smoke_stdout.txt").write_text(result.stdout + result.stderr)
    print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    context["smoke_exit_code"] = result.returncode
    try:
        result.check_returncode()
        smoke = json.loads((reports / "smoke_bank_transfer.json").read_text())
        if (smoke["status"] != "passed" or len(smoke["cases"]) != 9
                or smoke["model_api_calls"] != 0 or smoke["network_connection_attempts"] != 0):
            raise RuntimeError("예상한 smoke 결과와 다릅니다. 보고서를 확인하세요.")
        context["status"] = "passed"
        context["passed_scenarios"] = len(smoke["cases"])
        context["upstream_commit"] = smoke["upstream_commit"]
    finally:
        context["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        context_path.write_text(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    return smoke


def archive_results(root):
    reports = Path(root).resolve() / "reports"
    archive = reports / "colab_smoke_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name in ("smoke_bank_transfer.json", "colab_smoke_context.json", "colab_smoke_stdout.txt"):
            output.write(reports / name, arcname=name)
    print("다운로드할 결과:", archive, flush=True)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/OZLab-Swarm"))
    parser.add_argument("--uv", type=Path, help="이미 설치한 uv 0.12.23 경로; 생략하면 격리 설치")
    args = parser.parse_args(argv)
    root = prepare_checkout(args.work_dir)
    install_environment(root, args.uv)
    run_smoke(root)
    archive_results(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

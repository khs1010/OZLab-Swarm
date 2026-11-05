"""Standalone Colab job: works as a web cell or `colab run` file body.

No __file__, repository imports, account login, secrets or paid model APIs.
Only --stage live/all loads a local model; setup downloads public packages.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

SOURCE_URL = "https://github.com/khs1010/OZLab-Swarm.git"
SOURCE_COMMIT = "1e4441b072673cc239c2205e407b06fc32a24fd7"
DEFAULT_ROOT = "/content/ozlab-cot-source-" + SOURCE_COMMIT[:8]
DEFAULT_RESULTS = "/content/ozlab-cot-results"
DEFAULT_ENV = "/content/ozlab-cot-env"


def checkout(url, directory, commit):
    directory = Path(directory)
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", url, str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "checkout", "--detach", commit], check=True)
    if not (directory / ".git").exists():
        raise FileExistsError("기존 폴더를 보존합니다. 새 경로를 지정하세요.")
    head = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True)
    if head != commit or dirty:
        raise RuntimeError("기존 checkout이 고정 버전과 다릅니다. 변경을 보존하고 중단합니다.")
    return directory.resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(args):
    root = checkout(SOURCE_URL, args.work_dir, SOURCE_COMMIT)
    upstream = json.loads((root / "configs/upstream.json").read_text())
    checkout(upstream["repository"], root / upstream["directory"], upstream["commit"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def setup(args, root):
    from OZLab_Qwen_Bank_Transfer import stream_command
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    # Isolated environment; never replace the Colab kernel's torch installation.
    # Use Popen env through a small bounded Python launcher rather than changing
    # process-wide HOME, PATH, kernel packages or credentials.
    child_code = (
        "import os,subprocess,sys; "
        "env=os.environ.copy(); "
        "env.update(QWEN_ENV_PATH=sys.argv[1],QWEN_PYTHON_VERSION=sys.argv[2]); "
        "raise SystemExit(subprocess.call(['bash',sys.argv[3]],env=env))"
    )
    stream_command([sys.executable, "-u", "-c", child_code, str(args.env_path), args.python_version,
                    str(root / "scripts/setup_qwen_gpu.sh")], results / "setup.stdout.log", timeout=1800)
    stream_command([str(args.env_path / "bin/python"), str(root / "scripts/smoke_bank_transfer.py"),
                    "--output", str(results / "smoke.json")], results / "smoke.stdout.log", timeout=180)
    print("환경 준비·실제 SHADE smoke 통과. 아직 실제 Qwen 감시 실험을 실행하지 않았습니다.", flush=True)


def run_stage(args, root, *, offline):
    from OZLab_Qwen_Bank_Transfer import stream_command
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.results_dir / (("offline-" if offline else "live-") + timestamp)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    kind = "offline" if offline else "live"
    # Update BEFORE launch; packaging after a failure cannot return an old run.
    write_json(args.results_dir / f"latest_{kind}.json", {
        "output_dir": str(directory), "source_commit": SOURCE_COMMIT, "started_at": timestamp,
        "profile": args.profile, "mode": kind})
    config_name = "local-monitoring-pilot.json" if args.profile == "pilot" else "local-monitoring-paper-protocol.json"
    config = root / "configs" / config_name
    # The child supervisor handles hard process-group termination. The outer
    # allowance is longer so it can save its timeout report before this expires.
    settings = json.loads(config.read_text())
    code = stream_command([
        str(args.env_path / "bin/python"), "-u", str(root / "scripts/run_local_monitoring.py"),
        "--config", str(config), "--output-dir", str(directory),
        "--offline" if offline else "--live-local",
    ], args.results_dir / f"{kind}-{timestamp}.stdout.log",
        timeout=settings["run_seconds"] + 1860, accepted_exit_codes=(0, 1, 3))
    summary = directory / "run/summary.json"
    if summary.exists():
        report = json.loads(summary.read_text())
        print(f"{kind}: status={report['status']} reason={report['termination_reason']} "
              f"ended_episodes={report['episodes_completed']}/{report['episodes_requested']}", flush=True)
        for episode in report["episodes"]:
            evaluation = episode.get("evaluation") or {}
            print(f"  episode {episode['episode']}: {episode['termination_reason']}, "
                  f"main={evaluation.get('main_score')}, side={evaluation.get('upstream_side_task_success')}", flush=True)
    print(f"종료 코드 {code}. 코드 0도 정상 과제 성공이나 논문 재현을 뜻하지 않습니다.", flush=True)
    return code


def package(args, root):
    from local_monitoring.summary import public_summary
    pointer = args.results_dir / "latest_live.json"
    if not pointer.exists():
        raise FileNotFoundError("실제 모델 실행 기록이 없습니다. offline 결과를 실제 실행으로 포장하지 않습니다.")
    metadata = json.loads(pointer.read_text())
    directory = Path(metadata["output_dir"]).resolve()
    if (metadata.get("source_commit") != SOURCE_COMMIT or metadata.get("mode") != "live"
            or directory.parent != args.results_dir.resolve()
            or not directory.name.startswith("live-") or not directory.is_dir()):
        raise ValueError("현재 소스·결과 경로가 일치하지 않습니다.")
    raw_summary = directory / "run/summary.json"
    failure_path = directory / "process_failure.json"
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    if raw_summary.exists():
        report = json.loads(raw_summary.read_text())
        if (report.get("execution_mode") != "local_hf_real_model"
                or report.get("model_responses_mocked") is not False):
            raise ValueError("실제 모델 기록과 모의 응답 기록이 일치하지 않습니다.")
        summary = public_summary(report)
    else:
        summary = {"status": "failed_before_engine_report", "paper_reproduction": False,
                   "model_api_calls": 0, "source_commit": SOURCE_COMMIT,
                   "note": "Inspect private process/worker logs; no model result is claimed."}
    if failure:
        summary.update(status=failure.get("status", "process_error"),
                       termination_reason=failure.get("termination_reason", "process_error"),
                       run_complete=False, interrupted_episode_final_evaluation_available=False)
        for episode in summary.get("episodes", []):
            if episode.get("status") == "running":
                episode.update(status="incomplete", termination_reason=summary["termination_reason"])
    write_json(args.results_dir / "public_summary.json", summary)
    archive = args.results_dir / "private_results.zip"
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for item in sorted(directory.rglob("*")):
            if item.is_file() and not item.is_symlink() and item.suffix != ".tmp":
                stream.write(item, arcname=str(item.relative_to(directory)))
        stream.writestr("source.json", json.dumps(metadata, indent=2))
        for name in ("public_summary.json", "setup.stdout.log", "smoke.stdout.log"):
            item = args.results_dir / name
            if item.is_file():
                stream.write(item, arcname=name)
    temporary.replace(archive)
    print(f"요약: {args.results_dir / 'public_summary.json'}", flush=True)
    print(f"개인 보관 ZIP: {archive} — 원본 모델 출력·모의 계좌 상태 포함. GitHub에 공개하지 마세요.", flush=True)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description="Colab local Qwen monitoring pilot; no paid model API.")
    parser.add_argument("--stage", choices=("setup", "offline", "live", "package", "all"), default="all")
    parser.add_argument("--profile", choices=("pilot", "paper_protocol"), default="pilot")
    parser.add_argument("--work-dir", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--env-path", type=Path, default=Path(DEFAULT_ENV))
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS))
    parser.add_argument("--python-version", default="3.12.14")
    args = parser.parse_args(argv)
    root = prepare(args)
    code = 0
    if args.stage in ("setup", "all"):
        setup(args, root)
    if args.stage in ("offline", "all"):
        code = run_stage(args, root, offline=True)
        if code:
            return code
    if args.stage in ("live", "all"):
        code = run_stage(args, root, offline=False)
    if args.stage in ("package", "all"):
        package(args, root)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

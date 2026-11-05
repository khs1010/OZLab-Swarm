#!/usr/bin/env python3
"""Bounded local Qwen actor/monitor run; never calls a paid model API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/local-monitoring-pilot.json")
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-config", action="store_true", help="No weights, GPU or episode")
    modes.add_argument("--offline", action="store_true", help="Authored responses; actual SHADE tools")
    modes.add_argument("--live-local", action="store_true", help="Download/load local weights; GPU compute")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


class OfflineGenerator:
    """Authored integration fixture; no learned inference or model performance."""
    is_mock = True
    execution_mode = "offline_scripted"

    def manifest(self):
        return {"backend": "authored_offline_fixture", "model_api_calls": 0,
                "real_model_inference": False}

    def generate(self, messages, *, role, **kwargs):
        from local_monitoring.parsing import Completion
        if role == "monitor":
            text = "Authored fixture only. <verdict>10</verdict>"
        else:
            text = ('<think>This is an authored offline integration fixture.</think>\n'
                    '<tool_call>{"name":"end_task","arguments":{}}</tool_call>')
        return Completion(text=text, input_tokens=1, output_tokens=25,
                          finished=True, elapsed_seconds=0.0)


def main():
    args = arguments()
    from local_monitoring.config import Config
    from dataclasses import asdict
    config = Config.from_dict(json.loads(args.config.read_text()))
    if args.validate_config:
        print(json.dumps({"validation": "configuration_only", "config": asdict(config),
                          "model_api_calls": 0, "model_weights_loaded": False}, indent=2))
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required; use a new result directory")
    directory = args.output_dir.resolve()
    if not args.worker:
        # Hard process-group timeout also bounds loading/prefill if a CUDA call
        # cannot honor Transformers' soft generation deadline. No hidden retry.
        from OZLab_Qwen_Bank_Transfer import stream_command, save, now
        directory.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--config", str(args.config.resolve()), "--output-dir", str(directory / "run"),
                   "--offline" if args.offline else "--live-local", "--worker"]
        hard_seconds = config.run_seconds + 1800  # declared download/load allowance
        save(directory / "process.json", {"started_at": now(), "hard_seconds": hard_seconds,
             "load_allowance_seconds": 1800, "model_api_calls": 0,
             "mode": "offline_authored" if args.offline else "real_local_model"})
        started = time.monotonic()
        try:
            code = stream_command(command, directory / "worker.stdout.log",
                                  timeout=hard_seconds, accepted_exit_codes=(0, 1, 3))
        except TimeoutError:
            save(directory / "process_failure.json", {
                "status": "limit_reached", "termination_reason": "process_hard_timeout",
                "elapsed_seconds": time.monotonic() - started, "model_api_calls": 0,
                "final_evaluation_available": False,
                "note": "Worker killed; state_latest is a checkpoint, not a finalized evaluation."})
            print(f"Hard time limit reached; inspect {directory}. Final evaluation is unavailable.")
            return 3
        except BaseException as error:
            save(directory / "process_failure.json", {
                "status": "process_error", "exception_type": type(error).__name__,
                "elapsed_seconds": time.monotonic() - started, "model_api_calls": 0})
            print(f"Process failed: {type(error).__name__}; inspect {directory}", flush=True)
            return 1
        save(directory / "process_result.json", {"exit_code": code,
             "elapsed_seconds": time.monotonic() - started,
             "task_success_is_separate": True, "model_api_calls": 0})
        print(f"Results: {directory} | worker exit={code}; inspect run/summary.json for outcomes.")
        return int(code)
    from local_monitoring.engine import run
    generator = OfflineGenerator()
    if args.live_local:
        from local_monitoring.backend import LocalQwenBackend
        generator = LocalQwenBackend(config)
    # Downloads precede this boundary. Actor/monitor inference and SHADE tool
    # execution are local; even existing API-key environment variables aren't used.
    report = run(config, generator, directory)
    print(f"실행 상태: {report['status']} / {report['termination_reason']}")
    for episode in report.get("episodes", []):
        evaluation = episode.get("evaluation") or {}
        print(f"에피소드 {episode['episode']}: {episode['termination_reason']}, "
              f"정상 점수={evaluation.get('main_score')}, "
              f"기존 위반 판정={evaluation.get('upstream_side_task_success')}")
    print("상세 기록은 run/summary.json에 저장했습니다. 실행 종료와 과제 성공은 별개입니다.")
    reason = report.get("termination_reason", report.get("status", "unknown"))
    if report.get("run_complete") or report.get("status") == "completed":
        return 0
    if any(word in str(reason) for word in ("limit", "max_turn", "time", "truncated")):
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

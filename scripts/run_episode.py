"""Run one SHADE-Arena connection baseline; real API requires --live."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from scripts.smoke_bank_transfer import no_network
from single_agent.config import load_config
from single_agent.mock import connection_demo
from single_agent.runner import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--live", action="store_true", help="Make real, potentially billable API calls")
    modes.add_argument("--mock", action="store_true", help="Scripted offline flow; not a real model experiment")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.output_dir is not None:
            config.output_dir = args.output_dir
        if args.mock:
            with no_network() as attempts:
                summary = asyncio.run(run_episode(config, provider=connection_demo(), execution_mode="offline_mock"))
                if attempts:
                    raise RuntimeError("Offline execution attempted network access")
        else:
            summary = asyncio.run(run_episode(config, execution_mode="live"))
    except ValidationError as error:
        # Do not print input values: a misconfigured YAML could contain credentials.
        print("설정 오류: " + ", ".join(".".join(map(str, e["loc"])) for e in error.errors()), file=sys.stderr)
        return 2
    except Exception as error:
        print(f"실행 준비 실패: {type(error).__name__}. 기존 결과 폴더·설정·설치 상태를 확인하세요.", file=sys.stderr)
        return 2
    print(f"mode={summary['execution_mode']}, status={summary['status']}, reason={summary['termination_reason']}")
    print(f"기록: {config.output_dir}/summary.json")
    return {"completed": 0, "limit_reached": 3, "error": 1}[summary["status"]]


if __name__ == "__main__":
    raise SystemExit(main())

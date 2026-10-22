from __future__ import annotations

import argparse
import json

from uav_flight_memory.experiments.runner import (
    analyze_mapping,
    build_windows,
    download_selected,
    run_experiment_grid,
    select_flights,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="UAV Flight Memory pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("analyze-mapping")

    select_parser = sub.add_parser("select-flights")
    select_parser.add_argument("--seed", type=int, default=42)
    select_parser.add_argument("--normal-limit", type=int, default=120)
    select_parser.add_argument("--external-position-limit", type=int, default=160)

    download_parser = sub.add_parser("download-selected")
    download_parser.add_argument("--limit", type=int, default=None)

    windows_parser = sub.add_parser("build-windows")
    windows_parser.add_argument("--window-seconds", type=int, default=20)
    windows_parser.add_argument("--min-anomaly-coverage", type=float, default=0.10)

    experiments_parser = sub.add_parser("run-experiments")
    experiments_parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "analyze-mapping":
        payload = analyze_mapping()
    elif args.command == "select-flights":
        payload = select_flights(
            seed=args.seed,
            normal_limit=args.normal_limit,
            external_position_limit=args.external_position_limit,
        ).to_dict(orient="records")
    elif args.command == "download-selected":
        payload = download_selected(limit=args.limit).to_dict(orient="records")
    elif args.command == "build-windows":
        payload = {
            "windows": len(
                build_windows(
                    window_seconds=args.window_seconds,
                    min_anomaly_coverage=args.min_anomaly_coverage,
                )
            )
        }
    elif args.command == "run-experiments":
        payload = run_experiment_grid(seed=args.seed)
    else:
        raise ValueError(args.command)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


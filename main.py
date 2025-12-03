"""
AAVA Tool - Harmfulness Classification Training Pipeline

This script provides a simple interface for training and evaluating
models for classifying text content by harmfulness level.

Usage:
    python main.py                    # Run full pipeline
    python main.py --pipeline train   # Only data processing + training
    python main.py --pipeline evaluate # Only evaluation
    kedro run                          # Run via Kedro CLI
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def main():
    parser = argparse.ArgumentParser(
        description="AAVA Harmfulness Classification Training Pipeline"
    )
    parser.add_argument(
        "--pipeline",
        "-p",
        type=str,
        default="__default__",
        choices=["__default__", "data_processing", "model_training", "model_evaluation", "train", "evaluate"],
        help="Pipeline to run (default: full pipeline)",
    )
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        default="base",
        help="Kedro environment to use",
    )
    
    args = parser.parse_args()
    
    project_path = Path(__file__).parent
    bootstrap_project(project_path)
    
    print("=" * 60)
    print("AAVA Harmfulness Classification Pipeline")
    print("=" * 60)
    print(f"Running pipeline: {args.pipeline}")
    print(f"Environment: {args.env}")
    print("-" * 60)
    
    with KedroSession.create(project_path=project_path, env=args.env) as session:
        session.run(pipeline_name=args.pipeline)
    
    print("-" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

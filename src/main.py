import argparse
import yaml
import os
import sys

# Add src to path to allow relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import train
from evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="FLASH: Fast, Safe, and Architecture-Agnostic TTA")

    # Configuration file choice
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--smoke-test", action="store_true", help="Run with the smoke test configuration.")
    config_group.add_argument(
        "--full-experiment", action="store_true", help="Run with the full experiment configuration."
    )

    # Execution mode (default changes based on config type)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
        help="Execution mode: train the hyper-network or evaluate it.",
    )

    # Evaluation-specific arguments
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["exp1", "exp2", "exp3"],
        help="Specify which experiment to run in evaluation mode.",
    )

    args = parser.parse_args()

    # Set default mode based on config type
    if not args.mode:
        if args.smoke_test:
            args.mode = "train"  # Smoke tests default to training
        else:
            args.mode = "train"  # Full experiments also default to training first
    
    # Set default experiment for full experiments in evaluate mode
    if args.mode == "evaluate" and not args.experiment:
        if args.full_experiment:
            args.experiment = "exp1"  # Default to exp1 for full experiments
        else:
            parser.error("--experiment is required when --mode is 'evaluate'")

    # Determine config file path
    config_dir = os.path.join(os.path.dirname(__file__), "..", "config")
    if args.smoke_test:
        config_path = os.path.join(config_dir, "smoke_test.yaml")
    else:
        config_path = os.path.join(config_dir, "full_experiment.yaml")

    # Load configuration from YAML file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}", file=sys.stderr)
        sys.exit(1)

    # Update config with command-line args for convenience
    config["mode"] = args.mode
    if args.experiment:
        config.setdefault("evaluate", {})["experiment"] = args.experiment

    # Set up output directories from config
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    config["project"]["output_dir"] = os.path.join(base_dir, config["project"]["output_dir"])
    config["project"]["data_dir"] = os.path.join(base_dir, config["project"]["data_dir"])
    os.makedirs(config["project"]["output_dir"], exist_ok=True)
    os.makedirs(config["project"]["data_dir"], exist_ok=True)

    # Dispatch to the appropriate function
    if args.mode == "train":
        print("--- Starting Training Phase ---")
        train(config)
    else:  # evaluate
        print(f"--- Starting Evaluation Phase for {args.experiment} ---")
        evaluate(config)


if __name__ == "__main__":
    main()

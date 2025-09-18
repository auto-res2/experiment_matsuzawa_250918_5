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

    # Execution mode (default to evaluate for convenience)
    parser.add_argument(
        "--mode",
        type=str,
        default="evaluate",
        choices=["train", "evaluate"],
        help="Execution mode: train the hyper-network or evaluate it (default: evaluate).",
    )

    # Evaluation-specific arguments (optional now)
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["exp1", "exp2", "exp3", "all"],
        help="Specify which experiment to run in evaluation mode. If omitted, all experiments are run.",
    )

    args = parser.parse_args()

    # Determine config file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Go up one level from src to project root
    config_dir = os.path.join(base_dir, "config")
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
    else:
        # Default: run ALL experiments
        config.setdefault("evaluate", {})["experiment"] = "all"

    # Set up output directories from config
    # base_dir already calculated above
    config["project"]["output_dir"] = os.path.join(base_dir, config["project"]["output_dir"])
    config["project"]["data_dir"] = os.path.join(base_dir, config["project"]["data_dir"])
    os.makedirs(config["project"]["output_dir"], exist_ok=True)
    os.makedirs(config["project"]["data_dir"], exist_ok=True)

    # Dispatch to the appropriate function
    if args.mode == "train":
        print("--- Starting Training Phase ---")
        train(config)
    else:  # evaluate
        print(f"--- Starting Evaluation Phase ({config['evaluate']['experiment']}) ---")
        evaluate(config)


if __name__ == "__main__":
    main()

import argparse
import yaml
import sys
import os
import torch
import numpy as np
import random
import logging

# It's crucial to use relative imports for modules within the same package
from . import preprocess
from . import train
from . import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the CERTIFIED-OFFICER experimental pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help="Run a quick smoke test with a minimal configuration.")
    group.add_argument('--full-experiment', action='store_true', help="Run the full experiment with the complete configuration.")
    return parser.parse_args()

def load_config(config_path):
    logging.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}. Aborting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file: {e}. Aborting.")
        sys.exit(1)

def setup_environment(config):
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # These can slow down training, but are good for reproducibility
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logging.info(f"Environment set up with random seed: {seed}")

def main():
    args = parse_arguments()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else: # args.full_experiment
        config_path = 'config/full_experiment.yaml'

    config = load_config(config_path)
    config['smoke_test'] = args.smoke_test # Add flag to config for downstream use
    setup_environment(config)

    try:
        # --- Phase 1: Data Preprocessing ---
        logging.info("====== Starting Phase 1: Data Preprocessing ======")
        preprocess.run_preprocessing(config)
        logging.info("====== Finished Phase 1: Data Preprocessing ======")

        # --- Phase 2: Model Training ---
        logging.info("\n====== Starting Phase 2: Model Training ======")
        train.run_training(config)
        logging.info("====== Finished Phase 2: Model Training ======")

        # --- Phase 3: Evaluation ---
        logging.info("\n====== Starting Phase 3: Evaluation ======")
        evaluate.run_evaluation(config)
        logging.info("====== Finished Phase 3: Evaluation ======")

        logging.info("\nExperimental pipeline completed successfully!")

    except Exception as e:
        logging.critical(f"An unhandled exception occurred during the pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

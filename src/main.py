import argparse
import yaml
import os
import sys
import torch

from .preprocess import prepare_data
from .train import run_training
from .evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Run S3-GAT experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')

    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
    else:
        config_path = 'config/full_experiment.yaml'

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    # Set seed for reproducibility
    torch.manual_seed(config['global_settings']['seeds'][0])
    
    # --- 1. Preprocessing ---
    print("--- Stage 1: Data Preprocessing ---")
    all_datasets = prepare_data(config)
    print("--- Data Preprocessing Complete ---")

    # --- 2. Training & 3. Evaluation ---
    # Loop through experiments defined in the config
    for exp_key in config:
        if exp_key.startswith('experiment_'):
            exp_id = int(exp_key.split('_')[1])
            print(f'\n>>> Starting Experiment {exp_id}: {config[exp_key]["description"]} <<<')
            
            # Get the data for this experiment (using the first dataset specified)
            dataset_name = config[exp_key]['datasets'][0]
            data_snapshots = all_datasets[dataset_name]

            # --- Training Stage ---
            print(f'\n--- Stage 2: Training for Experiment {exp_id} ---')
            try:
                run_training(config, exp_id, data_snapshots)
                print(f'--- Training for Experiment {exp_id} Complete ---')
            except Exception as e:
                print(f"ERROR during training for Experiment {exp_id}: {e}", file=sys.stderr)
                # Decide whether to continue or exit
                # For now, we'll try to continue to the next experiment
                continue

            # --- Evaluation Stage ---
            print(f'\n--- Stage 3: Evaluation for Experiment {exp_id} ---')
            try:
                run_evaluation(config, exp_id, data_snapshots)
                print(f'--- Evaluation for Experiment {exp_id} Complete ---')
            except Exception as e:
                print(f"ERROR during evaluation for Experiment {exp_id}: {e}", file=sys.stderr)
                continue
    
    print("\nAll experiments finished.")

if __name__ == '__main__':
    main()

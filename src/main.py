import argparse
import yaml
import sys
import logging
from . import preprocess, train, evaluate
import torch
import numpy as np
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    p = argparse.ArgumentParser("CERTIFIED-OFFICER pipeline")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--smoke-test', action='store_true')
    g.add_argument('--full-experiment', action='store_true')
    return p.parse_args()


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    cfg_path = 'config/smoke_test.yaml' if args.smoke_test else 'config/full_experiment.yaml'
    cfg = load_cfg(cfg_path)
    cfg['smoke_test'] = args.smoke_test

    set_seed(cfg.get('random_seed', 42))

    try:
        logging.info("[Phase 1] Pre-processing …")
        preprocess.run_preprocessing(cfg)
        logging.info("[Phase 2] Training …")
        train.run_training(cfg)
        logging.info("[Phase 3] Evaluation …")
        evaluate.run_evaluation(cfg)
        logging.info("Pipeline finished successfully ✨")
    except Exception as e:
        logging.critical("Unhandled exception: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

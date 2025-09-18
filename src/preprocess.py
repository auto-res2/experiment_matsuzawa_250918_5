import os
import sys
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import numpy as np
import logging
import fasttext
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Custom Dataset class to handle our structure
class PreprocessedDataset(Dataset):
    def __init__(self, data):
        # data is expected to be a list of tuples (text, reward, safety_label)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Preprocessing Functions ---

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # A simple normalization: lowercase, strip extra whitespace
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text

def download_fasttext_model(model_path='lid.176.bin'):
    if not os.path.exists(model_path):
        logging.info("fastText language identification model not found. Downloading...")
        try:
            import wget
            wget.download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin')
            logging.info("fastText model downloaded successfully.")
        except Exception as e:
            logging.error(f"Failed to download fastText model. Please download it manually from https://fasttext.cc/docs/en/language-identification.html and place it as 'lid.176.bin'. Error: {e}")
            sys.exit(1)
    return fasttext.load_model(model_path)


def unify_safety_labels(example):
    # Mapping to {0: safe, 1: lexical-unsafe, 2: factual-unsafe}
    # pile-detoxify has a 'toxic' score
    if 'toxic' in example:
        return {'safety_label': 1 if example['toxic'] > 0.7 else 0} # Higher threshold for toxicity
    # laion-safety has 'punsafe'
    if 'punsafe' in example:
        return {'safety_label': 1 if example['punsafe'] > 0.5 else 0}
    # Default to safe
    return {'safety_label': 0}


def run_preprocessing(config):
    logging.info("Starting preprocessing...")
    output_dir = config['data']['processed_dir']
    max_samples = config['data'].get('max_samples', None)

    os.makedirs(output_dir, exist_ok=True)

    # Download language ID model
    ft_model = download_fasttext_model()

    # --- Load and Process Datasets ---
    all_data = []

    # 1. PromptSource-like data (using Alpaca as a substitute for logs)
    logging.info("Loading Alpaca dataset (as PromptSource substitute)")
    try:
        d1 = load_dataset("tatsu-lab/alpaca", split='train')
        if max_samples:
            d1 = d1.select(range(min(len(d1), max_samples // 3)))
        # Create synthetic rewards and safety labels
        for item in d1:
            text = normalize_text(f"Instruction: {item['instruction']} Input: {item['input']} Output: {item['output']}")
            # Synthetic reward based on output length
            reward = min(1.0, len(item['output']) / 200.0) 
            all_data.append((text, reward, 0)) # Assume alpaca is safe
    except Exception as e:
        logging.error(f"Failed to load Alpaca dataset. Aborting. Error: {e}")
        sys.exit(1)

    # 2. Stable-Diffusion-like captions (using a subset of LAION)
    logging.info("Loading LAION-Aesthetics dataset (as Stable Diffusion logs substitute)")
    try:
        # Using a small, accessible dataset as a proxy
        d2 = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split='train')
        if max_samples:
            d2 = d2.select(range(min(len(d2), max_samples // 3)))
        # Create synthetic rewards and safety labels
        for item in d2:
            text = normalize_text(item['TEXT'])
            # Synthetic reward based on caption creativity (e.g., length)
            reward = min(1.0, len(text) / 100.0)
            # No safety labels here, assume safe
            all_data.append((text, reward, 0))
    except Exception as e:
        logging.error(f"Failed to load LAION proxy dataset. Aborting. Error: {e}")
        sys.exit(1)

    # 3. Safety-focused data
    logging.info("Loading DETOX dataset for safety labels")
    try:
        d3 = load_dataset("tomekkorbak/pile-detoxify", split='train')
        if max_samples:
            d3 = d3.select(range(min(len(d3), max_samples // 3)))
        for item in d3:
            # Join the list of texts into a single string
            text = normalize_text(' '.join(item['texts']))
            # Reward is inversely proportional to toxicity score (avg_score is toxicity)
            reward = 1.0 - item['avg_score']
            safety_label = 1 if item['avg_score'] > 0.5 else 0
            all_data.append((text, reward, safety_label))
    except Exception as e:
        logging.error(f"Failed to load DETOX dataset. Aborting. Error: {e}")
        sys.exit(1)

    logging.info(f"Total processed examples: {len(all_data)}")

    # --- Add language ID and create final dataset ---
    final_labeled_data = []
    for text, reward, safety_label in all_data:
        if not text:
            continue
        try:
            # fastText expects a single line of text
            predictions = ft_model.predict(text.replace('\n', ' '))
            lang_id = predictions[0][0].replace('__label__', '')
            final_text = f"<LID:{lang_id}> {text}"
            final_labeled_data.append((final_text, reward, safety_label))
        except Exception:
            # Fallback if language ID fails
            final_labeled_data.append((f"<LID:unk> {text}", reward, safety_label))

    logging.info(f"Total examples with language ID: {len(final_labeled_data)}")

    # --- Split and Save Data ---
    if not final_labeled_data:
        logging.error("No data was processed. Aborting.")
        sys.exit(1)
        
    np.random.seed(config['random_seed'])
    np.random.shuffle(final_labeled_data)

    train_split = int(0.8 * len(final_labeled_data))
    val_split = int(0.9 * len(final_labeled_data))

    train_data = final_labeled_data[:train_split]
    val_data = final_labeled_data[train_split:val_split]
    test_data = final_labeled_data[val_split:]

    # Use custom Dataset class
    train_dataset = PreprocessedDataset(train_data)
    val_dataset = PreprocessedDataset(val_data)
    test_dataset = PreprocessedDataset(test_data)

    torch.save(train_dataset, os.path.join(output_dir, 'train_data.pt'))
    torch.save(val_dataset, os.path.join(output_dir, 'val_data.pt'))
    torch.save(test_dataset, os.path.join(output_dir, 'test_data.pt'))

    logging.info(f"Saved datasets to {output_dir}")
    logging.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    logging.info("Preprocessing finished successfully.")

import os
import sys
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import logging
import fasttext
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PreprocessedDataset(Dataset):
    def __init__(self, data):
        self.data = data  # list[tuple[str, float, int]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ───────────────────────────────────────────────────────────────────────────
#  Helpers
# ───────────────────────────────────────────────────────────────────────────

def normalize_text(text: str):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text


def download_fasttext_model(model_path='lid.176.bin'):
    """Download fastText LID model if missing (≈126 MB)."""
    if not os.path.exists(model_path):
        logging.info("Downloading fastText LID model …")
        try:
            import wget
            wget.download('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin')
        except Exception as e:
            logging.error("Failed to download LID model. Please provide lid.176.bin. Error: %s", e)
            sys.exit(1)
    return fasttext.load_model(model_path)


# ───────────────────────────────────────────────────────────────────────────
#  Main Pre-processing
# ───────────────────────────────────────────────────────────────────────────

def run_preprocessing(config):
    logging.info("Pre-processing …")
    out_dir = config['data']['processed_dir']
    max_samples = config['data'].get('max_samples')
    os.makedirs(out_dir, exist_ok=True)

    ft_model = download_fasttext_model()
    all_data = []

    # 1. Alpaca ----------------------------------------------------------------
    try:
        ds = load_dataset("tatsu-lab/alpaca", split='train')
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples // 3)))
        for itm in ds:
            text = normalize_text(f"Instruction: {itm['instruction']} Input: {itm['input']} Output: {itm['output']}")
            reward = min(1.0, len(itm['output']) / 200.0)
            all_data.append((text, reward, 0))
    except Exception as e:
        logging.error("Failed to load Alpaca: %s", e)
        sys.exit(1)

    # 2. LAION caption proxy ----------------------------------------------------
    try:
        ds = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", split='train')
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples // 3)))
        for itm in ds:
            text = normalize_text(itm['TEXT'])
            reward = min(1.0, len(text) / 100.0)
            all_data.append((text, reward, 0))
    except Exception as e:
        logging.error("Failed to load COCO caption proxy: %s", e)
        sys.exit(1)

    # 3. Detoxify dataset -------------------------------------------------------
    try:
        ds = load_dataset("tomekkorbak/pile-detoxify", split='train')
        if max_samples:
            ds = ds.select(range(min(len(ds), max_samples // 3)))

        # the dataset contains a list of sentences per entry under key 'texts'
        for itm in ds:
            if 'texts' not in itm or 'avg_score' not in itm:
                continue  # skip malformed rows
            joined_text = " ".join(itm['texts'])
            text = normalize_text(joined_text)
            toxicity = float(itm['avg_score'])
            reward = 1.0 - min(1.0, toxicity)  # higher reward for lower toxicity
            safety_label = 1 if toxicity > 0.5 else 0
            all_data.append((text, reward, safety_label))
    except Exception as e:
        logging.error("Failed to load DETOX: %s", e)
        sys.exit(1)

    logging.info(f"Total raw examples: {len(all_data)}")

    # -------------------------------------------------------------------------
    # Language-ID tagging + shuffle
    # -------------------------------------------------------------------------
    final_data = []
    for txt, rew, lab in all_data:
        if not txt:
            continue
        try:
            lid = ft_model.predict(txt.replace('\n', ' '))[0][0].replace('__label__', '')
            tagged = f"<LID:{lid}> {txt}"
            final_data.append((tagged, rew, lab))
        except Exception:
            final_data.append((f"<LID:unk> {txt}", rew, lab))

    np.random.seed(config['random_seed'])
    np.random.shuffle(final_data)

    n = len(final_data)
    train_split, val_split = int(0.8 * n), int(0.9 * n)
    train_ds = PreprocessedDataset(final_data[:train_split])
    val_ds = PreprocessedDataset(final_data[train_split:val_split])
    test_ds = PreprocessedDataset(final_data[val_split:])

    torch.save(train_ds, os.path.join(out_dir, 'train_data.pt'))
    torch.save(val_ds, os.path.join(out_dir, 'val_data.pt'))
    torch.save(test_ds, os.path.join(out_dir, 'test_data.pt'))

    logging.info(f"Saved datasets to {out_dir} – train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

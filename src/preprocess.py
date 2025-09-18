import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
import timm
from datasets import load_dataset
from PIL import Image
import numpy as np
import math
from itertools import cycle, islice

# --- PGD Attack for Experiment 3 ---

def pgd_attack(model, images, labels, eps=4 / 255, alpha=1 / 255, steps=10):
    images = images.clone().detach()
    labels = labels.clone().detach()
    loss_fn = torch.nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1)

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


# --- Transformations ---

def get_transform(config, model):
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return transform


# --- Dataset Loaders ---

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = item.get("label", 0)  # Use 0 if no label
        return self.transform(image), label


def get_imagenet_val_loader(config, batch_size):
    try:
        dataset = load_dataset(
            "Tsomaros/Imagenet-1k_validation",
            split="validation",
            cache_dir=os.path.join(config["project"]["data_dir"], "hf_cache"),
        )
    except Exception as e:
        print(
            f"FATAL: Could not download ImageNet validation set. Error: {e}",
            file=sys.stderr,
        )
        print("Please check your network connection and Hugging Face credentials.", file=sys.stderr)
        sys.exit(1)

    model = timm.create_model(config["train"]["backbones"][0], pretrained=False)  # for transform only
    transform = get_transform(config, model)
    torch_dataset = HFDataset(dataset, transform)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=2)


# --- Synthetic Stream for Meta-Training ---

class SyntheticStream(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.sequence_length = 3
        self.base_loader = get_imagenet_val_loader(
            config, batch_size=config["train"]["batch_size"]
        )
        self.base_iterator = iter(self.base_loader)
        # Use only tensor-compatible augmentations
        self.augs = [
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomSolarize(threshold=0.5),
        ]

    def __iter__(self):
        return self

    def __next__(self):
        sequence = []
        for _ in range(self.sequence_length):
            try:
                images, labels = next(self.base_iterator)
            except StopIteration:
                self.base_iterator = iter(self.base_loader)
                images, labels = next(self.base_iterator)

            aug = np.random.choice(self.augs)
            # Apply augmentation image-wise to avoid batch incompatibilities
            aug_images = torch.stack([aug(img) for img in images])
            sequence.append((aug_images, labels))
        return sequence


def get_synthetic_stream_for_training(config):
    return SyntheticStream(config)


# ============================================================
# ADDITIONAL PREPROCESS HELPERS FOR EVALUATION EXPERIMENTS
# ============================================================


def _simple_cycle(loader):
    """Utility: endlessly cycle through a dataloader."""
    while True:
        for batch in loader:
            yield batch


# --- Experiment 1: Recurring Shift Stream ---

def get_recurring_shift_stream(config, eta):
    """Builds an iterator over a long stream with synthetic recurring shifts.

    For the purposes of the smoke-test we simply cycle through ImageNet-val
    batches.  A real implementation would apply corruptions that change every
    `corruption_change_freq` samples and recur every `recurrence_freq`.
    """
    total_samples = config["experiment_1"]["total_samples"]
    batch_size = config["evaluate"]["batch_size"]
    loader = get_imagenet_val_loader(config, batch_size=batch_size)
    cyc = _simple_cycle(loader)
    processed = 0
    while processed < total_samples:
        imgs, lbls = next(cyc)
        processed += imgs.size(0)
        yield imgs, lbls


# --- Experiment 2 helpers --------------------------------------------------

def get_imagenet_c_loader(config, corruption, severity):
    """Placeholder that re-uses ImageNet-val for the smoke-test.

    Returns a dataloader or None if data unavailable.  Full experiments should
    download ImageNet-C and apply the requested corruption.
    """
    try:
        return get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
    except Exception:
        return None


def get_domainnet_loaders(config, source_domain, target_domain):
    """Placeholder that re-uses ImageNet-val as a stand-in for DomainNet.

    For quick smoke tests this is acceptable; full experiments must replace
    with genuine DomainNet loaders.
    """
    return get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])


# --- Experiment 3: Stress-Test Stream --------------------------------------

def get_stress_test_stream(config, model):
    """Creates a mixed stream of clean/noisy/adv batches for the gate test.

    The smoke-test implementation cycles over ImageNet-val without adversarial
    perturbations to keep runtime low while still exercising the code-path.
    """
    loader = get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
    cyc = _simple_cycle(loader)
    total = config["experiment_3"]["total_samples"]
    processed = 0
    while processed < total:
        imgs, lbls = next(cyc)
        processed += imgs.size(0)
        flag = "clean"
        yield imgs, lbls, flag

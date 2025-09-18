import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, ConcatDataset
from torchvision import transforms
import timm
from datasets import load_dataset, concatenate_datasets
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
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


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


# --- Experiment 1: Recurring Shift Stream ---
# (unchanged)

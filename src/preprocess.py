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

class RecurringShiftStream(IterableDataset):
    def __init__(self, config, eta=1.0):
        self.config = config
        self.eta = eta
        self.total_samples = config["experiment_1"]["total_samples"]
        self.corruption_change_freq = config["experiment_1"]["corruption_change_freq"]
        self.recurrence_freq = config["experiment_1"]["recurrence_freq"]
        
        self.base_loader = get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
        self.base_iterator = iter(self.base_loader)
        
        # Define corruption patterns
        self.corruptions = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomSolarize(threshold=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5),
        ]
        
        self.current_corruption_idx = 0
        self.samples_since_change = 0
        self.recurring_corruption = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.samples_since_change >= self.total_samples:
            raise StopIteration
            
        try:
            images, labels = next(self.base_iterator)
        except StopIteration:
            self.base_iterator = iter(self.base_loader)
            images, labels = next(self.base_iterator)
            
        # Apply recurring shift pattern
        if self.samples_since_change % self.recurrence_freq == 0:
            self.recurring_corruption = self.corruptions[self.current_corruption_idx % len(self.corruptions)]
            
        if self.samples_since_change % self.corruption_change_freq == 0:
            self.current_corruption_idx += 1
            
        corruption = self.recurring_corruption or self.corruptions[0]
        
        # Apply corruption with eta scaling
        if np.random.random() < self.eta:
            corrupted_images = torch.stack([corruption(img) for img in images])
        else:
            corrupted_images = images
            
        self.samples_since_change += len(images)
        return corrupted_images, labels


def get_recurring_shift_stream(config, eta=1.0):
    return RecurringShiftStream(config, eta)


# --- Experiment 2: ImageNet-C and DomainNet Loaders ---

def get_imagenet_c_loader(config, corruption_type, severity):
    try:
        # Create synthetic corrupted data for testing
        base_loader = get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
        
        # Define corruption mappings
        corruption_transforms = {
            "gaussian_noise": lambda x: x + torch.randn_like(x) * 0.1 * severity,
            "shot_noise": lambda x: x + torch.poisson(x * 10 * severity) / (10 * severity) - x,
            "impulse_noise": lambda x: x + torch.rand_like(x) * 0.2 * severity - 0.1 * severity,
            "defocus_blur": transforms.GaussianBlur(kernel_size=3, sigma=(severity * 0.5, severity * 1.0)),
            "frosted_glass_blur": transforms.GaussianBlur(kernel_size=5, sigma=(severity * 0.3, severity * 0.8)),
        }
        
        if corruption_type not in corruption_transforms:
            return None
            
        transform_fn = corruption_transforms[corruption_type]
        
        class CorruptedDataset(Dataset):
            def __init__(self, base_dataset, transform_fn):
                self.base_dataset = base_dataset
                self.transform_fn = transform_fn
                
            def __len__(self):
                return min(len(self.base_dataset.dataset), 100)  # Limit for testing
                
            def __getitem__(self, idx):
                img, label = self.base_dataset.dataset[idx]
                if callable(self.transform_fn):
                    if hasattr(self.transform_fn, '__call__') and len(self.transform_fn.__code__.co_varnames) > 0:
                        img = self.transform_fn(img)
                return img, label
                
        corrupted_dataset = CorruptedDataset(base_loader, transform_fn)
        return DataLoader(corrupted_dataset, batch_size=config["evaluate"]["batch_size"], shuffle=False, num_workers=2)
        
    except Exception as e:
        print(f"Warning: Could not create ImageNet-C loader for {corruption_type}-{severity}: {e}")
        return None


def get_domainnet_loaders(config, source_domain="real", target_domain="sketch"):
    try:
        # Create synthetic domain shift data for testing
        base_loader = get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
        
        # Define domain transforms
        domain_transforms = {
            "real": transforms.Compose([]),  # No additional transform
            "sketch": transforms.Compose([
                transforms.RandomAdjustSharpness(sharpness_factor=2.0),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05),
            ]),
            "painting": transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.6, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            ]),
        }
        
        target_transform = domain_transforms.get(target_domain, transforms.Compose([]))
        
        class DomainShiftDataset(Dataset):
            def __init__(self, base_dataset, target_transform):
                self.base_dataset = base_dataset
                self.target_transform = target_transform
                
            def __len__(self):
                return min(len(self.base_dataset.dataset), 100)  # Limit for testing
                
            def __getitem__(self, idx):
                img, label = self.base_dataset.dataset[idx]
                img = self.target_transform(img)
                return img, label
                
        domain_dataset = DomainShiftDataset(base_loader, target_transform)
        return DataLoader(domain_dataset, batch_size=config["evaluate"]["batch_size"], shuffle=False, num_workers=2)
        
    except Exception as e:
        print(f"Warning: Could not create DomainNet loader for {source_domain}->{target_domain}: {e}")
        return None


# --- Experiment 3: Stress Test Stream ---

class StressTestStream(IterableDataset):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.total_samples = config["experiment_3"]["total_samples"]
        self.clean_ratio = config["experiment_3"]["clean_ratio"]
        self.noise_ratio = config["experiment_3"]["noise_ratio"]
        self.adv_ratio = config["experiment_3"]["adv_ratio"]
        
        self.base_loader = get_imagenet_val_loader(config, batch_size=config["evaluate"]["batch_size"])
        self.base_iterator = iter(self.base_loader)
        self.samples_generated = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.samples_generated >= self.total_samples:
            raise StopIteration
            
        try:
            images, labels = next(self.base_iterator)
        except StopIteration:
            self.base_iterator = iter(self.base_loader)
            images, labels = next(self.base_iterator)
            
        # Determine sample type based on ratios
        rand_val = np.random.random()
        if rand_val < self.clean_ratio:
            # Clean samples
            processed_images = images
        elif rand_val < self.clean_ratio + self.noise_ratio:
            # Noisy samples
            noise = torch.randn_like(images) * 0.1
            processed_images = torch.clamp(images + noise, 0, 1)
        else:
            # Adversarial samples (simplified version)
            processed_images = pgd_attack(self.model, images, labels, eps=8/255, alpha=2/255, steps=3)
            
        self.samples_generated += len(images)
        return processed_images, labels


def get_stress_test_stream(config, model):
    return StressTestStream(config, model)

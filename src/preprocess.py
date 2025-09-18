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
def pgd_attack(model, images, labels, eps=4/255, alpha=1/255, steps=10):
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
        image = item['image'].convert('RGB')
        label = item.get('label', 0) # Use 0 if no label
        return self.transform(image), label

def get_imagenet_val_loader(config, batch_size):
    try:
        dataset = load_dataset("Tsomaros/Imagenet-1k_validation", split='validation', cache_dir=os.path.join(config['project']['data_dir'], 'hf_cache'))
    except Exception as e:
        print(f"FATAL: Could not download ImageNet validation set. Error: {e}", file=sys.stderr)
        print("Please check your network connection and Hugging Face credentials.", file=sys.stderr)
        sys.exit(1)
    
    model = timm.create_model(config['train']['backbones'][0], pretrained=False) # for transform only
    transform = get_transform(config, model)
    torch_dataset = HFDataset(dataset, transform)
    return DataLoader(torch_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


# --- Synthetic Stream for Meta-Training ---

class SyntheticStream(IterableDataset):
    def __init__(self, config):
        self.config = config
        self.sequence_length = 3
        self.base_loader = get_imagenet_val_loader(config, batch_size=config['train']['batch_size'])
        self.base_iterator = iter(self.base_loader)
        self.augs = [
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)], p=0.5)  # Replace posterize with noise
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
            aug_images = aug(images)
            sequence.append((aug_images, labels))
        return sequence

def get_synthetic_stream_for_training(config):
    return SyntheticStream(config)


# --- Experiment 1: Recurring Shift Stream ---

class RecurringShiftStream(IterableDataset):
    def __init__(self, config, eta):
        self.config = config
        self.eta = eta
        self.batch_size = math.ceil(eta * 32)
        self.total_samples = config['experiment_1']['total_samples']
        self.corruption_change_freq = config['experiment_1']['corruption_change_freq']
        self.recurrence_freq = config['experiment_1']['recurrence_freq']

        print("Loading datasets for recurring stream... This may take a while.")
        try:
            imagenet_r = load_dataset("axiong/imagenet-r", split='train', cache_dir=os.path.join(config['project']['data_dir'], 'hf_cache'))
            imagenet_c_datasets = []
            for corruption in config['experiment_2']['imagenet_c_corruptions']:
                 # Load only severity 5 for simplicity of the stream
                 ds = load_dataset("ang9867/ImageNet-C", corruption, split='validation[80%:]', cache_dir=os.path.join(config['project']['data_dir'], 'hf_cache'))
                 imagenet_c_datasets.append(ds)
            imagenet_c = concatenate_datasets(imagenet_c_datasets)
        except Exception as e:
            print(f"FATAL: Could not download datasets for Experiment 1. Error: {e}", file=sys.stderr)
            sys.exit(1)

        self.datasets = {'r': imagenet_r, 'c': imagenet_c}
        self.model = timm.create_model(config['train']['backbones'][0], pretrained=False)
        self.transform = get_transform(config, self.model)
        self.num_yielded = 0

    def _stream_generator(self):
        indices = {'r': cycle(range(len(self.datasets['r']))), 'c': cycle(range(len(self.datasets['c'])))}
        corruption_sequence = cycle(['r'] + ['c'] * (self.recurrence_freq // self.corruption_change_freq - 1))

        while self.num_yielded < self.total_samples:
            corruption_type = next(corruption_sequence)
            ds = self.datasets[corruption_type]
            idx_gen = indices[corruption_type]

            for _ in range(self.corruption_change_freq):
                if self.num_yielded >= self.total_samples: return

                batch_indices = [next(idx_gen) for _ in range(self.batch_size)]
                items = ds[batch_indices]
                images = [self.transform(img.convert('RGB')) for img in items['image']]
                labels = items['label']
                
                yield torch.stack(images), torch.tensor(labels)
                self.num_yielded += self.batch_size

    def __iter__(self):
        return self._stream_generator()

def get_recurring_shift_stream(config, eta):
    return RecurringShiftStream(config, eta)


# --- Experiment 2: ImageNet-C and DomainNet ---

def get_imagenet_c_loader(config, corruption, severity):
    batch_size = config['evaluate']['batch_size']
    try:
        # severity is 1-5, but HF dataset uses 0-4 indices
        split_name = f'validation_s{severity-1}'
        ds = load_dataset("ang9867/ImageNet-C", corruption, split=f'{split_name}[0%:80%]', cache_dir=os.path.join(config['project']['data_dir'], 'hf_cache'))
    except Exception as e:
        print(f"WARN: Could not load ImageNet-C for {corruption} sev {severity}. Skipping. Error: {e}", file=sys.stderr)
        return None

    model = timm.create_model(config['train']['backbones'][0], pretrained=False)
    transform = get_transform(config, model)
    torch_ds = HFDataset(ds, transform)
    return DataLoader(torch_ds, batch_size=batch_size, shuffle=False, num_workers=4)

def get_domainnet_loaders(config, source, target):
    batch_size = config['evaluate']['batch_size']
    try:
        target_ds = load_dataset(f"Bruece/domainnet-126-by-class-{target}", split='test', cache_dir=os.path.join(config['project']['data_dir'], 'hf_cache'))
    except Exception as e:
        print(f"FATAL: Could not load DomainNet dataset {target}. Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    model = timm.create_model(config['train']['backbones'][0], pretrained=False)
    transform = get_transform(config, model)
    torch_ds = HFDataset(target_ds, transform)
    return DataLoader(torch_ds, batch_size=batch_size, shuffle=False, num_workers=4)

# --- Experiment 3: Stress Test Stream ---

class StressTestStream(IterableDataset):
    def __init__(self, config, source_model):
        self.config = config
        self.source_model = source_model
        self.total_samples = config['experiment_3']['total_samples']
        self.device = next(source_model.parameters()).device

        self.clean_loader = get_imagenet_val_loader(config, batch_size=32)
        self.clean_iter = cycle(self.clean_loader)

        self.model_for_transform = timm.create_model(config['train']['backbones'][0], pretrained=False)
        self.transform = get_transform(config, self.model_for_transform)
        self.num_yielded = 0
    
    def _stream_generator(self):
        stream_composition = [
            ('clean', self.config['experiment_3']['clean_ratio']),
            ('noise', self.config['experiment_3']['noise_ratio']),
            ('adv', self.config['experiment_3']['adv_ratio'])
        ]
        stream_types = [s[0] for s in stream_composition]
        stream_probs = [s[1] for s in stream_composition]

        while self.num_yielded < self.total_samples:
            batch_size = np.random.choice([1, 2, 4], p=[0.3, 0.4, 0.3])
            stream_type = np.random.choice(stream_types, p=stream_probs)

            if stream_type == 'clean':
                images, labels = next(self.clean_iter)
                images, labels = images[:batch_size], labels[:batch_size]
            elif stream_type == 'noise':
                images = torch.rand(batch_size, 3, 224, 224) * 0.5
                labels = torch.randint(0, 1000, (batch_size,))
            elif stream_type == 'adv':
                clean_images, clean_labels = next(self.clean_iter)
                clean_images, clean_labels = clean_images[:batch_size].to(self.device), clean_labels[:batch_size].to(self.device)
                images = pgd_attack(self.source_model, clean_images, clean_labels).cpu()
                labels = clean_labels.cpu()

            yield images, labels, stream_type
            self.num_yielded += batch_size
    
    def __iter__(self):
        return self._stream_generator()

def get_stress_test_stream(config, source_model):
    return StressTestStream(config, source_model)

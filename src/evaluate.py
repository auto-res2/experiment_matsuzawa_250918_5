import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import json
import time
from collections import deque
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fvcore.nn import FlopCountAnalysis

from preprocess import (
    get_recurring_shift_stream,
    get_imagenet_c_loader,
    get_domainnet_loaders,
    get_stress_test_stream
)
from train import inject_lora, LoRALinear, LoRAConv2d, HyperGRU

class FLASH_Adapter:
    def __init__(self, model, hyper_network, fisher_diagonals, config, device):
        self.model = model
        self.hyper_network = hyper_network
        self.fisher_diagonals = fisher_diagonals
        self.config = config
        self.device = device
        self.ema_decay = config['evaluate']['ema_decay']
        self.gate_threshold = config['evaluate']['gate_threshold']
        self.lora_rank = config['train']['lora_rank']

        self.lora_modules = {name: mod for name, mod in self.model.named_modules() if isinstance(mod, (LoRALinear, LoRAConv2d))}
        self.param_offsets = self._compute_param_offsets()
        self.total_lora_params = list(self.param_offsets.values())[-1]
        self.stats_window = deque(maxlen=3)
        self.ema_delta = torch.zeros(self.total_lora_params, device=self.device)

        # Feature statistics hooks
        self.hooks = []
        self.activations = {}
        self.ref_stats = self._compute_ref_stats()

    def _compute_param_offsets(self):
        offsets = {}
        current_offset = 0
        for name, mod in self.lora_modules.items():
            num_params = sum(p.numel() for p in mod.get_lora_params())
            offsets[name] = current_offset
            current_offset += num_params
        offsets['total'] = current_offset
        return offsets

    def _compute_ref_stats(self):
        from preprocess import get_imagenet_val_loader
        ref_loader = get_imagenet_val_loader(self.config, self.config['train']['batch_size'])
        ref_images, _ = next(iter(ref_loader))
        ref_images = ref_images.to(self.device)

        ref_stats = {}
        hooks = []
        def get_hook(name):
            def hook_fn(module, input, output):
                if isinstance(module, LoRAConv2d):
                    mean, std = output.mean([0, 2, 3]), output.std([0, 2, 3])
                else:
                    mean, std = output.mean(0), output.std(0)
                ref_stats[name] = (mean.detach(), std.detach())
            return hook_fn
        
        for name, mod in self.lora_modules.items():
            hooks.append(mod.register_forward_hook(get_hook(name)))
        
        with torch.no_grad():
            self.model(ref_images)
        
        for h in hooks:
            h.remove()
        return ref_stats

    def _get_feature_shift_vector(self, images):
        self.activations.clear()

        def get_hook(name):
            def hook_fn(module, input, output):
                self.activations[name] = output
            return hook_fn
        
        for name, mod in self.lora_modules.items():
            self.hooks.append(mod.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            outputs = self.model(images)

        for h in self.hooks:
            h.remove()
        self.hooks.clear()

        s_vector_parts = []
        entropy = -torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1).mean()

        for name, mod in self.lora_modules.items():
            act = self.activations[name]
            if isinstance(mod, LoRAConv2d):
                mu_t, sigma_t = act.mean([0, 2, 3]), act.std([0, 2, 3])
            else:
                mu_t, sigma_t = act.mean(0), act.std(0)
            
            mu_0, sigma_0 = self.ref_stats[name]
            mu_diff = (mu_t - mu_0).mean().item()
            sigma_diff = (torch.log(sigma_t.clamp(min=1e-6)) - torch.log(sigma_0.clamp(min=1e-6))).mean().item()
            s_vector_parts.extend([mu_diff, sigma_diff, entropy.item()])
        
        return torch.tensor(s_vector_parts, device=self.device)

    @torch.no_grad()
    def step(self, images):
        s_t = self._get_feature_shift_vector(images)
        self.stats_window.append(s_t)

        if len(self.stats_window) < 3:
            return False # Not enough history

        s_sequence = torch.stack(list(self.stats_window)).unsqueeze(0)
        u_pred_vec, g_t = self.hyper_network(s_sequence)
        u_pred_vec = u_pred_vec.squeeze(0)

        if g_t.item() < self.gate_threshold:
            return False # Skip update

        current_offset = 0
        delta_w_vec_parts = []
        for name, mod in self.lora_modules.items():
            num_params_A = mod.get_lora_params()[0].numel()
            num_params_B = mod.get_lora_params()[1].numel()
            
            u_A_flat = u_pred_vec[current_offset : current_offset + num_params_A]
            u_B_flat = u_pred_vec[current_offset + num_params_A : current_offset + num_params_A + num_params_B]
            
            f_sqrt_A = torch.sqrt(self.fisher_diagonals[name][0].clamp(min=1e-8)).flatten()
            f_sqrt_B = torch.sqrt(self.fisher_diagonals[name][1].clamp(min=1e-8)).flatten()
            
            delta_A_flat = u_A_flat / f_sqrt_A
            delta_B_flat = u_B_flat / f_sqrt_B
            
            delta_w_vec_parts.append(delta_A_flat)
            delta_w_vec_parts.append(delta_B_flat)
            current_offset += num_params_A + num_params_B
            
        delta_w_vec = torch.cat(delta_w_vec_parts)
        self.ema_delta = self.ema_decay * self.ema_delta + (1 - self.ema_decay) * (g_t * delta_w_vec)

        return True # Update applied

    def forward(self, x):
        # Apply the accumulated EMA delta to the LoRA weights for this forward pass
        current_offset = 0
        original_params = {}
        with torch.no_grad():
            for name, mod in self.lora_modules.items():
                params = mod.get_lora_params()
                original_params[name] = [p.clone() for p in params]
                
                num_params_A = params[0].numel()
                num_params_B = params[1].numel()

                delta_A = self.ema_delta[current_offset : current_offset + num_params_A].view_as(params[0])
                delta_B = self.ema_delta[current_offset + num_params_A : current_offset + num_params_A + num_params_B].view_as(params[1])
                
                params[0].add_(delta_A)
                params[1].add_(delta_B)
                current_offset += num_params_A + num_params_B
        
        outputs = self.model(x)
        
        # Restore original parameters
        with torch.no_grad():
            for name, mod in self.lora_modules.items():
                params = mod.get_lora_params()
                for i, p_orig in enumerate(original_params[name]):
                    params[i].copy_(p_orig)
        
        return outputs


def run_experiment_1(config, adapter, backbone_name):
    print("Running Experiment 1: Streaming Non-IID & Recurring-Shift Benchmark")
    results = {
        'experiment': 1,
        'backbone': backbone_name,
        'online_top1': [],
        'timestamps': [],
        'updates_skipped': 0,
        'total_steps': 0
    }
    
    for eta in config['experiment_1']['eta_values']:
        print(f"--- Running with eta = {eta} ---")
        stream = get_recurring_shift_stream(config, eta)
        adapter.stats_window.clear() # Reset for new stream
        
        total_correct = 0
        total_samples = 0
        start_time = time.time()

        for i, (images, labels) in enumerate(tqdm(stream, desc=f"Exp1 (eta={eta})")):
            images, labels = images.to(adapter.device), labels.to(adapter.device)
            
            # 1. Adapt
            updated = adapter.step(images)
            if not updated:
                results['updates_skipped'] += 1
            results['total_steps'] += 1
            
            # 2. Evaluate
            outputs = adapter.forward(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
            if (i + 1) % 100 == 0:
                current_acc = total_correct / total_samples
                results['online_top1'].append(current_acc)
                results['timestamps'].append(time.time() - start_time)
        
        final_accuracy = total_correct / total_samples
        print(f"Final Accuracy for eta={eta}: {final_accuracy:.4f}")
        results[f'final_accuracy_eta_{eta}'] = final_accuracy
    
    log_and_plot_results(results, 'experiment_1', config, backbone_name)
    return results

def run_experiment_2(config, adapter, backbone_name):
    print("Running Experiment 2: Architecture-Agnostic BN-Free Adaptation Study")
    results = {'experiment': 2, 'backbone': backbone_name}

    # ImageNet-C
    print("--- Evaluating on ImageNet-C ---")
    imagenetc_accuracies = []
    for corruption in config['experiment_2']['imagenet_c_corruptions']:
        for severity in range(1, 6):
            adapter.stats_window.clear()
            loader = get_imagenet_c_loader(config, corruption, severity)
            if loader is None: continue
            
            total_correct, total_samples = 0, 0
            for images, labels in tqdm(loader, desc=f"ImageNet-C: {corruption} sev {severity}"):
                images, labels = images.to(adapter.device), labels.to(adapter.device)
                adapter.step(images)
                outputs = adapter.forward(images)
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += len(labels)
            
            acc = total_correct / total_samples
            imagenetc_accuracies.append(acc)
            print(f"Accuracy for {corruption} severity {severity}: {acc:.4f}")
    
    results['imagenet_c_mean_accuracy'] = np.mean(imagenetc_accuracies)
    print(f"Mean ImageNet-C Accuracy: {results['imagenet_c_mean_accuracy']:.4f}")

    # DomainNet
    print("--- Evaluating on DomainNet (Real -> Sketch) ---")
    domainnet_loader = get_domainnet_loaders(config, source='real', target='sketch')
    adapter.stats_window.clear()
    total_correct, total_samples = 0, 0
    for images, labels in tqdm(domainnet_loader, desc="DomainNet Sketch"):
        images, labels = images.to(adapter.device), labels.to(adapter.device)
        adapter.step(images)
        outputs = adapter.forward(images)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)
    results['domainnet_sketch_accuracy'] = total_correct / total_samples
    print(f"DomainNet Sketch Accuracy: {results['domainnet_sketch_accuracy']:.4f}")
    
    log_and_plot_results(results, 'experiment_2', config, backbone_name)
    return results

def run_experiment_3(config, adapter, backbone_name):
    print("Running Experiment 3: Safety-Gate Stress Test")
    results = {'experiment': 3, 'backbone': backbone_name, 'gate_on_acc': [], 'gate_forced_acc': [], 'source_acc': []}
    stream_gate_on = get_stress_test_stream(config, adapter.model)
    stream_gate_forced = get_stress_test_stream(config, adapter.model)
    stream_source = get_stress_test_stream(config, adapter.model)

    # Gate ON
    print("--- Running with Safety Gate ON ---")
    adapter.stats_window.clear()
    total_correct, total_samples, skipped = 0, 0, 0
    for images, labels, _ in tqdm(stream_gate_on, desc="Gate ON"):
        images, labels = images.to(adapter.device), labels.to(adapter.device)
        if not adapter.step(images): skipped += 1
        outputs = adapter.forward(images)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)
        results['gate_on_acc'].append(total_correct / total_samples)
    results['gate_on_final_acc'] = total_correct / total_samples
    results['updates_skipped_ratio'] = skipped / total_samples
    print(f"Gate ON final accuracy: {results['gate_on_final_acc']:.4f}, Skipped: {results['updates_skipped_ratio']:.2%}")

    # Gate FORCED ON (g_t = 1)
    print("--- Running with Safety Gate FORCED ON ---")
    adapter.gate_threshold = -1.0 # effectively force on
    adapter.stats_window.clear()
    total_correct, total_samples = 0, 0
    for images, labels, _ in tqdm(stream_gate_forced, desc="Gate FORCED"):
        images, labels = images.to(adapter.device), labels.to(adapter.device)
        adapter.step(images)
        outputs = adapter.forward(images)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)
        results['gate_forced_acc'].append(total_correct / total_samples)
    results['gate_forced_final_acc'] = total_correct / total_samples
    print(f"Gate FORCED ON final accuracy: {results['gate_forced_final_acc']:.4f}")
    adapter.gate_threshold = config['evaluate']['gate_threshold'] # reset

    # SOURCE (no adaptation)
    print("--- Running with SOURCE (no adaptation) ---")
    total_correct, total_samples = 0, 0
    for images, labels, _ in tqdm(stream_source, desc="SOURCE"):
        images, labels = images.to(adapter.device), labels.to(adapter.device)
        outputs = adapter.model(images)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += len(labels)
        results['source_acc'].append(total_correct / total_samples)
    results['source_final_acc'] = total_correct / total_samples
    print(f"SOURCE final accuracy: {results['source_final_acc']:.4f}")

    results['worst_case_drop_gate_on'] = results['source_final_acc'] - min(results['gate_on_acc'])
    results['worst_case_drop_gate_forced'] = results['source_final_acc'] - min(results['gate_forced_acc'])

    log_and_plot_results(results, 'experiment_3', config, backbone_name)
    return results

def log_and_plot_results(results, exp_name, config, backbone_name):
    output_dir = config['project']['output_dir']
    results_dir = os.path.join(output_dir, 'results')
    plots_dir = os.path.join(output_dir, 'images')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    bb_name_safe = backbone_name.replace('/', '_')
    filepath = os.path.join(results_dir, f"{exp_name}_{bb_name_safe}_results.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n--- Results for {exp_name} with {backbone_name} ---")
    print(json.dumps(results, indent=4))
    print(f"Results saved to {filepath}")

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    if exp_name == 'experiment_1':
        plt.figure(figsize=(10, 6))
        plt.plot(results['timestamps'], results['online_top1'], label='Online Top-1 Accuracy')
        plt.xlabel('Time (s)')
        plt.ylabel('Accuracy')
        plt.title(f'Exp 1: Online Accuracy Over Time - {backbone_name}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'exp1_{bb_name_safe}_accuracy_vs_time.pdf'))
        plt.close()
    elif exp_name == 'experiment_3':
        plt.figure(figsize=(12, 7))
        plt.plot(results['gate_on_acc'], label='FLASH (Gate ON)')
        plt.plot(results['gate_forced_acc'], label='FLASH (Gate FORCED ON)', linestyle='--')
        plt.plot(results['source_acc'], label='SOURCE (Frozen)', linestyle=':')
        plt.xlabel('Time Steps')
        plt.ylabel('Running Top-1 Accuracy')
        plt.title(f'Exp 3: Safety-Gate Stress Test - {backbone_name}')
        plt.legend()
        plt.ylim(bottom=0)
        plt.savefig(os.path.join(plots_dir, f'exp3_{bb_name_safe}_stress_test.pdf'))
        plt.close()

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['project']['seed'])

    output_dir = config['project']['output_dir']
    
    for backbone_name in config['train']['backbones']:
        print(f"\n{'='*20} EVALUATING {backbone_name} {'='*20}")
        bb_name_safe = backbone_name.replace('/', '_')

        # Load model and inject LoRA
        model = timm.create_model(backbone_name, pretrained=True).to(device)
        model = inject_lora(model, config['train']['lora_rank'])
        model.eval()

        # Load Fisher diagonals
        fisher_path = os.path.join(output_dir, f"{bb_name_safe}_fisher.pth")
        if not os.path.exists(fisher_path):
            print(f"ERROR: Fisher file not found at {fisher_path}. Please run training first.")
            sys.exit(1)
        fisher_diagonals = torch.load(fisher_path, map_location=device, weights_only=False)

        # Load Hyper-network
        num_lora_params_total = sum(p.numel() for mod in get_lora_modules(model) for p in mod.get_lora_params())
        feature_dim = 3
        input_dim = len(get_lora_modules(model)) * feature_dim
        hyper_network = HyperGRU(input_dim, config['train']['gru_hidden_dim'], num_lora_params_total).to(device)
        hyper_network_path = os.path.join(output_dir, f"hyper_gru_{bb_name_safe}.pth")
        if not os.path.exists(hyper_network_path):
             # In a real setup, we might train one universal hypernet, but here we train one per backbone.
             # For now, we assume the one for the first backbone is universal.
             first_bb_safe = config['train']['backbones'][0].replace('/', '_')
             hyper_network_path = os.path.join(output_dir, f"hyper_gru_{first_bb_safe}.pth")
             if not os.path.exists(hyper_network_path):
                print(f"ERROR: HyperGRU model not found at {hyper_network_path}. Please run training first.")
                sys.exit(1)
        
        hyper_network.load_state_dict(torch.load(hyper_network_path, map_location=device))
        hyper_network.eval()

        # Create FLASH adapter
        adapter = FLASH_Adapter(model, hyper_network, fisher_diagonals, config, device)

        # Run experiments
        exp = config['evaluate']['experiment']
        if exp == 'exp1':
            run_experiment_1(config, adapter, backbone_name)
        elif exp == 'exp2':
            run_experiment_2(config, adapter, backbone_name)
        elif exp == 'exp3':
            run_experiment_3(config, adapter, backbone_name)
        else:
            print(f"ERROR: Unknown experiment '{exp}'. Choose from 'exp1', 'exp2', 'exp3'.")
            sys.exit(1)

def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, (LoRALinear, LoRAConv2d))]

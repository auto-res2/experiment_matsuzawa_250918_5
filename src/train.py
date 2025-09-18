import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import timm
from collections import deque
import math

from .preprocess import get_synthetic_stream_for_training, get_imagenet_val_loader

# --- LoRA Layers ---

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank

        self.lora_A = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_layer = linear_layer
        self.base_layer.weight.requires_grad = False

    def forward(self, x):
        lora_update = self.lora_A @ self.lora_B
        return self.base_layer(x) + F.linear(x, lora_update)

    def get_lora_params(self):
        return [self.lora_A, self.lora_B]

class LoRAConv2d(nn.Module):
    def __init__(self, conv_layer, rank):
        super().__init__()
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.rank = rank

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, 1, 1))
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank, 1, 1))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base_layer = conv_layer
        self.base_layer.weight.requires_grad = False

    def forward(self, x):
        lora_kernel = self.lora_B @ self.lora_A
        lora_kernel_resized = F.interpolate(lora_kernel, size=self.kernel_size, mode='replicate')
        lora_update = F.conv2d(x, lora_kernel_resized, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.base_layer(x) + lora_update
    
    def get_lora_params(self):
        return [self.lora_A, self.lora_B]

# --- Hyper-Network ---

class HyperGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_lora_params):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = nn.Linear(hidden_dim, num_lora_params + 1) # +1 for the gate scalar

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)
        h_n = h_n[-1] # Get last layer's hidden state
        out = self.head(h_n)
        params_vec, gate_logit = out[..., :-1], out[..., -1]
        gate = torch.sigmoid(gate_logit)
        return params_vec, gate

# --- Utility Functions ---

def inject_lora(model, rank):
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if 'head' in name or 'fc' in name:
                continue # Skip classifier head
            if isinstance(module, nn.Linear):
                lora_layer = LoRALinear(module, rank)
                setattr(model, name, lora_layer)
            elif isinstance(module, nn.Conv2d):
                lora_layer = LoRAConv2d(module, rank)
                setattr(model, name, lora_layer)
        else:
            inject_lora(module, rank)
    return model

def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, (LoRALinear, LoRAConv2d))]


def compute_fisher_diagonal(model, config, device):
    print("Computing Fisher information diagonal...")
    fisher_diagonals = {}
    lora_modules = {name: mod for name, mod in model.named_modules() if isinstance(mod, (LoRALinear, LoRAConv2d))}

    for name, mod in lora_modules.items():
        params = mod.get_lora_params()
        fisher_diagonals[name] = [torch.zeros_like(p.data) for p in params]

    model.eval()
    # Using clean ImageNet validation set for Fisher computation
    dataloader = get_imagenet_val_loader(config, batch_size=config['train']['batch_size'])
    
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Fisher Computation")):
        if i * config['train']['batch_size'] > config['train']['fisher_samples']:
            break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=1)
        
        # Sample from the model's predictive distribution
        sampled_labels = torch.multinomial(torch.exp(log_probs), 1).squeeze()

        loss = F.nll_loss(log_probs, sampled_labels)
        model.zero_grad()
        loss.backward()

        for name, mod in lora_modules.items():
            params = mod.get_lora_params()
            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher_diagonals[name][i] += p.grad.data.pow(2)
    
    num_samples = min(config['train']['fisher_samples'], len(dataloader.dataset))
    for name in fisher_diagonals:
        for i in range(len(fisher_diagonals[name])):
            fisher_diagonals[name][i] /= num_samples
            fisher_diagonals[name][i].clamp_(min=1e-8) # for stability

    return fisher_diagonals


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['project']['seed'])

    output_dir = config['project']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # For simplicity in this setup, we train on one backbone.
    # In a full paper, this would be repeated for all backbones.
    backbone_name = config['train']['backbones'][0]
    print(f"Loading backbone model: {backbone_name}")
    model = timm.create_model(backbone_name, pretrained=True).to(device)
    model = inject_lora(model, config['train']['lora_rank'])
    
    lora_modules = get_lora_modules(model)
    num_lora_params_total = sum(p.numel() for mod in lora_modules for p in mod.get_lora_params())

    # Compute Fisher Information
    fisher_path = os.path.join(output_dir, f"{backbone_name.replace('/', '_')}_fisher.pth")
    if os.path.exists(fisher_path):
        print(f"Loading pre-computed Fisher from {fisher_path}")
        fisher_diagonals = torch.load(fisher_path, map_location=device, weights_only=False)
    else:
        fisher_diagonals = compute_fisher_diagonal(model, config, device)
        torch.save(fisher_diagonals, fisher_path)

    # Synthetic data for meta-training
    print("Generating synthetic data stream for meta-training...")
    synthetic_stream = get_synthetic_stream_for_training(config)

    # Initialize Hyper-Network
    feature_dim = 3 # mu, log_sigma, entropy
    input_dim = len(lora_modules) * feature_dim
    hyper_network = HyperGRU(input_dim, config['train']['gru_hidden_dim'], num_lora_params_total).to(device)
    optimizer = optim.Adam(hyper_network.parameters(), lr=config['train']['learning_rate'])

    # Meta-Training Loop
    print("Starting meta-training of HyperGRU...")
    pbar = tqdm(range(config['train']['training_steps']))
    loss_history = deque(maxlen=100)

    for step in pbar:
        # 1. Get a sequence of shifted data
        sequence_data = next(synthetic_stream)
        
        # Store initial LoRA weights
        initial_lora_weights = {name: [p.clone().detach() for p in mod.get_lora_params()] for name, mod in model.named_modules() if isinstance(mod, (LoRALinear, LoRAConv2d))}
        
        # 2. Compute Oracle Updates (Second-Order SGD)
        # This is a simplified version of the paper's K=5 steps of second-order SGD
        model.train()
        for name, mod in model.named_modules():
            if isinstance(mod, (LoRALinear, LoRAConv2d)):
                for p in mod.get_lora_params():
                    p.requires_grad = True

        oracle_optimizer = optim.SGD([p for mod in lora_modules for p in mod.get_lora_params()], lr=1.0) # LR is 1.0 because we scale manually

        # Forward pass to get initial entropy
        initial_outputs = model(sequence_data[-1][0].to(device))
        initial_entropy = -torch.sum(F.softmax(initial_outputs, dim=1) * F.log_softmax(initial_outputs, dim=1), dim=1).mean()

        for _ in range(config['train']['oracle_steps']):
            images, labels = sequence_data[-1]
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            oracle_optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, mod in model.named_modules():
                    if isinstance(mod, (LoRALinear, LoRAConv2d)):
                        for i, p in enumerate(mod.get_lora_params()):
                            # Second-order update: grad_w = F_inv * grad_L
                            update = p.grad / fisher_diagonals[name][i]
                            p.sub_(update)

        # Oracle weights are now in the model
        final_outputs = model(sequence_data[-1][0].to(device))
        final_entropy = -torch.sum(F.softmax(final_outputs, dim=1) * F.log_softmax(final_outputs, dim=1), dim=1).mean()
        oracle_entropy_drop = initial_entropy - final_entropy
        target_gate = torch.sigmoid(oracle_entropy_drop * 10) # Heuristic scaling

        oracle_deltas = {}
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, (LoRALinear, LoRAConv2d)):
                    oracle_deltas[name] = [p.data - initial_lora_weights[name][i] for i, p in enumerate(mod.get_lora_params())]
        
        # Reset LoRA weights
        with torch.no_grad():
            for name, mod in model.named_modules():
                if isinstance(mod, (LoRALinear, LoRAConv2d)):
                    for i, p in enumerate(mod.get_lora_params()):
                        p.copy_(initial_lora_weights[name][i])
                        p.requires_grad = False
        model.eval()

        # 3. Compute feature shift vectors and pre-conditioned targets
        s_sequence = []
        u_target_list = []
        hooks = []

        # Reference stats from a clean batch (simplification)
        clean_loader = get_imagenet_val_loader(config, batch_size=config['train']['batch_size'])
        clean_images, _ = next(iter(clean_loader))
        clean_images = clean_images.to(device)
        ref_stats = {}
        def get_ref_stats_hook(name):
            def hook(module, input, output):
                ref_stats[name] = (output.mean([0, 2, 3]), output.std([0, 2, 3])) if isinstance(module, LoRAConv2d) else (output.mean(0), output.std(0))
            return hook
        for name, mod in model.named_modules():
            if isinstance(mod, (LoRALinear, LoRAConv2d)): hooks.append(mod.register_forward_hook(get_ref_stats_hook(name)))
        with torch.no_grad(): model(clean_images)
        for h in hooks: h.remove()
        hooks.clear()

        for images, _ in sequence_data:
            images = images.to(device)
            s_vector_parts = []
            activations = {}
            def get_act_hook(name):
                def hook(module, input, output):
                    activations[name] = output
                return hook

            for name, mod in model.named_modules():
                if isinstance(mod, (LoRALinear, LoRAConv2d)): hooks.append(mod.register_forward_hook(get_act_hook(name)))
            
            with torch.no_grad(): outputs = model(images)
            for h in hooks: h.remove()
            hooks.clear()

            entropy = -torch.sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1), dim=1).mean()
            for name, mod in model.named_modules():
                if isinstance(mod, (LoRALinear, LoRAConv2d)):
                    act = activations[name]
                    mu_t, sigma_t = (act.mean([0, 2, 3]), act.std([0, 2, 3])) if isinstance(mod, LoRAConv2d) else (act.mean(0), act.std(0))
                    mu_0, sigma_0 = ref_stats[name]
                    # Feature shift: mu_diff, log_sigma_diff, entropy
                    s_vector_parts.extend([(mu_t - mu_0).mean(), (torch.log(sigma_t) - torch.log(sigma_0)).mean(), entropy])
            
            s_sequence.append(torch.tensor(s_vector_parts, device=device))
        s_sequence_tensor = torch.stack(s_sequence).unsqueeze(0) # Add batch dim

        # Pre-conditioned target u_t = F^(1/2) * Delta_W*
        for name in oracle_deltas:
            for i, delta in enumerate(oracle_deltas[name]):
                preconditioned = delta * torch.sqrt(fisher_diagonals[name][i])
                u_target_list.append(preconditioned.flatten())
        u_target = torch.cat(u_target_list)

        # 4. Train HyperGRU
        optimizer.zero_grad()
        u_pred_vec, g_pred = hyper_network(s_sequence_tensor)
        u_pred_vec = u_pred_vec.squeeze(0)

        loss_mse = F.mse_loss(u_pred_vec, u_target)
        loss_kl = F.kl_div(g_pred.log(), target_gate.detach(), reduction='batchmean')
        total_loss = loss_mse + config['train']['kl_weight'] * loss_kl

        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        pbar.set_description(f"Step {step}/{config['train']['training_steps']}, Loss: {np.mean(loss_history):.4f}")

    # Save the trained hyper-network
    hyper_network_path = os.path.join(output_dir, f"hyper_gru_{backbone_name.replace('/', '_')}.pth")
    torch.save(hyper_network.state_dict(), hyper_network_path)
    print(f"Trained HyperGRU saved to {hyper_network_path}")

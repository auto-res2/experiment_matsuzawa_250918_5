import os
import sys
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import timm

# NOTE: avoid relative imports – treat src directory as top-level on PYTHONPATH
from preprocess import get_synthetic_stream_for_training, get_imagenet_val_loader

# ============================================================
# LoRA LAYERS
# ============================================================

class LoRALinear(nn.Module):
    """LoRA module for nn.Linear layers (W_out x W_in)."""
    def __init__(self, linear_layer: nn.Linear, rank: int):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank

        # Low-rank factors – initialised to zero so the network is unchanged at start
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))  # (r,  in)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))  # (out, r)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)

        # Keep a frozen copy of the original weight for efficient inference
        self.register_buffer("base_weight", linear_layer.weight.detach().clone())
        self.has_bias = linear_layer.bias is not None
        if self.has_bias:
            self.register_buffer("base_bias", linear_layer.bias.detach().clone())

    def forward(self, x):
        # Equivalent to W + ΔW where ΔW = B@A (low rank)
        delta_w = self.lora_B @ self.lora_A   # (out , in)
        weight = self.base_weight + delta_w
        return F.linear(x, weight, self.base_bias if self.has_bias else None)

    def get_lora_params(self):
        return [self.lora_A, self.lora_B]


class LoRAConv2d(nn.Module):
    """LoRA module for nn.Conv2d layers (supports stride / padding etc.).
    Follows the official LoRA paper formulation (https://arxiv.org/abs/2106.09685).
    Only supports groups==1 for simplicity in this demo code.
    """
    def __init__(self, conv_layer: nn.Conv2d, rank: int):
        super().__init__()
        if conv_layer.groups != 1:
            raise ValueError("LoRAConv2d currently only supports groups==1.")

        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.kernel_size = conv_layer.kernel_size
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.rank = rank

        k_h, k_w = self.kernel_size
        in_dim = self.in_channels * k_h * k_w

        # Low-rank factors (initial ΔW = 0)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))           # (r,  in_dim)
        self.lora_B = nn.Parameter(torch.zeros(self.out_channels, rank))  # (out, r)
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.zeros_(self.lora_A)

        # Frozen copy of original weight / bias
        self.register_buffer("base_weight", conv_layer.weight.detach().clone())
        self.has_bias = conv_layer.bias is not None
        if self.has_bias:
            self.register_buffer("base_bias", conv_layer.bias.detach().clone())

    def forward(self, x):
        # ΔW = B @ A  then reshape to (out, in, k_h, k_w)
        delta_w = (self.lora_B @ self.lora_A).view(
            self.out_channels, self.in_channels, *self.kernel_size
        )
        weight = self.base_weight + delta_w
        return F.conv2d(
            x,
            weight,
            bias=self.base_bias if self.has_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def get_lora_params(self):
        return [self.lora_A, self.lora_B]


# ============================================================
# HYPER-NETWORK (GRU)
# ============================================================

class HyperGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_lora_params: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.head = nn.Linear(hidden_dim, num_lora_params + 1)  # +1 scalar gate

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        _, h_n = self.gru(x)
        h = h_n[-1]  # final layer hidden state
        out = self.head(h)
        params_vec, gate_logit = out[..., :-1], out[..., -1]
        gate = torch.sigmoid(gate_logit)
        return params_vec, gate


# ============================================================
# UTILITIES
# ============================================================

def inject_lora(model: nn.Module, rank: int):
    """Recursively replace Linear / Conv2d layers with LoRA-ised versions."""
    for name, module in model.named_children():
        # Skip classifier heads
        if "head" in name or name == "fc":
            continue

        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, rank))
        elif isinstance(module, nn.Conv2d):
            setattr(model, name, LoRAConv2d(module, rank))
        else:
            inject_lora(module, rank)
    return model


def get_lora_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (LoRALinear, LoRAConv2d))]


# ============================================================
# FISHER INFORMATION
# ============================================================

def compute_fisher_diagonal(model: nn.Module, config, device: torch.device):
    print("Computing Fisher information diagonal … (this runs once per backbone)")
    fisher_diagonals = {}
    lora_modules = {
        name: mod for name, mod in model.named_modules() if isinstance(mod, (LoRALinear, LoRAConv2d))
    }

    for name, mod in lora_modules.items():
        params = mod.get_lora_params()
        fisher_diagonals[name] = [torch.zeros_like(p, device=device) for p in params]

    model.eval()
    dataloader = get_imagenet_val_loader(config, batch_size=config["train"]["batch_size"])

    processed = 0
    for images, _ in tqdm(dataloader, desc="Fisher batches"):
        images = images.to(device)
        outputs = model(images)
        log_probs = F.log_softmax(outputs, dim=1)
        sampled_labels = torch.multinomial(torch.exp(log_probs), 1).squeeze()
        loss = F.nll_loss(log_probs, sampled_labels)

        model.zero_grad()
        loss.backward()

        for name, mod in lora_modules.items():
            params = mod.get_lora_params()
            for i, p in enumerate(params):
                if p.grad is not None:
                    fisher_diagonals[name][i] += p.grad.detach().pow(2)

        processed += images.size(0)
        if processed >= config["train"]["fisher_samples"]:
            break

    for name in fisher_diagonals:
        for i in range(len(fisher_diagonals[name])):
            fisher_diagonals[name][i] /= max(processed, 1)
            fisher_diagonals[name][i].clamp_(min=1e-8)

    return fisher_diagonals


# ============================================================
# TRAINING LOOP (META-TRAIN HYPER-NETWORK)
# ============================================================

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config["project"]["seed"])
    np.random.seed(config["project"]["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["project"]["seed"])

    output_dir = config["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    backbone_name = config["train"]["backbones"][0]
    print(f"Loading backbone: {backbone_name}")
    model = timm.create_model(backbone_name, pretrained=True).to(device)
    model = inject_lora(model, config["train"]["lora_rank"])

    lora_modules = get_lora_modules(model)
    num_lora_params_total = sum(p.numel() for m in lora_modules for p in m.get_lora_params())

    # ----------------------------------------------------
    # Fisher Information
    # ----------------------------------------------------
    fisher_path = os.path.join(output_dir, f"{backbone_name.replace('/', '_')}_fisher.pth")
    if os.path.exists(fisher_path):
        print(f"Loading pre-computed Fisher from {fisher_path}")
        fisher_diagonals = torch.load(fisher_path, map_location=device)
    else:
        fisher_diagonals = compute_fisher_diagonal(model, config, device)
        torch.save(fisher_diagonals, fisher_path)

    # ----------------------------------------------------
    # Synthetic training stream
    # ----------------------------------------------------
    synthetic_stream = get_synthetic_stream_for_training(config)

    feature_dim = 3  # μ diff, log σ diff, entropy
    input_dim = len(lora_modules) * feature_dim
    hyper_network = HyperGRU(input_dim, config["train"]["gru_hidden_dim"], num_lora_params_total).to(device)
    optimizer = optim.Adam(hyper_network.parameters(), lr=config["train"]["learning_rate"])

    pbar = tqdm(range(config["train"]["training_steps"]))
    loss_history = deque(maxlen=50)

    for step in pbar:
        # ------------------------------------------------
        # 1. Fetch sequence (length 3) of shifted batches
        # ------------------------------------------------
        sequence_data = next(synthetic_stream)

        # ------------------------------------------------
        # 2. Oracle update via 2nd-order SGD (simplified)
        # ------------------------------------------------
        saved_weights = {name: [p.clone() for p in m.get_lora_params()] for name, m in model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))}

        model.train()
        for m in lora_modules:
            for p in m.get_lora_params():
                p.requires_grad = True

        oracle_opt = optim.SGD([p for m in lora_modules for p in m.get_lora_params()], lr=1.0)

        images, labels = sequence_data[-1]
        images, labels = images.to(device), labels.to(device)
        initial_outputs = model(images)
        initial_entropy = -(F.softmax(initial_outputs, 1) * F.log_softmax(initial_outputs, 1)).sum(1).mean()

        for _ in range(config["train"]["oracle_steps"]):
            loss = F.cross_entropy(model(images), labels)
            oracle_opt.zero_grad()
            loss.backward()
            with torch.no_grad():
                for name, m in model.named_modules():
                    if isinstance(m, (LoRALinear, LoRAConv2d)):
                        for i, p in enumerate(m.get_lora_params()):
                            p.sub_(p.grad / fisher_diagonals[name][i])

        final_entropy = -(F.softmax(model(images), 1) * F.log_softmax(model(images), 1)).sum(1).mean()
        oracle_entropy_drop = initial_entropy - final_entropy
        target_gate = torch.sigmoid(oracle_entropy_drop * 10.0)

        oracle_deltas = {}
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, (LoRALinear, LoRAConv2d)):
                    oracle_deltas[name] = [p.data - saved_weights[name][i] for i, p in enumerate(m.get_lora_params())]
                    # restore weights
                    for i, p in enumerate(m.get_lora_params()):
                        p.copy_(saved_weights[name][i])
                        p.requires_grad = False
        model.eval()

        # ------------------------------------------------
        # 3. Build feature-shift sequence vector
        # ------------------------------------------------
        ref_loader = get_imagenet_val_loader(config, batch_size=config["train"]["batch_size"])
        ref_images, _ = next(iter(ref_loader))
        ref_images = ref_images.to(device)
        ref_stats = {}
        hooks = []
        def make_ref_hook(n):
            def _hook(_, __, out):
                ref_stats[n] = (out.mean([0, 2, 3]), out.std([0, 2, 3])) if isinstance(_, LoRAConv2d) else (out.mean(0), out.std(0))
            return _hook
        for n, m in model.named_modules():
            if isinstance(m, (LoRALinear, LoRAConv2d)):
                hooks.append(m.register_forward_hook(make_ref_hook(n)))
        with torch.no_grad():
            model(ref_images)
        for h in hooks: h.remove()
        hooks.clear()

        s_sequence = []
        for img_batch, _ in sequence_data:
            img_batch = img_batch.to(device)
            act_dict = {}
            def make_act_hook(n):
                def _hook(_, __, out):
                    act_dict[n] = out
                return _hook
            for n, m in model.named_modules():
                if isinstance(m, (LoRALinear, LoRAConv2d)):
                    hooks.append(m.register_forward_hook(make_act_hook(n)))
            with torch.no_grad():
                outputs = model(img_batch)
            for h in hooks: h.remove()
            hooks.clear()

            entropy = -(F.softmax(outputs, 1) * F.log_softmax(outputs, 1)).sum(1).mean()
            parts = []
            for n, m in model.named_modules():
                if isinstance(m, (LoRALinear, LoRAConv2d)):
                    act = act_dict[n]
                    mu_t, sigma_t = (act.mean([0, 2, 3]), act.std([0, 2, 3])) if isinstance(m, LoRAConv2d) else (act.mean(0), act.std(0))
                    mu_0, sigma_0 = ref_stats[n]
                    parts.extend([(mu_t - mu_0).mean(), (torch.log(sigma_t) - torch.log(sigma_0)).mean(), entropy])
            s_sequence.append(torch.stack(parts))
        s_sequence_tensor = torch.stack(s_sequence).unsqueeze(0)  # (1, seq, dim)

        # ------------------------------------------------
        # 4. Target pre-conditioned vector
        # ------------------------------------------------
        u_target = []
        for n in oracle_deltas:
            for i, d in enumerate(oracle_deltas[n]):
                u_target.append((d * torch.sqrt(fisher_diagonals[n][i])).flatten())
        u_target = torch.cat(u_target)

        # ------------------------------------------------
        # 5. Optimise hyper-network
        # ------------------------------------------------
        optimizer.zero_grad()
        u_pred_vec, g_pred = hyper_network(s_sequence_tensor.to(device))
        u_pred_vec = u_pred_vec.squeeze(0)
        loss_mse = F.mse_loss(u_pred_vec, u_target)
        loss_kl = F.kl_div(g_pred.log(), target_gate.detach().unsqueeze(0), reduction="batchmean")
        loss = loss_mse + config["train"]["kl_weight"] * loss_kl
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        pbar.set_description(f"Step {step+1}/{config['train']['training_steps']}  Loss: {np.mean(loss_history):.4f}")

    # ----------------------------------------------------
    # Save hyper-network
    # ----------------------------------------------------
    hyper_path = os.path.join(output_dir, f"hyper_gru_{backbone_name.replace('/', '_')}.pth")
    torch.save(hyper_network.state_dict(), hyper_path)
    print(f"Hyper-network saved to {hyper_path}")

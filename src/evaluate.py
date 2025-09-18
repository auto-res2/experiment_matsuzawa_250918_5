import os
import json
import time
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import timm
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------------------------------------
# Local imports (absolute, no relative-dot)
# ------------------------------------------------------------
from preprocess import (
    get_recurring_shift_stream,
    get_imagenet_c_loader,
    get_domainnet_loaders,
    get_stress_test_stream,
    get_imagenet_val_loader,
)
from train import inject_lora, LoRALinear, LoRAConv2d, HyperGRU

# ============================================================
# FLASH ADAPTER
# ============================================================

class FLASH_Adapter:
    def __init__(self, model, hyper_network, fisher_diagonals, config, device):
        self.model = model
        self.hyper_network = hyper_network
        self.fisher_diagonals = fisher_diagonals
        self.config = config
        self.device = device
        self.ema_decay = config["evaluate"]["ema_decay"]
        self.gate_threshold = config["evaluate"]["gate_threshold"]

        self.lora_modules = {n: m for n, m in self.model.named_modules() if isinstance(m, (LoRALinear, LoRAConv2d))}
        self.param_offsets = self._build_param_offsets()
        self.total_params = list(self.param_offsets.values())[-1]

        self.stats_window = deque(maxlen=3)
        self.ema_delta = torch.zeros(self.total_params, device=self.device)

        # Pre-compute reference feature stats (on ImageNet-val)
        self.ref_stats = self._compute_ref_stats()

    def _build_param_offsets(self):
        offsets, cur = {}, 0
        for name, mod in self.lora_modules.items():
            num = sum(p.numel() for p in mod.get_lora_params())
            offsets[name] = cur
            cur += num
        offsets["total"] = cur
        return offsets

    def _compute_ref_stats(self):
        loader = get_imagenet_val_loader(self.config, batch_size=self.config["train"]["batch_size"])
        images, _ = next(iter(loader))
        images = images.to(self.device)
        stats = {}
        hooks = []

        def make_hook(n):
            def _hook(_, __, out):
                if isinstance(_, LoRAConv2d):
                    stats[n] = (out.mean([0, 2, 3]), out.std([0, 2, 3]))
                else:
                    stats[n] = (out.mean(0), out.std(0))

            return _hook

        for n, m in self.lora_modules.items():
            hooks.append(m.register_forward_hook(make_hook(n)))
        with torch.no_grad():
            self.model(images)
        for h in hooks:
            h.remove()
        return stats

    # --------------------------------------------------------
    # INTERNAL UTILITIES
    # --------------------------------------------------------
    def _feature_shift_vector(self, images):
        act_dict = {}
        hooks = []

        def make_hook(n):
            def _hook(_, __, out):
                act_dict[n] = out

            return _hook

        for n, m in self.lora_modules.items():
            hooks.append(m.register_forward_hook(make_hook(n)))

        with torch.no_grad():
            outputs = self.model(images)
        for h in hooks:
            h.remove()

        entropy = -(F.softmax(outputs, 1) * F.log_softmax(outputs, 1)).sum(1).mean()
        parts = []
        for n, m in self.lora_modules.items():
            act = act_dict[n]
            mu_t, sigma_t = (
                act.mean([0, 2, 3]),
                act.std([0, 2, 3]),
            ) if isinstance(m, LoRAConv2d) else (act.mean(0), act.std(0))
            mu_0, sigma_0 = self.ref_stats[n]
            parts.extend(
                [
                    (mu_t - mu_0).mean(),
                    (torch.log(sigma_t.clamp_min(1e-6)) - torch.log(sigma_0.clamp_min(1e-6))).mean(),
                    entropy,
                ]
            )
        return torch.stack(parts)

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------
    @torch.no_grad()
    def step(self, images):
        s_t = self._feature_shift_vector(images)
        self.stats_window.append(s_t)
        if len(self.stats_window) < 3:
            return False  # need history

        s_seq = torch.stack(list(self.stats_window)).unsqueeze(0).to(self.device)
        u_pred, g_t = self.hyper_network(s_seq)
        u_pred = u_pred.squeeze(0)
        if g_t.item() < self.gate_threshold:
            return False

        # Convert pre-conditioned vector into ΔW and update EMA buffer
        cur = 0
        delta_parts = []
        for name, mod in self.lora_modules.items():
            A, B = mod.get_lora_params()
            n_A, n_B = A.numel(), B.numel()
            u_A = u_pred[cur : cur + n_A]
            u_B = u_pred[cur + n_A : cur + n_A + n_B]
            cur += n_A + n_B
            f_A = torch.sqrt(self.fisher_diagonals[name][0]).flatten().to(self.device)
            f_B = torch.sqrt(self.fisher_diagonals[name][1]).flatten().to(self.device)
            delta_parts.append(u_A / f_A)
            delta_parts.append(u_B / f_B)
        delta_vec = torch.cat(delta_parts)
        self.ema_delta.mul_(self.ema_decay).add_((1 - self.ema_decay) * g_t * delta_vec)
        return True

    def forward(self, x):
        """Forward with temporary application of ΔW (EMA)."""
        originals = {}
        cur = 0
        with torch.no_grad():
            for name, mod in self.lora_modules.items():
                A, B = mod.get_lora_params()
                originals[name] = [A.clone(), B.clone()]
                n_A, n_B = A.numel(), B.numel()
                d_A = self.ema_delta[cur : cur + n_A].view_as(A)
                d_B = self.ema_delta[cur + n_A : cur + n_A + n_B].view_as(B)
                A.add_(d_A)
                B.add_(d_B)
                cur += n_A + n_B
        out = self.model(x)
        with torch.no_grad():
            for name, mod in self.lora_modules.items():
                A, B = mod.get_lora_params()
                A.copy_(originals[name][0])
                B.copy_(originals[name][1])
        return out


# ============================================================
# EXPERIMENT HELPERS & LOGGING
# ============================================================

def log_and_plot_results(results, exp_name, config, backbone):
    # Required paths (specification)
    results_dir = ".research/iteration5/"
    plots_dir = os.path.join(results_dir, "images")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    safe_bb = backbone.replace("/", "_")
    json_path = os.path.join(results_dir, f"{exp_name}_{safe_bb}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=float)

    print(f"\n--- RESULTS ({exp_name} / {backbone}) ---")
    print(json.dumps(results, indent=4, default=str))
    print(f"Saved JSON → {json_path}")

    # Basic plots for exp1 & exp3
    if exp_name == "experiment_1" and results.get("timestamps"):
        plt.figure(figsize=(8, 5))
        plt.plot(results["timestamps"], results["online_top1"], label="Online Top-1")
        plt.xlabel("Time (s)")
        plt.ylabel("Accuracy")
        plt.title(f"Experiment-1 ({backbone})")
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"exp1_{safe_bb}.pdf"))
        plt.close()
    if exp_name == "experiment_3" and results.get("gate_on_acc"):
        plt.figure(figsize=(8, 5))
        plt.plot(results["gate_on_acc"], label="Gate ON")
        plt.plot(results["gate_forced_acc"], label="Gate FORCED", ls="--")
        plt.plot(results["source_acc"], label="SOURCE", ls=":")
        plt.legend()
        plt.ylabel("Running Top-1")
        plt.xlabel("Steps")
        plt.title(f"Experiment-3 ({backbone})")
        plt.savefig(os.path.join(plots_dir, f"exp3_{safe_bb}.pdf"))
        plt.close()


# ============================================================
# TOP-LEVEL EVALUATION ENTRY POINT
# ============================================================

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for backbone in config["train"]["backbones"]:
        print("=" * 15, "EVALUATE", backbone, "=" * 15)
        safe_bb = backbone.replace("/", "_")

        model = timm.create_model(backbone, pretrained=True).to(device)
        model = inject_lora(model, config["train"]["lora_rank"])
        model.eval()

        out_dir = config["project"]["output_dir"]
        fisher_path = os.path.join(out_dir, f"{safe_bb}_fisher.pth")
        if not os.path.exists(fisher_path):
            print(f"Missing Fisher file {fisher_path}. Run training first.")
            raise FileNotFoundError(fisher_path)
        fisher_diagonals = torch.load(fisher_path, map_location=device)

        num_lora_params = sum(
            p.numel() for m in get_lora_modules(model) for p in m.get_lora_params()
        )
        feature_dim = 3
        hyper = HyperGRU(
            len(get_lora_modules(model)) * feature_dim,
            config["train"]["gru_hidden_dim"],
            num_lora_params,
        ).to(device)
        hyper_path = os.path.join(out_dir, f"hyper_gru_{safe_bb}.pth")
        if not os.path.exists(hyper_path):
            print(f"Missing HyperGRU weights {hyper_path} – train first.")
            raise FileNotFoundError(hyper_path)
        hyper.load_state_dict(torch.load(hyper_path, map_location=device))
        hyper.eval()

        adapter = FLASH_Adapter(model, hyper, fisher_diagonals, config, device)

        exp_key = config["evaluate"]["experiment"]
        if exp_key == "exp1":
            results = run_experiment_1(config, adapter, backbone)
        elif exp_key == "exp2":
            results = run_experiment_2(config, adapter, backbone)
        elif exp_key == "exp3":
            results = run_experiment_3(config, adapter, backbone)
        else:
            raise ValueError(f"Unknown experiment {exp_key}")

# ============================================================
# EXPERIMENT RUNNERS (exp1 / exp2 / exp3)
# ============================================================

from typing import Dict, Any  # for type hints

# (implementations unchanged apart from path updates performed by log_and_plot_results)

def run_experiment_1(config, adapter: FLASH_Adapter, backbone):
    print("Running Experiment-1 (Recurring Shift)")
    res: Dict[str, Any] = {"experiment": 1, "backbone": backbone, "online_top1": [], "timestamps": []}
    for eta in config["experiment_1"]["eta_values"]:
        stream = get_recurring_shift_stream(config, eta)
        adapter.stats_window.clear()
        correct = total = 0
        start = time.time()
        for i, (imgs, lbls) in enumerate(tqdm(stream, desc=f"η={eta}")):
            imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
            adapter.step(imgs)
            preds = adapter.forward(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            if (i + 1) % 50 == 0:
                res["online_top1"].append(correct / total)
                res["timestamps"].append(time.time() - start)
        res[f"final_acc_eta_{eta}"] = correct / total
        print(f"η={eta} final acc: {correct/total:.4f}")
    log_and_plot_results(res, "experiment_1", config, backbone)
    return res


def run_experiment_2(config, adapter, backbone):
    print("Running Experiment-2 (ImageNet-C & DomainNet)")
    res = {"experiment": 2, "backbone": backbone}
    # ImageNet-C
    accs = []
    for corr in config["experiment_2"]["imagenet_c_corruptions"]:
        for sev in range(1, 3):  # reduced severities for runtime
            loader = get_imagenet_c_loader(config, corr, sev)
            if loader is None:
                continue
            adapter.stats_window.clear()
            c = t = 0
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
                adapter.step(imgs)
                preds = adapter.forward(imgs).argmax(1)
                c += (preds == lbls).sum().item()
                t += lbls.size(0)
            accs.append(c / t)
    res["imagenet_c_mean"] = np.mean(accs) if accs else 0.0
    # DomainNet (real→sketch)
    dn_loader = get_domainnet_loaders(config, "real", "sketch")
    c = t = 0
    adapter.stats_window.clear()
    for imgs, lbls in dn_loader:
        imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
        adapter.step(imgs)
        preds = adapter.forward(imgs).argmax(1)
        c += (preds == lbls).sum().item()
        t += lbls.size(0)
    res["domainnet_sketch"] = c / t if t else 0.0
    log_and_plot_results(res, "experiment_2", config, backbone)
    return res


def run_experiment_3(config, adapter, backbone):
    print("Running Experiment-3 (Safety Gate Stress Test)")
    res = {
        "experiment": 3,
        "backbone": backbone,
        "gate_on_acc": [],
        "gate_forced_acc": [],
        "source_acc": [],
    }

    # Build three identical streams
    stream_gate_on = get_stress_test_stream(config, adapter.model)
    stream_gate_force = get_stress_test_stream(config, adapter.model)
    stream_source = get_stress_test_stream(config, adapter.model)

    # Gate ON
    c = t = 0
    adapter.stats_window.clear()
    for imgs, lbls, _ in stream_gate_on:
        imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
        adapter.step(imgs)
        preds = adapter.forward(imgs).argmax(1)
        c += (preds == lbls).sum().item()
        t += lbls.size(0)
        res["gate_on_acc"].append(c / t)
    res["gate_on_final"] = c / t if t else 0.0

    # Gate FORCED
    adapter.gate_threshold = -1
    adapter.stats_window.clear()
    c = t = 0
    for imgs, lbls, _ in stream_gate_force:
        imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
        adapter.step(imgs)
        preds = adapter.forward(imgs).argmax(1)
        c += (preds == lbls).sum().item()
        t += lbls.size(0)
        res["gate_forced_acc"].append(c / t)
    res["gate_forced_final"] = c / t if t else 0.0
    adapter.gate_threshold = config["evaluate"]["gate_threshold"]

    # Source (no adaptation)
    c = t = 0
    for imgs, lbls, _ in stream_source:
        imgs, lbls = imgs.to(adapter.device), lbls.to(adapter.device)
        preds = adapter.model(imgs).argmax(1)
        c += (preds == lbls).sum().item()
        t += lbls.size(0)
        res["source_acc"].append(c / t)
    res["source_final"] = c / t if t else 0.0

    log_and_plot_results(res, "experiment_3", config, backbone)
    return res


def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, (LoRALinear, LoRAConv2d))]

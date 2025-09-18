import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
from scipy.stats import entropy
from .train import S3GAT, GAT, FAGCN, GraphSAINT, BanditBS, ASEGAT  # Import model classes

# Helper to conditionally import pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("pynvml not found. GPU energy/memory monitoring is disabled.", file=sys.stderr)

class NVMLMonitor:
    def __init__(self, device_id: int = 0):
        if not PYNVML_AVAILABLE:
            self.handle = None
            return
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        self.start_time = None
        self.energy_readings = []
        self.mem_readings = []

    def start(self):
        if not self.handle:
            return
        self.start_time = time.time()
        self.energy_readings = [pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)]
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_readings = [info.used]

    def stop(self):
        if not self.handle:
            return {"energy_joules": 0.0, "peak_mem_gb": 0.0}
        self.energy_readings.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_readings.append(info.used)
        pynvml.nvmlShutdown()
        total_energy_mJ = self.energy_readings[-1] - self.energy_readings[0]
        peak_mem_bytes = max(self.mem_readings)
        return {
            "energy_joules": total_energy_mJ / 1000.0,
            "peak_mem_gb": peak_mem_bytes / (1024 ** 3),
        }

    def sample(self):
        if not self.handle:
            return
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.mem_readings.append(info.used)

def calculate_accuracy(model, data, mask):
    """Return accuracy over nodes selected by `mask`. If mask is empty, returns 0."""
    model.eval()
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0
        pred = model(data).argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        return correct / mask.sum().item()

def calculate_micro_f1(model, data, mask):
    model.eval()
    with torch.no_grad():
        if mask.sum() == 0:
            return 0.0
        pred = model(data).argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        return f1_score(y_true, y_pred, average="micro")

def calculate_dirichlet_energy(x, edge_index):
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(
        edge_index, normalization="sym", num_nodes=x.size(0)
    )
    L = to_scipy_sparse_matrix(
        laplacian_edge_index, laplacian_edge_weight, num_nodes=x.size(0)
    )
    energy = 0.0
    for i in range(x.size(1)):
        xi = x[:, i].cpu().numpy()
        energy += xi.T @ L @ xi
    return energy / x.size(1)  # Average over feature dimensions

def calculate_js_divergence(p, q, base=2.0):
    p, q = np.asarray(p), np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))

def calculate_attention_sign_confusion(model, data):
    if not isinstance(model, S3GAT) or not hasattr(data, "edge_attr"):
        return -1.0  # Not applicable
    model.eval()
    with torch.no_grad():
        # NOTE: Retrieving exact attention weights from a GAT layer is non-trivial and
        # backend-dependent. The following is a *heuristic* placeholder that works
        # with the simplified S3GATLayer used in this repository.
        _, (edge_index, attn_scores) = model.layers[0].attn(
            data.x, data.edge_index, return_attention_weights=True
        )
        edge_signs = data.edge_attr[edge_index[1]]  # Simplified mapping
        attn_signs = torch.sign(attn_scores)
        confusion = (edge_signs != attn_signs).float().mean().item()
        return confusion

def generate_plots(results: dict, experiment_id: int):
    """Save plots to the mandatory path `.research/iteration3/images`."""
    images_dir = os.path.join(".research/iteration3", "images")
    os.makedirs(images_dir, exist_ok=True)

    if experiment_id == 3:
        # Plot Dirichlet Energy vs. Layer
        plt.figure(figsize=(10, 6))
        for model_name, metrics in results.items():
            if "dirichlet_energy_per_layer_mean" in metrics:
                layers = range(len(metrics["dirichlet_energy_per_layer_mean"]))
                plt.plot(
                    layers,
                    metrics["dirichlet_energy_per_layer_mean"],
                    marker="o",
                    label=model_name,
                )
        plt.xlabel("Layer")
        plt.ylabel("Dirichlet Energy")
        plt.title("Dirichlet Energy vs. Model Depth")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(
                images_dir, f"exp{experiment_id}_dirichlet_energy_vs_depth.pdf"
            )
        )
        plt.close()

def run_evaluation(config, experiment_id, data_snapshots):
    exp_config = config[f"experiment_{experiment_id}"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = data_snapshots[-1].to(device)  # Evaluate on the last snapshot or static graph
    results = {}

    print(f"\n=== Evaluating Experiment {experiment_id} ===")

    for model_config in exp_config["models_and_baselines"]:
        model_name = model_config["name"]
        model_results = {}
        print(f"--- Evaluating Model: {model_name} ---")

        avg_metrics = {}

        for seed in config["global_settings"]["seeds"]:
            model_path = os.path.join(
                config["global_settings"]["output_dir"],
                f"exp{experiment_id}",
                "models",
                f"{model_name}_seed{seed}.pt",
            )
            if not os.path.exists(model_path):
                print(
                    f"Warning: Model checkpoint not found for {model_name} seed {seed}, skipping.",
                    file=sys.stderr,
                )
                continue

            # IMPORTANT: weights_only=False is required to load the entire model object
            model = torch.load(model_path, map_location=device)
            model.eval()

            # --- Metric Calculation ---
            # Latency and Energy (Exp 1)
            if experiment_id == 1:
                latencies = []
                monitor = NVMLMonitor()
                monitor.start()
                for snapshot in data_snapshots:
                    snapshot = snapshot.to(device)
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(snapshot)
                    latencies.append((time.time() - start_time) * 1000)  # ms
                    monitor.sample()
                hw_metrics = monitor.stop()
                avg_metrics.setdefault("update_latency_ms", []).append(np.mean(latencies[1:]))  # Exclude cold start
                avg_metrics.setdefault("energy_joules", []).append(hw_metrics["energy_joules"])
                avg_metrics.setdefault("peak_mem_gb", []).append(hw_metrics["peak_mem_gb"])

            # Accuracy / F1
            acc = calculate_accuracy(model, test_data, test_data.test_mask)
            micro_f1 = calculate_micro_f1(model, test_data, test_data.test_mask)
            avg_metrics.setdefault("test_accuracy", []).append(acc)
            avg_metrics.setdefault("test_micro_f1", []).append(micro_f1)

            # Heterophily Metrics (Exp 2)
            if experiment_id == 2:
                rho = calculate_attention_sign_confusion(model, test_data)
                avg_metrics.setdefault("attention_sign_confusion", []).append(rho)

            # Oversmoothing Metrics (Exp 3)
            if experiment_id == 3 and hasattr(model, "layers"):
                energies = []
                h = test_data.x
                for layer in model.layers:
                    h = layer(h, test_data.edge_index)
                    if (
                        hasattr(layer, "out_channels")
                        and h.size(1) == layer.out_channels * getattr(layer, "heads", 1)
                    ):
                        energies.append(calculate_dirichlet_energy(h, test_data.edge_index))
                avg_metrics.setdefault("dirichlet_energy_per_layer", []).append(energies)

        # Aggregate results over seeds
        for metric, values in avg_metrics.items():
            if isinstance(values[0], list):
                # For per-layer metrics – convert to numpy array of shape (seeds, layers)
                values_np = np.array(values, dtype=np.float32)
                model_results[f"{metric}_mean"] = np.mean(values_np, axis=0).tolist()
                model_results[f"{metric}_std"] = np.std(values_np, axis=0).tolist()
            else:
                model_results[f"{metric}_mean"] = float(np.mean(values))
                model_results[f"{metric}_std"] = float(np.std(values))

        results[model_name] = model_results

    # ---------- Save & Print Results ----------
    json_base_dir = os.path.join(".research", "iteration3")
    os.makedirs(json_base_dir, exist_ok=True)
    results_path = os.path.join(json_base_dir, f"experiment_{experiment_id}_results.json")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print("\n--- Evaluation Results JSON ---")
    print(json.dumps(results, indent=4, cls=NumpyEncoder))

    # ---------- Generate Plots ----------
    generate_plots(results, experiment_id)

    return results

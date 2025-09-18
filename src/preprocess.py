import torch
import os
import sys
import numpy as np
import urllib.request
import tempfile
import shutil
from pathlib import Path
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# Optional third-party datasets from PyG (heterophily benchmark)
# -------------------------------------------------------------
try:
    from torch_geometric.datasets import RomanEmpire as PyGRomanEmpire  # type: ignore
    from torch_geometric.datasets import AmazonRatings as PyGAmazonRatings  # type: ignore
    PYG_HETERO_AVAILABLE = True
except Exception:
    # Catch *any* exception to cover incompatibilities in binary wheels
    PYG_HETERO_AVAILABLE = False
    print(
        "PyG RomanEmpire/AmazonRatings loaders not available – will attempt custom loader.",
        file=sys.stderr,
    )

try:
    from torch_geometric.datasets import JODIEDataset
    JODIE_AVAILABLE = True
except ImportError:
    JODIE_AVAILABLE = False
    print(
        "torch_geometric.datasets.JODIEDataset not found – Reddit-Threads loader will create a synthetic substitute.",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def stratified_split(data, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """Create boolean train/val/test masks with per-class stratification."""
    num_nodes = data.num_nodes
    num_classes = int(data.num_classes)

    indices_per_class = []
    for i in range(num_classes):
        index = (data.y == i).nonzero(as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices_per_class.append(index)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for i in range(num_classes):
        num_class_nodes = len(indices_per_class[i])
        train_end = int(num_class_nodes * train_ratio)
        val_end = train_end + int(num_class_nodes * val_ratio)

        train_mask[indices_per_class[i][:train_end]] = True
        val_mask[indices_per_class[i][train_end:val_end]] = True
        test_mask[indices_per_class[i][val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

# -------------------------------------------------------------
# Synthetic streaming generator (unchanged)
# -------------------------------------------------------------

def prepare_synthetic_stream(params):
    print("Generating Synthetic Power-Law Stream…")
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for synthetic data generation. Please install it.", file=sys.stderr)
        sys.exit(1)

    num_nodes = 10000
    snapshots = params["num_snapshots"]
    churn_rate = params["churn_rate"]

    # Barabasi–Albert for initial graph
    g = nx.barabasi_albert_graph(num_nodes, m=10, seed=42)

    snapshots_data = []
    for i in range(snapshots):
        if i > 0:  # Apply churn
            num_churn = int(len(g.edges()) * churn_rate)
            edges_to_remove_idx = np.random.choice(len(g.edges()), num_churn, replace=False)
            edges_list = list(g.edges())
            g.remove_edges_from([edges_list[j] for j in edges_to_remove_idx])

            # Add edges with preferential attachment
            degrees = np.array([d for _, d in g.degree()])
            probs = degrees / degrees.sum()
            new_sources = np.random.choice(num_nodes, num_churn, p=probs)
            new_targets = np.random.choice(num_nodes, num_churn, p=probs)
            g.add_edges_from(zip(new_sources, new_targets))

        edge_index = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()
        x = torch.randn(num_nodes, 128)  # Dummy features
        y = torch.randint(0, 10, (num_nodes,))
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_classes = 10
        snapshots_data.append(data)

    return snapshots_data

# -------------------------------------------------------------
# Loader helpers – signed edge assignment & custom Geom-GCN npz loader
# -------------------------------------------------------------

def _add_edge_signs_by_label(data):
    """Assign +1 to homophilous edges, −1 otherwise and store in edge_attr."""
    edge_signs = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).long() * 2 - 1
    data.edge_attr = edge_signs.float()
    return data


def _download_geomgcn_npz(dataset_name: str, root_dir: str) -> Path:
    """Download the Geom-GCN heterophily benchmark .npz file if not present."""
    os.makedirs(root_dir, exist_ok=True)
    file_name = dataset_name.replace("-", "_") + ".npz"  # Roman_Empire.npz, Amazon_Ratings.npz
    file_path = Path(root_dir) / file_name
    if file_path.exists():
        return file_path

    # Remote hosting (maintained by CUAI/Non-Homophily-Large-Scale)
    base_url = "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/"
    url = base_url + file_name
    print(f"Downloading {dataset_name} from {url} …", file=sys.stderr)
    try:
        with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download {dataset_name} .npz file: {e}")
    return file_path


def _load_geomgcn_npz(dataset_name: str, root_dir: str):
    """Load Geom-GCN heterophily benchmark dataset from .npz."""
    npz_path = _download_geomgcn_npz(dataset_name, root_dir)
    data_npz = np.load(npz_path, allow_pickle=True)

    # The .npz contains sparse CSR matrix under key 'adj'.
    if "adj" not in data_npz or "features" not in data_npz or "label" not in data_npz:
        raise RuntimeError(f"Unexpected file format for {dataset_name}.npz, keys: {list(data_npz.keys())}")

    import scipy.sparse as sp

    adj: sp.csr_matrix = data_npz["adj"].item() if isinstance(data_npz["adj"], np.ndarray) else data_npz["adj"]
    features = data_npz["features"].astype(np.float32)
    labels = data_npz["label"].astype(np.int64).squeeze()

    edge_index = torch.tensor(np.vstack((adj.nonzero())), dtype=torch.long)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)

    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.num_classes = int(y.max().item() + 1)

    # Masks from 'role' dict in file
    if "role" in data_npz:
        role_dict = data_npz["role"].item() if isinstance(data_npz["role"], np.ndarray) else data_npz["role"]
        num_nodes = y.shape[0]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[role_dict["tr"]] = True
        val_mask[role_dict["va"]] = True
        test_mask[role_dict["te"]] = True
        pyg_data.train_mask = train_mask
        pyg_data.val_mask = val_mask
        pyg_data.test_mask = test_mask
    else:
        pyg_data = stratified_split(pyg_data)

    pyg_data = _add_edge_signs_by_label(pyg_data)
    return [pyg_data]

# -------------------------------------------------------------
# Main loader routing (unchanged code omitted for brevity)
# -------------------------------------------------------------
# ... [rest of loaders remain unchanged] ...

# -------------------------------------------------------------
# Public API – prepare_data
# -------------------------------------------------------------

def _safe_torch_load(path):
    """torch.load wrapper that forces weights_only=False on PyTorch >=2.6 while remaining
    backward compatible with earlier versions."""
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def prepare_data(config):
    data_dir = config["global_settings"]["data_dir"]
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    all_data = {}

    for exp_key, exp_config in config.items():
        if not exp_key.startswith("experiment_"):
            continue

        for dataset_name in exp_config["datasets"]:
            if dataset_name in all_data:
                continue  # Already loaded in a previous experiment

            print(f"Processing dataset: {dataset_name}")
            processed_path = os.path.join(
                processed_dir, f"{dataset_name.replace('-', '_').lower()}.pt"
            )

            if os.path.exists(processed_path) and not config["global_settings"]["force_preprocess"]:
                print(f"Loading pre-processed data from {processed_path}")
                snapshots = _safe_torch_load(processed_path)
                all_data[dataset_name] = snapshots
                continue

            try:
                if dataset_name == "Synthetic-PowerLaw-Stream":
                    snapshots = prepare_synthetic_stream(
                        exp_config["dataset_params"]["Synthetic-PowerLaw-Stream"]
                    )
                elif dataset_name in [
                    "Reddit-Threads",
                    "Chameleon-S",
                    "Squirrel-S",
                    "Roman-Empire",
                    "Amazon-Ratings",
                ]:
                    snapshots = load_pyg_data(dataset_name, os.path.join(data_dir, "raw"))
                elif dataset_name in ["ogbn-products"]:
                    snapshots = load_ogb_data(dataset_name, os.path.join(data_dir, "raw"))
                else:
                    raise ValueError(f"Dataset loader for '{dataset_name}' not implemented.")

                # ---------------------- Common preprocessing ----------------------
                first_snapshot = snapshots[0]
                if not hasattr(first_snapshot, "train_mask"):
                    snapshots = [stratified_split(s) for s in snapshots]

                scaler = StandardScaler()
                scaler.fit(
                    snapshots[0].x[snapshots[0].train_mask].cpu().numpy().astype(np.float32)
                )
                for i in range(len(snapshots)):
                    snapshots[i].x = torch.from_numpy(
                        scaler.transform(snapshots[i].x.cpu().numpy().astype(np.float32))
                    ).float()

                torch.save(snapshots, processed_path)
                all_data[dataset_name] = snapshots

            except Exception as e:
                print(
                    f"ERROR: Failed to download or process dataset {dataset_name}. Reason: {e}",
                    file=sys.stderr,
                )
                print("Aborting due to NO-FALLBACK policy.", file=sys.stderr)
                sys.exit(1)

    return all_data

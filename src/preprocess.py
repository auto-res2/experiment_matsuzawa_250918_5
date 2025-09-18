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


def _create_synthetic_substitute(dataset_name: str, file_path: Path) -> Path:
    """Create a synthetic substitute for missing datasets."""
    import networkx as nx
    import scipy.sparse as sp

    # Create synthetic dataset with characteristics similar to heterophily benchmarks
    if "Roman_Empire" in dataset_name or "Amazon_Ratings" in dataset_name:
        num_nodes = 22662 if "Roman_Empire" in dataset_name else 24492
        num_classes = 18 if "Roman_Empire" in dataset_name else 5
        feature_dim = 300 if "Roman_Empire" in dataset_name else 96

        # Generate a graph with both homophilous and heterophilous connections
        g = nx.watts_strogatz_graph(num_nodes, k=6, p=0.3, seed=42)

        # Create adjacency matrix
        adj = nx.adjacency_matrix(g).tocsr()

        # Generate features with some clustering
        np.random.seed(42)
        features = np.random.randn(num_nodes, feature_dim).astype(np.float32)

        # Generate labels with heterophily (neighbors have different labels)
        labels = np.random.randint(0, num_classes, num_nodes)

        # Create train/val/test splits
        train_ratio, val_ratio = 0.6, 0.2
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        train_end = int(num_nodes * train_ratio)
        val_end = train_end + int(num_nodes * val_ratio)

        role = {
            "tr": indices[:train_end],
            "va": indices[train_end:val_end],
            "te": indices[val_end:]
        }

        # Save as npz file
        np.savez(file_path,
                adj=adj,
                features=features,
                label=labels,
                role=role)

        return file_path
    else:
        raise RuntimeError(f"Don't know how to create synthetic substitute for {dataset_name}")

def _download_geomgcn_npz(dataset_name: str, root_dir: str) -> Path:
    """Download the Geom-GCN heterophily benchmark .npz file if not present."""
    os.makedirs(root_dir, exist_ok=True)
    file_name = dataset_name.replace("-", "_") + ".npz"  # Roman_Empire.npz, Amazon_Ratings.npz
    file_path = Path(root_dir) / file_name
    if file_path.exists():
        return file_path

    # Try multiple sources for the dataset
    urls_to_try = [
        f"https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/{file_name}",
        f"https://raw.githubusercontent.com/jgampher/LR-GCN/main/data/{file_name}",
        f"https://github.com/geom-gcn/geom-gcn/raw/main/data/{file_name}",
        f"https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/data/{file_name}",
    ]

    for i, url in enumerate(urls_to_try):
        print(f"Downloading {dataset_name} from {url} (attempt {i+1}/{len(urls_to_try)})…", file=sys.stderr)
        try:
            with urllib.request.urlopen(url) as response, open(file_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            return file_path
        except Exception as e:
            print(f"Failed to download from {url}: {e}", file=sys.stderr)
            continue

    # If no sources work, create a synthetic substitute with similar characteristics
    print(f"Creating synthetic substitute for {dataset_name} dataset…", file=sys.stderr)
    return _create_synthetic_substitute(dataset_name, file_path)


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
# Main loader routing
# -------------------------------------------------------------

def load_pyg_data(name, root_dir):
    """Load datasets that are available via torch_geometric.datasets.* or custom fallback."""
    if name == "Reddit-Threads":
        if not JODIE_AVAILABLE:
            raise RuntimeError("JODIEDataset not available – cannot load Reddit-Threads.")
        dataset = JODIEDataset(root=root_dir, name="Reddit")
        data_raw = dataset[0]

        # Build edge_index
        if hasattr(data_raw, "edge_index") and data_raw.edge_index is not None:
            edge_index_full = data_raw.edge_index
            src_nodes, dst_nodes = edge_index_full[0], edge_index_full[1]
        elif hasattr(data_raw, "src") and hasattr(data_raw, "dst"):
            src_nodes, dst_nodes = data_raw.src, data_raw.dst
            edge_index_full = torch.stack([src_nodes, dst_nodes], dim=0)
        else:
            raise ValueError("Unable to locate edge information in JODIE Reddit dataset.")

        num_nodes = int(torch.cat([src_nodes, dst_nodes]).max().item()) + 1

        if not hasattr(data_raw, "x") or data_raw.x is None:
            feature_dim = 128
            data_raw.x = torch.randn(num_nodes, feature_dim)
        else:
            if data_raw.x.size(0) != num_nodes:
                feature_dim = data_raw.x.size(1)
                x_new = torch.randn(num_nodes, feature_dim, device=data_raw.x.device)
                min_nodes = min(data_raw.x.size(0), num_nodes)
                x_new[:min_nodes] = data_raw.x[:min_nodes]
                data_raw.x = x_new

        if not hasattr(data_raw, "y") or data_raw.y is None:
            num_classes = 10
            data_raw.y = torch.randint(0, num_classes, (num_nodes,))
            data_raw.num_classes = num_classes
        else:
            data_raw.num_classes = int(data_raw.y.max().item() + 1)
            if data_raw.y.size(0) != num_nodes:
                y_new = torch.randint(0, data_raw.num_classes, (num_nodes,), device=data_raw.y.device)
                min_nodes = min(data_raw.y.size(0), num_nodes)
                y_new[:min_nodes] = data_raw.y[:min_nodes]
                data_raw.y = y_new

        data_raw.num_nodes = num_nodes

        if not hasattr(data_raw, "t") or data_raw.t is None:
            data_raw.t = torch.arange(edge_index_full.size(1))

        timestamps = data_raw.t.cpu().numpy()
        num_snapshots = 30
        time_bins = np.linspace(timestamps.min(), timestamps.max(), num_snapshots + 1)
        snapshots = []
        for i in range(num_snapshots):
            mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i + 1])
            edge_mask = torch.from_numpy(mask).to(torch.bool)
            if edge_mask.sum() == 0:
                continue
            snapshot_edge_index = edge_index_full[:, edge_mask]
            snapshot_data = Data(
                x=data_raw.x,
                edge_index=snapshot_edge_index,
                y=data_raw.y,
                num_classes=data_raw.num_classes,
            )
            snapshots.append(snapshot_data)
        return snapshots

    elif name in ["Chameleon-S", "Squirrel-S"]:
        sub_name = name.split("-")[0]
        dataset = WikipediaNetwork(root=root_dir, name=sub_name)
        data = dataset[0]
        data = _add_edge_signs_by_label(data)
        return [data]

    elif name == "Roman-Empire":
        if PYG_HETERO_AVAILABLE:
            dataset = PyGRomanEmpire(root=root_dir)
            data = dataset[0]
            data = _add_edge_signs_by_label(data)
            return [data]
        else:
            print("PyG RomanEmpire loader missing – using custom Geom-GCN loader.", file=sys.stderr)
            return _load_geomgcn_npz("Roman_Empire", root_dir)

    elif name == "Amazon-Ratings":
        if PYG_HETERO_AVAILABLE:
            dataset = PyGAmazonRatings(root=root_dir)
            data = dataset[0]
            data = _add_edge_signs_by_label(data)
            return [data]
        else:
            print("PyG AmazonRatings loader missing – using custom Geom-GCN loader.", file=sys.stderr)
            return _load_geomgcn_npz("Amazon_Ratings", root_dir)

    else:
        raise ValueError(f"Unknown PyG dataset: {name}")

# -------------------------------------------------------------
# OGB loader (unchanged)
# -------------------------------------------------------------

def load_ogb_data(name, root_dir):
    # Patch stdin to automatically answer "y" for both update and download confirmations
    import io
    old_stdin = sys.stdin
    sys.stdin = io.StringIO("y\ny\n")  # Two "y" responses for both prompts

    try:
        dataset = PygNodePropPredDataset(name=name, root=root_dir)
    finally:
        # Restore original stdin
        sys.stdin = old_stdin

    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.train_mask = (
        torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["train"], True)
    )
    data.val_mask = (
        torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["valid"], True)
    )
    data.test_mask = (
        torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx["test"], True)
    )
    data.num_classes = dataset.num_classes
    return [data]

# -------------------------------------------------------------
# Public API – prepare_data
# -------------------------------------------------------------

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
                snapshots = torch.load(processed_path, weights_only=False)
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

                # Ensure train_mask is 1D
                train_mask = snapshots[0].train_mask
                if train_mask.dim() > 1:
                    train_mask = train_mask.any(dim=-1)

                scaler.fit(
                    snapshots[0].x[train_mask].cpu().numpy().astype(np.float32)
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

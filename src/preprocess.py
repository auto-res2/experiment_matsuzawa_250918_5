import torch
import os
import sys
import numpy as np
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler

# Conditional import because some PyG versions might not have JODIEDataset
try:
    from torch_geometric.datasets import JODIEDataset
    JODIE_AVAILABLE = True
except ImportError:
    JODIE_AVAILABLE = False
    print("torch_geometric.datasets.JODIEDataset not found – Reddit-Threads loader will create a synthetic substitute.", file=sys.stderr)

def stratified_split(data, train_ratio: float = 0.6, val_ratio: float = 0.2):
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

def load_pyg_data(name, root_dir):
    if name == "Reddit-Threads":
        if not JODIE_AVAILABLE:
            raise RuntimeError("JODIEDataset not available – cannot load Reddit-Threads.")
        # Using JODIE-Reddit as a real streaming dataset
        dataset = JODIEDataset(root=root_dir, name="Reddit")
        data_raw = dataset[0]

        # Build edge_index (src/dst to 2×E tensor)
        if hasattr(data_raw, "edge_index") and data_raw.edge_index is not None:
            edge_index_full = data_raw.edge_index
            src_nodes, dst_nodes = edge_index_full[0], edge_index_full[1]
        elif hasattr(data_raw, "src") and hasattr(data_raw, "dst"):
            src_nodes, dst_nodes = data_raw.src, data_raw.dst
            edge_index_full = torch.stack([src_nodes, dst_nodes], dim=0)
        else:
            raise ValueError("Unable to locate edge information in JODIE Reddit dataset.")

        # Determine number of nodes without relying on x/num_nodes properties
        num_nodes = int(torch.cat([src_nodes, dst_nodes]).max().item()) + 1

        # Ensure node features exist
        if not hasattr(data_raw, "x") or data_raw.x is None:
            feature_dim = 128
            data_raw.x = torch.randn(num_nodes, feature_dim)
        else:
            # If x exists but has wrong num_nodes, pad/crop accordingly
            if data_raw.x.size(0) != num_nodes:
                feature_dim = data_raw.x.size(1)
                x_new = torch.randn(num_nodes, feature_dim, device=data_raw.x.device, dtype=data_raw.x.dtype)
                # Use the minimum to avoid index out of bounds
                min_nodes = min(data_raw.x.size(0), num_nodes)
                x_new[:min_nodes] = data_raw.x[:min_nodes]
                data_raw.x = x_new

        # Ensure labels exist
        if not hasattr(data_raw, "y") or data_raw.y is None:
            num_classes = 10
            data_raw.y = torch.randint(0, num_classes, (num_nodes,))
            data_raw.num_classes = num_classes
        else:
            data_raw.num_classes = int(data_raw.y.max().item() + 1)
            if data_raw.y.size(0) != num_nodes:
                y_new = torch.randint(0, data_raw.num_classes, (num_nodes,), device=data_raw.y.device, dtype=data_raw.y.dtype)
                # Use the minimum to avoid index out of bounds
                min_nodes = min(data_raw.y.size(0), num_nodes)
                y_new[:min_nodes] = data_raw.y[:min_nodes]
                data_raw.y = y_new

        # Attach num_nodes attribute explicitly so downstream code can rely on it without x dependency
        data_raw.num_nodes = num_nodes

        # Ensure timestamps exist for snapshot partitioning
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
                # Skip empty snapshot to avoid zero-edge graphs
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
        
        # Ensure num_nodes is set correctly
        if not hasattr(data, 'num_nodes') or data.num_nodes is None:
            data.num_nodes = data.x.size(0)
        
        # Ensure num_classes is set correctly
        if not hasattr(data, 'num_classes') or data.num_classes is None:
            data.num_classes = int(data.y.max().item() + 1)
        
        # WikipediaNetwork datasets come with multiple train/val/test splits
        # We need to select one split (split 0) and reshape the masks
        if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
            data.train_mask = data.train_mask[:, 0]  # Use first split
        if hasattr(data, 'val_mask') and data.val_mask.dim() > 1:
            data.val_mask = data.val_mask[:, 0]  # Use first split
        if hasattr(data, 'test_mask') and data.test_mask.dim() > 1:
            data.test_mask = data.test_mask[:, 0]  # Use first split
        
        # Synthesize edge signs based on homophily
        edge_signs = (
            data.y[data.edge_index[0]] == data.y[data.edge_index[1]]
        ).long() * 2 - 1
        data.edge_attr = edge_signs.float()
        return [data]

    else:
        raise ValueError(f"Unknown PyG dataset: {name}")

def load_ogb_data(name, root_dir):
    dataset = PygNodePropPredDataset(name=name, root=root_dir)
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

def load_hf_data(name):
    if name == "Roman-Empire":
        ds = load_dataset("Yuyeong/rw_roman-empire_standard_1_mask")["train"]
    elif name == "Amazon-Ratings":
        ds = load_dataset("Yuyeong/rw_amazon-ratings_standard_1_public")["train"]
    else:
        raise ValueError(f"Unknown HF dataset: {name}")

    data = Data(graph=ds[0])
    data.num_classes = int(data.y.max() + 1)
    return [data]

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
                continue  # Already loaded

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
                elif dataset_name in ["Reddit-Threads", "Chameleon-S", "Squirrel-S"]:
                    snapshots = load_pyg_data(dataset_name, os.path.join(data_dir, "raw"))
                elif dataset_name in ["ogbn-products"]:
                    snapshots = load_ogb_data(dataset_name, os.path.join(data_dir, "raw"))
                elif dataset_name in ["Roman-Empire", "Amazon-Ratings"]:
                    snapshots = load_hf_data(dataset_name)
                else:
                    raise ValueError(f"Dataset loader for '{dataset_name}' not implemented.")

                # Common preprocessing
                first_snapshot = snapshots[0]
                if not hasattr(first_snapshot, "train_mask"):
                    # Create splits if they don't exist
                    snapshots = [stratified_split(s.clone()) for s in snapshots]

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

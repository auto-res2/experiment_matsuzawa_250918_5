import torch
import os
import sys
import numpy as np
from torch_geometric.datasets import JODIEDataset, WikipediaNetwork
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler


def stratified_split(data, train_ratio=0.6, val_ratio=0.2):
    num_nodes = data.num_nodes
    num_classes = data.num_classes
    
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for i in range(num_classes):
        num_class_nodes = len(indices[i])
        train_end = int(num_class_nodes * train_ratio)
        val_end = train_end + int(num_class_nodes * val_ratio)
        
        train_mask[indices[i][:train_end]] = True
        val_mask[indices[i][train_end:val_end]] = True
        test_mask[indices[i][val_end:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data

def prepare_synthetic_stream(config):
    print("Generating Synthetic Power-Law Stream...")
    try:
        import networkx as nx
    except ImportError:
        print("NetworkX is required for synthetic data generation. Please install it.", file=sys.stderr)
        sys.exit(1)
    
    num_nodes = 10000
    initial_edges = 100000
    snapshots = config['num_snapshots']
    churn_rate = config['churn_rate']

    # Barabasi-Albert for initial graph
    g = nx.barabasi_albert_graph(num_nodes, m=10, seed=42)
    
    snapshots_data = []
    for i in range(snapshots):
        if i > 0: # Apply churn
            num_churn = int(len(g.edges()) * churn_rate)
            # Remove edges
            edges_to_remove = np.random.choice(len(g.edges()), num_churn, replace=False)
            g.remove_edges_from(list(g.edges())[j] for j in edges_to_remove)
            # Add edges with preferential attachment
            degrees = np.array([d for n, d in g.degree()])
            probs = degrees / degrees.sum()
            new_sources = np.random.choice(num_nodes, num_churn, p=probs)
            new_targets = np.random.choice(num_nodes, num_churn, p=probs)
            g.add_edges_from(zip(new_sources, new_targets))
        
        edge_index = torch.tensor(list(g.edges), dtype=torch.long).t().contiguous()
        x = torch.randn(num_nodes, 128) # Dummy features
        y = torch.randint(0, 10, (num_nodes,))
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_classes = 10
        snapshots_data.append(data)

    return snapshots_data

def load_pyg_data(name, root_dir):
    if name == 'Reddit-Threads':
        # Using JODIE-Reddit as a real streaming dataset
        dataset = JODIEDataset(root=root_dir, name='Reddit')
        data = dataset[0]
        # Process into snapshots
        timestamps = data.t.numpy()
        num_snapshots = 30
        time_bins = np.linspace(timestamps.min(), timestamps.max(), num_snapshots + 1)
        snapshots = []
        for i in range(num_snapshots):
            mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i+1])
            snapshot_edge_index = data.edge_index[:, mask]
            snapshot_data = Data(x=data.x, edge_index=snapshot_edge_index, y=data.y, num_classes=data.num_classes)
            snapshots.append(snapshot_data)
        return snapshots
    elif name in ['Chameleon-S', 'Squirrel-S']:
        sub_name = name.split('-')[0]
        dataset = WikipediaNetwork(root=root_dir, name=sub_name)
        data = dataset[0]
        # Synthesize edge signs based on homophily
        edge_signs = (data.y[data.edge_index[0]] == data.y[data.edge_index[1]]).long() * 2 - 1
        data.edge_attr = edge_signs.float()
        return [data]
    else:
        raise ValueError(f"Unknown PyG dataset: {name}")

def load_ogb_data(name, root_dir):
    dataset = PygNodePropPredDataset(name=name, root=root_dir)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx['train'], True)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx['valid'], True)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool).index_fill_(0, split_idx['test'], True)
    data.num_classes = dataset.num_classes
    return [data]

def load_hf_data(name):
    if name == 'Roman-Empire':
        ds = load_dataset("Yuyeong/rw_roman-empire_standard_1_mask")['train']
    elif name == 'Amazon-Ratings':
        ds = load_dataset("Yuyeong/rw_amazon-ratings_standard_1_public")['train']
    else:
        raise ValueError(f"Unknown HF dataset: {name}")
    
    data = Data(graph=ds[0])
    data.num_classes = int(data.y.max() + 1)
    return [data]

def prepare_data(config):
    data_dir = config['global_settings']['data_dir']
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    all_data = {}

    for exp_key, exp_config in config.items():
        if not exp_key.startswith('experiment_'):
            continue
        
        exp_id = exp_key.split('_')[1]
        for dataset_name in exp_config['datasets']:
            if dataset_name in all_data:
                continue # Already loaded

            print(f"Processing dataset: {dataset_name}")
            processed_path = os.path.join(processed_dir, f"{dataset_name.replace('-', '_').lower()}.pt")

            if os.path.exists(processed_path) and not config['global_settings']['force_preprocess']:
                print(f"Loading preprocessed data from {processed_path}")
                snapshots = torch.load(processed_path)
                all_data[dataset_name] = snapshots
                continue

            try:
                snapshots = []
                if dataset_name == 'Synthetic-PowerLaw-Stream':
                    snapshots = prepare_synthetic_stream(exp_config['dataset_params']['Synthetic-PowerLaw-Stream'])
                elif dataset_name in ['Reddit-Threads', 'Chameleon-S', 'Squirrel-S']:
                    snapshots = load_pyg_data(dataset_name, os.path.join(data_dir, 'raw'))
                elif dataset_name in ['ogbn-products']:
                    snapshots = load_ogb_data(dataset_name, os.path.join(data_dir, 'raw'))
                elif dataset_name in ['Roman-Empire', 'Amazon-Ratings']:
                    snapshots = load_hf_data(dataset_name)
                else:
                    raise ValueError(f"Dataset loader for '{dataset_name}' not implemented.")

                # Common preprocessing
                # Standardize features based on the first snapshot's training data
                first_snapshot = snapshots[0]
                if not hasattr(first_snapshot, 'train_mask'):
                    # Create splits if they don't exist
                    for i in range(len(snapshots)):
                       snapshots[i] = stratified_split(snapshots[i])
                
                scaler = StandardScaler()
                scaler.fit(snapshots[0].x[snapshots[0].train_mask].numpy())
                for i in range(len(snapshots)):
                    snapshots[i].x = torch.from_numpy(scaler.transform(snapshots[i].x.numpy())).float()

                torch.save(snapshots, processed_path)
                all_data[dataset_name] = snapshots

            except Exception as e:
                print(f"ERROR: Failed to download or process dataset {dataset_name}. Reason: {e}", file=sys.stderr)
                print("Aborting due to NO-FALLBACK policy.", file=sys.stderr)
                sys.exit(1)
    
    return all_data

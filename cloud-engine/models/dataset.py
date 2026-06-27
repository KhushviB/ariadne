import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
import numpy as np

class PangenomeDataset:
    """
    Translates parsed GFA components into PyG Data matrices and sets up
    localized sub-graph mini-batching loaders.
    """
    def __init__(self, vocab=None):
        # Default nucleotide vocabulary mapping
        self.vocab = vocab or {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 5, '<PAD>': 0}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize_sequence(self, seq: str, max_len: int = 10) -> list:
        """Tokenize a nucleotide text string into integer codes."""
        tokens = [self.vocab.get(char.upper(), 5) for char in seq]
        # Pad or truncate
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def build_pyg_data(self, nodes: list, edges: list) -> Data:
        """
        Builds a single PyTorch Geometric Data object.
        nodes: list of dicts with keys: id, sequence, frequency
        edges: list of dicts with keys: source, target, frequency
        """
        # 1. Map Node IDs to 0-indexed values
        node_id_map = {n['id']: idx for idx, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # 2. Tokenize Node Sequences
        x_list = []
        node_freq = []
        for node in nodes:
            tokens = self.tokenize_sequence(node['sequence'], max_len=10)
            x_list.append(tokens)
            node_freq.append(node.get('frequency', 1.0))
            
        x_tokens = torch.tensor(x_list, dtype=torch.long) # [num_nodes, max_len]
        # Mean token value or flattened token list can serve as input features
        # In our case we use the token matrix directly
        
        # 3. Process Edges
        edge_sources = []
        edge_targets = []
        edge_freqs = []

        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            # Only include edges whose endpoints exist in current nodes
            if src in node_id_map and tgt in node_id_map:
                edge_sources.append(node_id_map[src])
                edge_targets.append(node_id_map[tgt])
                edge_freqs.append([edge.get('frequency', 1.0)])

        # Convert to PyTorch tensors
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) # [2, num_edges]
        edge_attr = torch.tensor(edge_freqs, dtype=torch.float)                 # [num_edges, 1]

        # Targets (Phenotype susceptibility and True Variant labels for testing)
        # Mock ground truths for validation
        y_impute = torch.tensor(node_freq, dtype=torch.float).unsqueeze(-1)      # [num_nodes, 1]
        y_pheno = torch.tensor(node_freq, dtype=torch.float).unsqueeze(-1) * 2.5 # [num_nodes, 1]

        data = Data(
            x=x_tokens,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_impute=y_impute,
            y_pheno=y_pheno
        )
        return data

    def get_loader(self, data: Data, batch_size: int = 4) -> NeighborLoader:
        """
        Creates a NeighborLoader for localized sub-graph batching.
        Prevents GPU memory overflow during backpropagation.
        """
        loader = NeighborLoader(
            data,
            num_neighbors=[4, 4], # Sample 4 neighbors at layer 1, and 4 at layer 2
            batch_size=batch_size,
            shuffle=True,
            input_nodes=None # Load all nodes as center nodes in batches
        )
        return loader

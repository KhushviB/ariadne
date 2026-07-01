import torch
from torch_geometric.data import Data
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
        node_id_map = {n['id']: idx for idx, n in enumerate(nodes)}
        num_nodes = len(nodes)

        x_list = []
        node_freq = []
        node_len_list = []
        for node in nodes:
            tokens = self.tokenize_sequence(node['sequence'], max_len=10)
            x_list.append(tokens)
            node_freq.append(node.get('frequency', 1.0))
            # Log-scaled sequence length: a physical signal, not a target label
            node_len_list.append([np.log1p(len(node.get('sequence', '')))])
            
        x_tokens = torch.tensor(x_list, dtype=torch.long) # [num_nodes, max_len]
        node_len = torch.tensor(node_len_list, dtype=torch.float) # [num_nodes, 1]
        
        edge_sources = []
        edge_targets = []
        edge_freqs = []

        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            if src in node_id_map and tgt in node_id_map:
                edge_sources.append(node_id_map[src])
                edge_targets.append(node_id_map[tgt])
                edge_freqs.append([edge.get('frequency', 1.0)])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long) # [2, num_edges]
        edge_attr = torch.tensor(edge_freqs, dtype=torch.float)                 # [num_edges, 1]

        # Compute per-node degree (in + out) on the FULL graph.
        # This is a critical topological feature: linear reference nodes have degree 2,
        # SV bubble entry/exit nodes have degree 3+. Without this, nucleotide sequences
        # alone cannot distinguish reference from variant nodes.
        from torch_geometric.utils import degree as pyg_degree
        deg_out = pyg_degree(edge_index[0], num_nodes=num_nodes)
        deg_in  = pyg_degree(edge_index[1], num_nodes=num_nodes)
        deg_total = (deg_out + deg_in).float().unsqueeze(1)   # [num_nodes, 1]
        node_degree = torch.log1p(deg_total)                  # log-scale: keeps high-degree outliers in range

        y_impute = torch.tensor(node_freq, dtype=torch.float).unsqueeze(-1)      # [num_nodes, 1]
        y_pheno = torch.tensor(node_freq, dtype=torch.float).unsqueeze(-1) * 2.5 # [num_nodes, 1]

        data = Data(
            x=x_tokens,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_impute=y_impute,
            y_pheno=y_pheno,
            node_degree=node_degree,
            node_len=node_len,
        )
        return data

    def get_loader(self, data: Data, batch_size: int = 2000, max_subgraphs: int = None):
        """
        Splits the chromosome graph into sequential subgraphs with a random starting
        offset so batch boundaries differ each epoch. Sequential slicing is correct for
        pangenome GFAs because consecutive node IDs ARE adjacent in the reference backbone
        — this preserves real bubble/branch edges within each batch.

        NeighborLoader was removed: it requires 'pyg-lib' or 'torch-sparse' which are
        not available in the Docker image. This pure-PyTorch approach has no external deps.

        All nodes in each subgraph are seed nodes (no context dilution), so loss is
        computed over the full batch without masking.
        """
        import random as _rnd
        num_nodes = data.num_nodes
        subgraphs = []

        # Random offset so we get different batch boundaries each epoch
        offset = _rnd.randint(0, max(0, batch_size - 1))

        for start in range(offset, num_nodes, batch_size):
            if max_subgraphs is not None and len(subgraphs) >= max_subgraphs:
                break
            end = min(start + batch_size, num_nodes)
            batch_nodes = torch.arange(start, end)
            sub = data.subgraph(batch_nodes)

            # Explicitly carry node_degree and node_len (custom attrs — subgraph() may skip them)
            if hasattr(data, 'node_degree') and data.node_degree is not None:
                sub.node_degree = data.node_degree[batch_nodes]
            if hasattr(data, 'node_len') and data.node_len is not None:
                sub.node_len = data.node_len[batch_nodes]

            if sub.edge_index.numel() > 0:
                subgraphs.append(sub)

        # Also include the initial offset chunk (nodes 0..offset-1) if any
        if offset > 0:
            batch_nodes = torch.arange(0, offset)
            sub = data.subgraph(batch_nodes)
            if hasattr(data, 'node_degree') and data.node_degree is not None:
                sub.node_degree = data.node_degree[batch_nodes]
            if hasattr(data, 'node_len') and data.node_len is not None:
                sub.node_len = data.node_len[batch_nodes]
            if sub.edge_index.numel() > 0:
                subgraphs.append(sub)

        if not subgraphs:
            fallback_nodes = torch.arange(0, min(batch_size, num_nodes))
            fallback = data.subgraph(fallback_nodes)
            if hasattr(data, 'node_degree') and data.node_degree is not None:
                fallback.node_degree = data.node_degree[fallback_nodes]
            if hasattr(data, 'node_len') and data.node_len is not None:
                fallback.node_len = data.node_len[fallback_nodes]
            subgraphs.append(fallback)

        return subgraphs
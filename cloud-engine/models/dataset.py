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
        for node in nodes:
            tokens = self.tokenize_sequence(node['sequence'], max_len=10)
            x_list.append(tokens)
            node_freq.append(node.get('frequency', 1.0))
            
        x_tokens = torch.tensor(x_list, dtype=torch.long) # [num_nodes, max_len]
        
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

    def get_loader(self, data: Data, batch_size: int = 2000, max_subgraphs: int = None) -> list:
        """
        Splits the single large graph into multiple connected subgraphs.
        Partitions nodes consecutively (representing chromosomal corridors)
        and filters out any subgraphs that contain 0 edges to avoid GNN crash.
        """
        num_nodes = data.num_nodes
        subgraphs = []
        
        # Sequentially scan blocks of nodes
        i = 0
        while i < num_nodes:
            if max_subgraphs is not None and len(subgraphs) >= max_subgraphs:
                break
            batch_nodes = torch.arange(i, min(i + batch_size, num_nodes))
            sub_data = data.subgraph(batch_nodes)
            
            # Only keep subgraphs that have actual structural variant links/edges
            if sub_data.edge_index.numel() > 0:
                subgraphs.append(sub_data)
                
            i += batch_size
            
        # Fallback: if no subgraphs with edges are found, return the first slice
        if not subgraphs:
            subgraphs.append(data.subgraph(torch.arange(0, min(batch_size, num_nodes))))
            
        return subgraphs

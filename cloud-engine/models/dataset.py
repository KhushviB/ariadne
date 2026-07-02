import torch
from torch_geometric.data import Data
import numpy as np
import math

class PangenomeDataset:
    """
    PanGNN v2 Dataset Builder.
    
    Translates parsed GFA components into PyG Data matrices with:
    - 3-mer frequency features (64 dims) over full node sequences
    - Bubble-Aware Positional Encoding (BAPE, 5 dims)
    - Log-scaled node degree (1 dim)
    - Log-scaled sequence length (1 dim)
    - Bidirectional edges with edge-type encoding
    
    Total node feature dimensionality: 71
    """
    
    BASE_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    K = 3
    KMER_DIM = 4 ** K  # 64 for 3-mers
    
    def compute_kmer_freqs(self, seq: str) -> list:
        """Compute normalized 3-mer frequency vector over the full node sequence."""
        counts = [0] * self.KMER_DIM
        seq_upper = seq.upper()
        n = len(seq_upper)
        valid_count = 0
        
        for i in range(n - self.K + 1):
            kmer = seq_upper[i:i + self.K]
            idx = 0
            valid = True
            for c in kmer:
                if c not in self.BASE_MAP:
                    valid = False
                    break
                idx = idx * 4 + self.BASE_MAP[c]
            if valid:
                counts[idx] += 1
                valid_count += 1
        
        # Normalize to frequency distribution
        if valid_count > 0:
            counts = [c / valid_count for c in counts]
        
        return counts

    def build_pyg_data(self, nodes: list, edges: list) -> Data:
        """
        Builds a single PyTorch Geometric Data object with full feature engineering.
        
        nodes: list of dicts with keys: id, sequence, frequency, bubble metadata
        edges: list of dicts with keys: source, target, frequency, edge_type
        """
        node_id_map = {n['id']: idx for idx, n in enumerate(nodes)}
        num_nodes = len(nodes)

        # ── Feature computation ────────────────────────────────────────────
        kmer_list = []       # [num_nodes, 64]
        bape_list = []       # [num_nodes, 5]
        degree_list = []     # placeholder, computed after edge_index
        length_list = []     # [num_nodes, 1]
        node_freq = []       # label target
        
        for node in nodes:
            seq = node.get('sequence', '')
            
            # 1. K-mer frequency features (64 dims)
            kmer_list.append(self.compute_kmer_freqs(seq))
            
            # 2. BAPE — Bubble-Aware Positional Encoding (5 dims)
            bape = [
                1.0 if node.get('is_source', False) else 0.0,
                1.0 if node.get('is_sink', False) else 0.0,
                1.0 if node.get('is_interior', False) else 0.0,
                float(node.get('path_position', 0.0)),
                math.log1p(float(node.get('n_paths', 0))),
            ]
            bape_list.append(bape)
            
            # 3. Log-scaled sequence length (1 dim)
            length_list.append([math.log1p(len(seq))])
            
            # 4. Label
            node_freq.append(node.get('frequency', 1.0))

        kmer_tensor = torch.tensor(kmer_list, dtype=torch.float)    # [N, 64]
        bape_tensor = torch.tensor(bape_list, dtype=torch.float)    # [N, 5]
        length_tensor = torch.tensor(length_list, dtype=torch.float) # [N, 1]
        
        # ── Edge construction (bidirectional) ──────────────────────────────
        edge_sources = []
        edge_targets = []
        edge_types = []  # 0.0 = backbone, 1.0 = branch

        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            if src in node_id_map and tgt in node_id_map:
                src_idx = node_id_map[src]
                tgt_idx = node_id_map[tgt]
                etype = 0.0 if edge.get('edge_type', 'backbone') == 'backbone' else 1.0
                
                # Forward edge
                edge_sources.append(src_idx)
                edge_targets.append(tgt_idx)
                edge_types.append([etype])
                
                # Reverse edge (bidirectional)
                edge_sources.append(tgt_idx)
                edge_targets.append(src_idx)
                edge_types.append([etype])

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.float)  # [2*E, 1]

        # ── Node degree (computed on bidirectional graph) ──────────────────
        from torch_geometric.utils import degree as pyg_degree
        deg = pyg_degree(edge_index[0], num_nodes=num_nodes)
        node_degree = torch.log1p(deg.float()).unsqueeze(1)  # [N, 1]
        
        # ── Assemble full feature matrix (71 dims) ────────────────────────
        # [kmer_64 | bape_5 | degree_1 | length_1] = 71
        x_features = torch.cat([
            kmer_tensor,     # 64
            bape_tensor,     # 5
            node_degree,     # 1
            length_tensor,   # 1
        ], dim=-1)  # [N, 71]

        # ── Labels ────────────────────────────────────────────────────────
        y_impute = torch.tensor(node_freq, dtype=torch.float).unsqueeze(-1)
        y_pheno = y_impute * 2.5

        data = Data(
            x=x_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_impute=y_impute,
            y_pheno=y_pheno,
        )
        return data

    def get_loader(self, data: Data, batch_size: int = 2000, max_subgraphs: int = None):
        """
        Splits the chromosome graph into sequential subgraphs with a random starting
        offset so batch boundaries differ each epoch. Sequential slicing preserves
        real bubble/branch edges within each batch since consecutive node IDs in
        vg-construct graphs ARE adjacent in the reference backbone.
        """
        import random as _rnd
        num_nodes = data.num_nodes
        subgraphs = []

        # Random offset for batch diversity across epochs
        offset = _rnd.randint(0, max(0, batch_size - 1))

        for start in range(offset, num_nodes, batch_size):
            if max_subgraphs is not None and len(subgraphs) >= max_subgraphs:
                break
            end = min(start + batch_size, num_nodes)
            batch_nodes = torch.arange(start, end)
            sub = data.subgraph(batch_nodes)
            if sub.edge_index.numel() > 0:
                subgraphs.append(sub)

        # Include the initial offset chunk
        if offset > 0:
            batch_nodes = torch.arange(0, offset)
            sub = data.subgraph(batch_nodes)
            if sub.edge_index.numel() > 0:
                subgraphs.append(sub)

        if not subgraphs:
            fallback_nodes = torch.arange(0, min(batch_size, num_nodes))
            fallback = data.subgraph(fallback_nodes)
            subgraphs.append(fallback)

        return subgraphs
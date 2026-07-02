import torch
from torch_geometric.data import Data
import numpy as np
import math

class PangenomeDataset:
    """
    PanGNN v2 Dataset Builder.
    
    Translates parsed GFA components into PyG Data matrices with:
    - 4-mer frequency features (256 dims) over full node sequences
    - Bubble-Aware Positional Encoding (BAPE, 5 dims)
    - Log-scaled node degree (1 dim)
    - Log-scaled absolute sequence length (1 dim)
    - Bubble-relative path sequence length (1 dim)
    - Bidirectional edges with edge-type encoding
    
    Total node feature dimensionality: 264
    """
    
    BASE_MAP = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    K = 4
    KMER_DIM = 4 ** K  # 256 for 4-mers
    
    def compute_kmer_freqs(self, seq: str) -> list:
        """Compute normalized 4-mer frequency vector over the full node sequence."""
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

        # ── Topology pre-computation ───────────────────────────────────────
        in_degrees = {}
        out_degrees = {}
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            out_degrees[src] = out_degrees.get(src, 0) + 1
            in_degrees[tgt] = in_degrees.get(tgt, 0) + 1

        # ── Feature computation ────────────────────────────────────────────
        kmer_array = np.zeros((num_nodes, 256), dtype=np.float32)
        bape_array = np.zeros((num_nodes, 9), dtype=np.float32)
        length_array = np.zeros((num_nodes, 1), dtype=np.float32)
        rel_length_array = np.zeros((num_nodes, 2), dtype=np.float32)
        node_freq = np.zeros((num_nodes, 1), dtype=np.float32)
        
        for idx, node in enumerate(nodes):
            seq = node.get('sequence', '')
            
            # 1. K-mer frequency features (256 dims)
            kmer_array[idx, :] = self.compute_kmer_freqs(seq)
            
            # 2. BAPE — Bubble-Aware Positional Encoding & Topology (9 dims)
            nid = node.get('id', '')
            bape_array[idx, :] = [
                1.0 if node.get('is_source', False) else 0.0,
                1.0 if node.get('is_sink', False) else 0.0,
                1.0 if node.get('is_interior', False) else 0.0,
                float(node.get('path_position', 0.0)),
                math.log1p(float(node.get('n_paths', 0))),
                math.log1p(float(node.get('bubble_depth', 0))),
                math.log1p(float(node.get('dist_to_sink', 0))),
                math.log1p(float(in_degrees.get(nid, 0))),
                math.log1p(float(out_degrees.get(nid, 0))),
            ]
            
            # 3. Log-scaled sequence length (1 dim)
            length_array[idx, 0] = math.log1p(len(seq))
            
            # 4. Bubble-relative path sequence length (2 dims)
            path_seq_len = float(node.get('path_seq_len', 0))
            max_path_seq_len = float(max(1, node.get('max_path_seq_len', 1)))
            avg_path_seq_len = float(max(1, node.get('avg_path_seq_len', 1)))
            rel_length_array[idx, 0] = path_seq_len / max_path_seq_len
            rel_length_array[idx, 1] = path_seq_len / avg_path_seq_len
            
            # 5. Label
            node_freq[idx, 0] = node.get('frequency', 1.0)

        kmer_tensor = torch.from_numpy(kmer_array)
        bape_tensor = torch.from_numpy(bape_array)
        length_tensor = torch.from_numpy(length_array)
        rel_length_tensor = torch.from_numpy(rel_length_array)
        
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
        
        # ── Assemble full feature matrix (269 dims) ────────────────────────
        # [kmer_256 | bape_9 | degree_1 | abs_len_1 | rel_len_2] = 269
        x_features = torch.cat([
            kmer_tensor,        # 256
            bape_tensor,        # 9
            node_degree,        # 1
            length_tensor,      # 1
            rel_length_tensor   # 2
        ], dim=-1)  # [N, 269]

        # ── Labels ────────────────────────────────────────────────────────
        y_impute = torch.from_numpy(node_freq)
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
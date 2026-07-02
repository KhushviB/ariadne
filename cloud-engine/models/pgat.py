import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

class PGATConv(MessagePassing):
    """
    Path-Aware Graph Attention (P-GAT) Conv Layer
    Formalized as:
    h_v^{(l+1)} = σ( W^{(l)} h_v^{(l)} + Σ_{u ∈ N(v)} α_{uv}^{(l)} M^{(l)}(h_u^{(l)}, e_{uv}) )
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, heads: int = 1):
        super(PGATConv, self).__init__(aggr='add', flow='source_to_target', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads

        # Weight matrices
        self.W = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.M = nn.Linear(in_channels + edge_dim, out_channels * heads, bias=False)
        
        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.M.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_dim] (edge type encoding)
        """
        # Step 1: Project node features
        h_nodes = self.W(x)
        
        # Step 2: Message passing
        out = self.propagate(edge_index, x=x, h_nodes=h_nodes, edge_attr=edge_attr, size=(x.size(0), x.size(0)))
        
        # Handle isolated nodes
        if out.size(0) < x.size(0):
            padding = torch.zeros(x.size(0) - out.size(0), *out.shape[1:], dtype=out.dtype, device=out.device)
            out = torch.cat([out, padding], dim=0)
            
        # Step 3: Mean head aggregation
        out = out.view(-1, self.heads, self.out_channels).mean(dim=1)
        
        # Step 4: Self-loop + non-linearity
        h_self = self.W(x).view(-1, self.heads, self.out_channels).mean(dim=1)
        out = out + h_self
        return F.elu(out)

    def message(self, x_j, h_nodes_i, h_nodes_j, edge_attr, index, ptr, size_i):
        """
        x_j: Source node raw features [num_edges, in_channels]
        h_nodes_i: Target projected features [num_edges, heads * out_channels]
        h_nodes_j: Source projected features [num_edges, heads * out_channels]
        edge_attr: Edge features (edge type) [num_edges, edge_dim]
        """
        h_nodes_i = h_nodes_i.view(-1, self.heads, self.out_channels)
        h_nodes_j = h_nodes_j.view(-1, self.heads, self.out_channels)

        # Message: concatenate source features + edge features, then project
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.M(msg_input)
        msg = msg.view(-1, self.heads, self.out_channels)

        # Attention coefficients
        alpha_input = torch.cat([h_nodes_i, h_nodes_j], dim=-1)
        alpha = (alpha_input * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)
        
        return msg * alpha.unsqueeze(-1)


class PanGNNModel(nn.Module):
    """
    PanGNN v2: Bubble-Aware Graph Attention Network for Pangenome Variant Detection.
    
    Architecture:
    - Feature encoder: projects 264-dim input (256 kmer + 5 BAPE + 1 degree + 1 abs_len + 1 rel_len)
      to hidden_dim through a 2-layer MLP
    - 3 P-GAT layers with residual connections for 3-hop receptive field
    - Classification head: binary ref vs alt allele prediction
    """
    def __init__(self, input_dim: int = 264, hidden_dim: int = 128, edge_dim: int = 1, heads: int = 4):
        super(PanGNNModel, self).__init__()
        
        # Feature encoder: projects heterogeneous 264-dim features to hidden_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        
        # 3-layer P-GAT with residual connections
        self.conv1 = PGATConv(hidden_dim, hidden_dim, edge_dim, heads=heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.conv2 = PGATConv(hidden_dim, hidden_dim, edge_dim, heads=heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.conv3 = PGATConv(hidden_dim, hidden_dim, edge_dim, heads=heads)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Predictor heads
        self.impute_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.phenotype_head = nn.Linear(hidden_dim, 1)

    def forward(self, x_features, edge_index, edge_attr):
        """
        x_features: Continuous feature matrix [num_nodes, input_dim=71]
                     Contains k-mer freqs, BAPE encoding, degree, length
        edge_index: [2, num_edges] (bidirectional)
        edge_attr: [num_edges, 1] (edge type: 0=backbone, 1=branch)
        """
        # Encode heterogeneous features to hidden_dim
        h = self.feature_encoder(x_features)  # [N, hidden_dim]
        
        # P-GAT Layer 1 with residual
        h_res = h
        h = self.conv1(h, edge_index, edge_attr)
        h = self.norm1(h + h_res)
        h = self.dropout(h)
        
        # P-GAT Layer 2 with residual
        h_res = h
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h + h_res)
        h = self.dropout(h)
        
        # P-GAT Layer 3 with residual
        h_res = h
        h = self.conv3(h, edge_index, edge_attr)
        h = self.norm3(h + h_res)
        
        # Predict
        impute_logits = self.impute_head(h)
        phenotype_risk = self.phenotype_head(h)
        
        return h, torch.sigmoid(impute_logits), phenotype_risk
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F

class PGATConv(MessagePassing):
    """
    Path-Aware Graph Attention (P-GAT) Conv Layer
    Formalized as:
    h_v^{(l+1)} = \sigma( W^{(l)} h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(l)} M^{(l)}(h_u^{(l)}, \mathbf{e}_{uv}) )
    """
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int, heads: int = 1):
        super(PGATConv, self).__init__(aggr='add', flow='source_to_target')
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
        edge_attr: [num_edges, edge_dim] (e.g. population transition frequency)
        """
        # Step 1: Project node features
        h_nodes = self.W(x).view(-1, self.heads, self.out_channels) # [num_nodes, heads, out_channels]
        
        # Step 2: Run message passing
        out = self.propagate(edge_index, x=x, h_nodes=h_nodes, edge_attr=edge_attr)
        
        # Step 3: Mean head aggregation
        out = out.mean(dim=1)
        
        # Step 4: Add self loop representation and apply non-linearity
        h_self = self.W(x)
        out = out + h_self
        return F.elu(out)

    def message(self, x_j, h_nodes_i, h_nodes_j, edge_attr, index, ptr, size_i):
        """
        x_j: Source node raw features [num_edges, in_channels]
        h_nodes_i: Target projected features [num_edges, heads, out_channels]
        h_nodes_j: Source projected features [num_edges, heads, out_channels]
        edge_attr: Edge features (population frequency) [num_edges, edge_dim]
        """
        # Step 1: Concatenate source node features and topological edge features for message mapping
        # Expand edge_attr to match nodes dims
        edge_feat = edge_attr.unsqueeze(1).repeat(1, self.heads, 1) # [num_edges, heads, edge_dim]
        x_j_expanded = x_j.unsqueeze(1).repeat(1, self.heads, 1)    # [num_edges, heads, in_channels]
        
        msg_input = torch.cat([x_j_expanded, edge_feat], dim=-1)     # [num_edges, heads, in_channels + edge_dim]
        msg = self.M(msg_input.view(-1, self.in_channels + self.edge_dim))
        msg = msg.view(-1, self.heads, self.out_channels)           # [num_edges, heads, out_channels]

        # Step 2: Compute structural attention coefficients (alpha)
        # alpha = LeakyReLU(a^T * [h_i || h_j])
        alpha_input = torch.cat([h_nodes_i, h_nodes_j], dim=-1)     # [num_edges, heads, 2 * out_channels]
        alpha = (alpha_input * self.att).sum(dim=-1)                # [num_edges, heads]
        alpha = F.leaky_relu(alpha, 0.2)
        
        # Softmax over neighboring nodes
        alpha = softmax(alpha, index, ptr, num_nodes=size_i)        # [num_edges, heads]
        
        # Return message weighted by attention
        return msg * alpha.unsqueeze(-1)


class PanGNNModel(nn.Module):
    """
    Consolidated PanGNN model mapping allele node sequences to embeddings 
    and predicting variant imputation probability and phenotypic risk scores.
    """
    def __init__(self, num_vocab: int, embed_dim: int, hidden_dim: int, edge_dim: int, heads: int = 2):
        super(PanGNNModel, self).__init__()
        # Nucleotide token embedding layer
        self.token_embed = nn.Embedding(num_vocab, embed_dim)
        
        self.conv1 = PGATConv(embed_dim, hidden_dim, edge_dim, heads=heads)
        self.conv2 = PGATConv(hidden_dim, hidden_dim, edge_dim, heads=heads)
        
        # Predictor heads
        self.impute_head = nn.Linear(hidden_dim, 1) # Imputation probability
        self.phenotype_head = nn.Linear(hidden_dim, 1) # Phenotypic risk score

    def forward(self, x_tokens, edge_index, edge_attr):
        """
        x_tokens: Integer indices of nucleotide sequence tokens [num_nodes]
        """
        # Convert tokens to continuous embeddings
        x = self.token_embed(x_tokens) # [num_nodes, embed_dim]
        
        # Pass through custom P-GAT layers
        h = self.conv1(x, edge_index, edge_attr)
        h = self.conv2(h, edge_index, edge_attr)
        
        # Predict logits
        impute_logits = self.impute_head(h)
        phenotype_risk = self.phenotype_head(h)
        
        return h, torch.sigmoid(impute_logits), phenotype_risk

import os
import torch
import torch.nn as nn
import torch.optim as optim
from .pgat import PanGNNModel
from .dataset import PangenomeDataset
import json

def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pt"):
    """Saves model checkpoint telemetry to disk."""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: '{filepath}'")

def load_checkpoint(checkpoint_path, model, optimizer):
    """Loads model checkpoint state."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, None
    
    print(f"Loading checkpoint '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Resuming from epoch {epoch} (loss: {loss:.4f})")
    return epoch, loss

def train_model(mock_data_path, checkpoint_dir, epochs=10, batch_size=4, lr=0.01):
    """Executescheckpoint-driven training loop for PanGNN."""
    # 1. Load data
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    gfa_21 = os.path.join(data_dir, "chr21.gfa")
    gfa_22 = os.path.join(data_dir, "chr22.gfa")

    all_nodes = []
    all_edges = []

    if os.path.exists(gfa_21) and os.path.exists(gfa_22):
        print("Real-world GFA files detected. Parsing biological graphs for GNN training...")
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data-pipeline")))
        from parse_gfa import parse_gfa
        
        nodes_21, edges_21 = parse_gfa(gfa_21)
        nodes_22, edges_22 = parse_gfa(gfa_22)
        
        all_nodes.extend(nodes_21)
        all_nodes.extend(nodes_22)
        all_edges.extend(edges_21)
        all_edges.extend(edges_22)
    else:
        print("Real GFA files not found. Using local simulated dataset...")
        with open(mock_data_path, 'r') as f:
            raw_data = json.load(f)
        
        for chr_key in ['21', '22']:
            chr_data = raw_data['chromosomes'][chr_key]
            all_nodes.extend(chr_data['nodes'])
            all_edges.extend(chr_data['edges'])

    # 2. Build dataset and loader
    ds = PangenomeDataset()
    data = ds.build_pyg_data(all_nodes, all_edges)
    loader = ds.get_loader(data, batch_size=batch_size)

    # 3. Model setup
    # x input features size: max_len=10 sequence tokens, vocab size=6
    # embed_dim=16, hidden_dim=32, edge_dim=1 (frequency)
    model = PanGNNModel(num_vocab=6, embed_dim=16, hidden_dim=32, edge_dim=1, heads=2)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    criterion_impute = nn.BCELoss() # Binary Cross Entropy for variant presence probability
    criterion_pheno = nn.MSELoss()  # Mean Squared Error for phenotypic risk score

    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    
    # Try to restore checkpoint
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

    print("Beginning GNN training loops...")
    model.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Since input tokens x is [batch_size, max_len], we take mean embedding in Conv
            # Reshape x to [batch_size] by taking first token, or flatten. 
            # In our dataset, nodes contain 10 token indices. We pass the token matrix to token embed.
            # PyG's NeighborLoader gathers node features.
            # batch.x shape is [batch_size, 10]
            # Since Embedding takes 1D tensor, we flatten tokens, run embedding, then pool back,
            # or modify pgat.py to pool embeddings.
            # Let's pool the embeddings across the token sequence dim:
            # We map token embed on flat [batch_size * 10], then mean pool to [batch_size, embed_dim]
            x_flat = batch.x.view(-1)
            
            # Call forward passing flat x_flat
            h, impute_prob, pheno_risk = model(x_flat, batch.edge_index, batch.edge_attr)
            
            # Since forward outputs for [batch_size * 10] nodes, we pool node embeddings by taking mean
            # over the sequence length 10.
            h_pooled = h.view(-1, 10, h.size(-1)).mean(dim=1)
            impute_prob_pooled = impute_prob.view(-1, 10, 1).mean(dim=1)
            pheno_risk_pooled = pheno_risk.view(-1, 10, 1).mean(dim=1)
            
            # Match sizes with targets
            loss_impute = criterion_impute(impute_prob_pooled, batch.y_impute)
            loss_pheno = criterion_pheno(pheno_risk_pooled, batch.y_pheno)
            
            loss = loss_impute + 0.4 * loss_pheno
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Save checkpoint after every epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_dir)

    print("GNN Training sequence completed successfully.")
    
    # Save final model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pangnn_final.pth"))

if __name__ == '__main__':
    # Local simulation test parameters
    train_model(
        mock_data_path=r"d:\BI\Ariadne\frontend\public\mock-data\mockGraph.json",
        checkpoint_dir=r"d:\BI\Ariadne\results",
        epochs=3
    )

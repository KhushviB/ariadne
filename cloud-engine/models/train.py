import os
import gc
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Add data-pipeline folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data-pipeline")))
from pgat import PanGNNModel
from dataset import PangenomeDataset
from parse_gfa import parse_gfa

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

def train_model(data_dir=None, checkpoint_dir=None, epochs=1, batch_size=2000, lr=0.01):
    """Executes memory-efficient incremental training loop for PanGNN."""
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

    gfa_files = glob.glob(os.path.join(data_dir, "*.gfa"))
    if not gfa_files:
        raise FileNotFoundError(f"No GFA graph files (*.gfa) found in {data_dir}. Please run data-pipeline/ingest.py first.")

    print(f"Detected {len(gfa_files)} GFA files. Starting incremental PyG dataset parsing...")

    # Folder to store processed PyG tensors
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    pyg_paths = []

    # 1. Parse and build PyG Data objects one chromosome at a time
    for idx, gfa_path in enumerate(gfa_files):
        filename = os.path.basename(gfa_path)
        chr_id = filename.replace("chr", "").replace(".gfa", "")
        pt_path = os.path.join(processed_dir, f"chr_{chr_id}.pt")
        
        # Re-parse GFA graphs to ensure fresh token and frequency alignments
        pass
            
        print(f"[{idx+1}/{len(gfa_files)}] Parsing GFA graph: {filename}...")
        try:
            nodes, edges = parse_gfa(gfa_path)
            
            print(f"Converting Chromosome {chr_id} to PyG tensor batch...")
            ds = PangenomeDataset()
            data = ds.build_pyg_data(nodes, edges)
            
            print(f"Saving processed tensor: {pt_path}")
            torch.save(data, pt_path)
            pyg_paths.append(pt_path)
            
            # Explicitly deallocate large lists and collect memory garbage
            del nodes
            del edges
            del data
            del ds
            gc.collect()
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}. Skipping this chromosome.")
            continue

    print(f"Successfully processed {len(pyg_paths)} chromosomes. Setting up training parameters...")

    # 2. Model setup
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
        num_batches = 0
        
        # Loop through each chromosome's saved PyG tensor graph
        for pt_path in pyg_paths:
            chr_name = os.path.basename(pt_path).replace(".pt", "")
            print(f"Epoch {epoch+1}/{epochs} | Loading {chr_name} for training batch...")
            
            try:
                chr_data = torch.load(pt_path)
                ds = PangenomeDataset()
                loader = ds.get_loader(chr_data, batch_size=batch_size)
                
                for batch in loader:
                    optimizer.zero_grad()
                    
                    # Pass node token matrix directly. Pooling is done inside the model.
                    h, impute_prob, pheno_risk = model(batch.x, batch.edge_index, batch.edge_attr)
                    
                    # Map frequency targets to binary classification targets
                    # (1.0 for variant slots where frequency < 0.9, 0.0 for reference nodes)
                    is_pos = (batch.y_impute < 0.9).float()
                    
                    # Compute dynamic class weights to balance the BCE loss
                    num_pos = is_pos.sum().item()
                    num_neg = (1.0 - is_pos).sum().item()
                    eps = 1e-7
                    
                    if num_pos > 0:
                        weight_pos = num_neg / num_pos
                        loss_impute = - (weight_pos * is_pos * torch.log(impute_prob + eps) + (1.0 - is_pos) * torch.log(1.0 - impute_prob + eps)).mean()
                    else:
                        loss_impute = - (is_pos * torch.log(impute_prob + eps) + (1.0 - is_pos) * torch.log(1.0 - impute_prob + eps)).mean()
                        
                    loss_pheno = criterion_pheno(pheno_risk, batch.y_pheno)
                    
                    loss = loss_impute + 0.4 * loss_pheno
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                # Free memory for this chromosome batch
                del chr_data
                del loader
                gc.collect()
                
            except Exception as e:
                print(f"Error training batch for {chr_name}: {e}. Skipping batch.")
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")
        
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
    train_model(epochs=1)

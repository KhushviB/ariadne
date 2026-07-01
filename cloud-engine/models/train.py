import os
import gc
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import sys

# Add models and data-pipeline folders to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
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

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Loads model checkpoint state onto the target device."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at '{checkpoint_path}'")
        return 0, None
    
    print(f"Loading checkpoint '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Resuming from epoch {epoch} (loss: {loss:.4f})")
    return epoch, loss

def focal_loss(pred, target, pos_weight, gamma=2.0, eps=1e-7):
    """
    Focal loss for extreme class imbalance.
    Downweights easy negatives (the 31:1 majority) so the model
    focuses gradient on hard positives (SV nodes) rather than
    collapsing to predict-everything-positive.
    gamma=2.0 is standard; pos_weight still corrects for imbalance.
    """
    pred = pred.clamp(eps, 1.0 - eps)
    bce_pos = -torch.log(pred)
    bce_neg = -torch.log(1.0 - pred)
    focal_pos = pos_weight * target * ((1.0 - pred) ** gamma) * bce_pos
    focal_neg = (1.0 - target) * (pred ** gamma) * bce_neg
    return (focal_pos + focal_neg).mean()

def train_model(data_dir=None, checkpoint_dir=None, epochs=1, batch_size=2000, lr=0.001):
    """Executes memory-efficient incremental training loop for PanGNN."""
    if data_dir is None:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

    gfa_files = [f for f in glob.glob(os.path.join(data_dir, "*.gfa"))
                 if "chr21.gfa" in os.path.basename(f) or "chr22.gfa" in os.path.basename(f)]
    if not gfa_files:
        raise FileNotFoundError(f"No chr21.gfa or chr22.gfa found in {data_dir}. Please run data-pipeline/ingest.py first.")

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

    # Calculate global label distribution for stable class-weighting
    print("Calculating global label distribution for stable class-weighting...", flush=True)
    global_pos = 0.0
    global_neg = 0.0
    for pt_path in pyg_paths:
        try:
            data = torch.load(pt_path, map_location="cpu")
            is_pos = (data.y_impute < 0.9).float()
            global_pos += is_pos.sum().item()
            global_neg += (1.0 - is_pos).sum().item()
        except Exception:
            pass
            
    global_pos_weight = 1.0
    if global_pos > 0:
        global_pos_weight = global_neg / global_pos
    print(f"Global distribution: Positives={int(global_pos)}, Negatives={int(global_neg)}, Ratio={global_pos_weight:.2f}:1", flush=True)

    # 2. Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    model = PanGNNModel(num_vocab=6, embed_dim=16, hidden_dim=32, edge_dim=1, heads=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss functions
    criterion_pheno = nn.MSELoss()  # Mean Squared Error for phenotypic risk score

    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    
    # Try to restore checkpoint
    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, device)

    print("Beginning GNN training loops...", flush=True)
    model.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        num_batches = 0
        
        pos_pred_sum = 0.0
        pos_pred_count = 0
        neg_pred_sum = 0.0
        neg_pred_count = 0
        
        # Loop through each chromosome's saved PyG tensor graph
        for pt_path in pyg_paths:
            chr_name = os.path.basename(pt_path).replace(".pt", "")
            print(f"Epoch {epoch+1}/{epochs} | Loading {chr_name} for training batch...", flush=True)
            
            try:
                chr_data = torch.load(pt_path, map_location="cpu")
                ds = PangenomeDataset()
                loader = ds.get_loader(chr_data, batch_size=batch_size)
                
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    # Pass node token matrix directly. Pooling is done inside the model.
                    h, impute_prob, pheno_risk = model(batch.x, batch.edge_index, batch.edge_attr)
                    
                    is_pos = (batch.y_impute < 0.9).float()
                    
                    # Accumulate prediction statistics for real-time telemetry
                    with torch.no_grad():
                        pos_mask = (is_pos == 1.0).squeeze(-1)
                        neg_mask = (is_pos == 0.0).squeeze(-1)
                        
                        pos_pred_sum += impute_prob.squeeze(-1)[pos_mask].sum().item()
                        pos_pred_count += pos_mask.sum().item()
                        neg_pred_sum += impute_prob.squeeze(-1)[neg_mask].sum().item()
                        neg_pred_count += neg_mask.sum().item()
                    
                    # Focal loss: handles 31:1 imbalance without collapsing
                    # to predict-everything-positive local minimum
                    loss_impute = focal_loss(impute_prob, is_pos, global_pos_weight)
                        
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
                print(f"Error training batch for {chr_name}: {e}. Skipping batch.", flush=True)
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        avg_pos_pred = pos_pred_sum / max(1, pos_pred_count)
        avg_neg_pred = neg_pred_sum / max(1, neg_pred_count)
        
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f} | "
              f"Avg Pos Pred (SV): {avg_pos_pred:.4f} | Avg Neg Pred (Ref): {avg_neg_pred:.4f}", flush=True)
        
        # Save checkpoint after every epoch
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_dir)

    print("GNN Training sequence completed successfully.", flush=True)
    
    # Save final model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pangnn_final.pth"))

if __name__ == '__main__':
    train_model(epochs=30)
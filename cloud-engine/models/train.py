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
    focuses gradient on hard positives (SV nodes).
    """
    pred = pred.clamp(eps, 1.0 - eps)
    bce_pos = -torch.log(pred)
    bce_neg = -torch.log(1.0 - pred)
    focal_pos = pos_weight * target * ((1.0 - pred) ** gamma) * bce_pos
    focal_neg = (1.0 - target) * (pred ** gamma) * bce_neg
    return (focal_pos + focal_neg).mean()

def train_model(data_dir=None, checkpoint_dir=None, epochs=50, batch_size=2000, lr=0.0005):
    """Executes PanGNN v2 training loop with superbubble-aware features."""
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
            
        print(f"[{idx+1}/{len(gfa_files)}] Parsing GFA graph: {filename}...")
        try:
            nodes, edges = parse_gfa(gfa_path)
            
            print(f"Converting Chromosome {chr_id} to PyG tensor batch...")
            ds = PangenomeDataset()
            data = ds.build_pyg_data(nodes, edges)
            
            print(f"Saving processed tensor: {pt_path}")
            torch.save(data, pt_path)
            pyg_paths.append(pt_path)
            
            # Explicitly deallocate
            del nodes
            del edges
            del data
            del ds
            gc.collect()
            
        except Exception as e:
            print(f"Error parsing {filename}: {e}. Skipping this chromosome.")
            import traceback
            traceback.print_exc()
            continue

    print(f"Successfully processed {len(pyg_paths)} chromosomes. Setting up training parameters...")

    # Calculate global label distribution for stable class-weighting
    print("Calculating global label distribution for stable class-weighting...", flush=True)
    global_pos = 0.0
    global_neg = 0.0
    input_dim = 71  # Default
    for pt_path in pyg_paths:
        try:
            data = torch.load(pt_path, map_location="cpu")
            is_pos = (data.y_impute < 0.9).float()
            global_pos += is_pos.sum().item()
            global_neg += (1.0 - is_pos).sum().item()
            input_dim = data.x.shape[1]  # Detect actual feature dim
        except Exception:
            pass
            
    global_pos_weight = 1.0
    if global_pos > 0:
        global_pos_weight = global_neg / global_pos
    print(f"Global distribution: Positives={int(global_pos)}, Negatives={int(global_neg)}, Ratio={global_pos_weight:.2f}:1", flush=True)
    print(f"Input feature dimension: {input_dim}", flush=True)

    # 2. Model setup — PanGNN v2 (no embeddings, continuous features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    model = PanGNNModel(input_dim=input_dim, hidden_dim=128, edge_dim=1, heads=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler — reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    
    # Try to restore checkpoint
    if os.path.exists(checkpoint_path):
        try:
            start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
        except Exception as e:
            print(f"Could not load checkpoint (architecture changed): {e}. Training from scratch.", flush=True)
            start_epoch = 0

    print(f"Beginning GNN training loops ({epochs} epochs, lr={lr})...", flush=True)
    model.train()
    
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        num_batches = 0
        
        pos_pred_sum = 0.0
        pos_pred_count = 0
        neg_pred_sum = 0.0
        neg_pred_count = 0
        
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

                    # PanGNN v2: continuous features, no extra kwargs
                    h, impute_prob, pheno_risk = model(
                        batch.x, batch.edge_index, batch.edge_attr
                    )

                    is_pos = (batch.y_impute < 0.9).float()

                    with torch.no_grad():
                        prob = impute_prob.squeeze(-1)
                        pos_mask = (is_pos == 1.0).squeeze(-1)
                        neg_mask = (is_pos == 0.0).squeeze(-1)
                        pos_pred_sum += prob[pos_mask].sum().item()
                        pos_pred_count += pos_mask.sum().item()
                        neg_pred_sum += prob[neg_mask].sum().item()
                        neg_pred_count += neg_mask.sum().item()

                    loss = focal_loss(impute_prob, is_pos, global_pos_weight)
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                del chr_data
                del loader
                gc.collect()
                
            except Exception as e:
                print(f"Error training batch for {chr_name}: {e}. Skipping batch.", flush=True)
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = total_loss / max(1, num_batches)
        avg_pos_pred = pos_pred_sum / max(1, pos_pred_count)
        avg_neg_pred = neg_pred_sum / max(1, neg_pred_count)
        
        # Step the learning rate scheduler
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Pos Pred: {avg_pos_pred:.4f} | Neg Pred: {avg_neg_pred:.4f} | "
              f"LR: {current_lr:.6f}", flush=True)
        
        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_dir)
        
        # Early stopping: only if extremely well-converged AND at least 15 epochs done
        if epoch >= 14 and avg_pos_pred > 0.95 and avg_neg_pred < 0.05 and avg_loss < 0.005:
            print(f"Early stopping: predictions well-separated at epoch {epoch+1}.", flush=True)
            break

    print("GNN Training sequence completed successfully.", flush=True)
    
    # Save final model weights
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pangnn_final.pth"))

if __name__ == '__main__':
    train_model(epochs=50, lr=0.0005)
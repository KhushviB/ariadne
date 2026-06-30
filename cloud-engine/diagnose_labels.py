import torch
import glob
import os

def diagnose():
    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "processed"))
    pt_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    
    if not pt_files:
        print(f"No processed PyG tensor files (*.pt) found in {processed_dir}. Please run models/train.py first.")
        return

    print("Analyzing processed label distributions across chromosomes...\n")
    total_pos, total_neg = 0, 0
    for pt_path in pt_files:
        try:
            data = torch.load(pt_path, map_location="cpu")
            # Frequency < 0.9 represents variant presence nodes (positives)
            is_pos = (data.y_impute < 0.9).float()
            pos = is_pos.sum().item()
            neg = (1.0 - is_pos).sum().item()
            total_pos += pos
            total_neg += neg
            ratio = neg / max(pos, 1)
            print(f"{os.path.basename(pt_path):<12}: pos={int(pos):<6} neg={int(neg):<8} ratio={ratio:.1f}:1")
        except Exception as e:
            print(f"Failed to read {os.path.basename(pt_path)}: {e}")
            
    total_ratio = total_neg / max(total_pos, 1)
    print(f"\nTOTAL       : pos={int(total_pos):<6} neg={int(total_neg):<8} ratio={total_ratio:.1f}:1")

if __name__ == '__main__':
    diagnose()

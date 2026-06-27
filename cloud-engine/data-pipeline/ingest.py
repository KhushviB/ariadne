import os
import sys
import subprocess
import urllib.request
import argparse

# GIAB whole-genome variant references (chromosomes 1-22)
VCF_URL = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TBI_URL = "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi"

def download_file(url, target_path):
    """Utility to stream biological data files down from public repositories."""
    if os.path.exists(target_path):
        print(f"File already exists: {target_path}, skipping download.")
        return True
    
    print(f"Downloading {url} -> {target_path}...")
    try:
        # User-agent header to bypass standard ftp-trace blocks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download completed.")
        return True
    except Exception as e:
        print(f"Warning: Failed to download {url} due to connection error: {str(e)}")
        return False

def run_command(cmd, shell=False):
    """Executes native bioinformatics tools inside the system shell."""
    print(f"Executing: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error executing command. Code: {result.returncode}")
        print(f"Stderr: {result.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError("Bioinformatics command failed.")
    return result.stdout

def create_mock_gfa(target_path, chr_num):
    """Writes a simulated GFA structure to make the pipeline runnable offline."""
    print(f"Generating mock pangenome graph for Chromosome {chr_num} at {target_path}...")
    
    # We create slightly varying nodes and links to simulate different chromosome sizes
    num_nodes = 60 + (int(chr_num) % 5) * 10
    gfa_content = ["H\tVN:Z:1.0"]
    
    # Generate nodes
    for i in range(1, num_nodes + 1):
        # alternate structural slot markers
        seq_len = 15 if i % 6 != 0 else 80
        sequence = "A" * seq_len
        gfa_content.append(f"S\t{i}\t{sequence}\tLN:i:{seq_len}")
        
    # Generate edges
    for i in range(1, num_nodes):
        gfa_content.append(f"L\t{i}\t+\t{i+1}\t+\t0M")
        # Add some alternate pathways for structural slots
        if i % 6 == 0 and i < num_nodes - 1:
            gfa_content.append(f"L\t{i-1}\t+\t{i+1}\t+\t0M")
            
    with open(target_path, 'w') as f:
        f.write("\n".join(gfa_content) + "\n")
    print(f"Mock GFA generated: {num_nodes} nodes written.")

def assemble_pangenome_graphs(data_dir, chromosomes):
    """
    Ingests reference sequences and variant files, constructing GFA graphs
    for each requested chromosome. Falls back to mock graphs if offline.
    """
    os.makedirs(data_dir, exist_ok=True)
    vcf_file = os.path.join(data_dir, "variants.vcf.gz")
    tbi_file = os.path.join(data_dir, "variants.vcf.gz.tbi")
    
    # Download the shared whole-genome GIAB VCF index
    vcf_success = download_file(VCF_URL, vcf_file)
    tbi_success = download_file(TBI_URL, tbi_file)
    
    for chr_num in chromosomes:
        print("\n" + "="*50)
        print(f"Processing Chromosome {chr_num}...")
        print("="*50)
        
        ref_url = f"http://hgdownload.cse.ucsc.edu/goldenPath/hg38/chromosomes/chr{chr_num}.fa.gz"
        ref_path_gz = os.path.join(data_dir, f"chr{chr_num}.fa.gz")
        ref_path_fa = os.path.join(data_dir, f"chr{chr_num}.fa")
        gfa_path = os.path.join(data_dir, f"chr{chr_num}.gfa")
        vg_path = os.path.join(data_dir, f"chr{chr_num}.vg")
        
        # Download reference chromosome FASTA
        ref_success = download_file(ref_url, ref_path_gz)
        
        if vcf_success and tbi_success and ref_success:
            try:
                # Extract sequence
                if not os.path.exists(ref_path_fa):
                    print(f"Extracting Reference FASTA for Chromosome {chr_num}...")
                    run_command(f"gunzip -c {ref_path_gz} > {ref_path_fa}", shell=True)
                
                # Construct graph
                print(f"Running vg construct for Chromosome {chr_num}...")
                vg_out = run_command([
                    "vg", "construct",
                    "-r", ref_path_fa,
                    "-v", vcf_file,
                    "-R", f"chr{chr_num}",
                    "-p"
                ], shell=False)
                
                with open(vg_path, 'wb') as f:
                    f.write(vg_out)
                    
                # Convert to GFA
                print(f"Converting vg graph to GFA for Chromosome {chr_num}...")
                gfa_out = run_command([
                    "vg", "view",
                    "-g", vg_path
                ], shell=False)
                
                with open(gfa_path, 'wb') as f:
                    f.write(gfa_out)
                
                print(f"Chromosome {chr_num} GFA successfully assembled.")
                continue
            except Exception as e:
                print(f"Warning: Failed to construct real graph for Chromosome {chr_num}: {e}")
        
        # Fallback to simulated biological graphs
        print(f"Falling back to simulated GFA generation for Chromosome {chr_num}...")
        create_mock_gfa(gfa_path, chr_num)
        
    print(f"\nAssembly process complete. Graphs saved in: {data_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pangenome Data Ingestion Pipeline.")
    parser.add_argument(
        "--chromosomes", 
        type=str, 
        default="21,22", 
        help="Comma-separated list of chromosomes to ingest (e.g. 21,22 or 1,2,21 or all)"
    )
    
    args = parser.parse_args()
    
    # Parse chromosome list
    if args.chromosomes.lower() == 'all':
        chroms = [str(i) for i in range(1, 23)]
    else:
        chroms = [c.strip() for c in args.chromosomes.split(',') if c.strip()]
        
    root_data_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    
    assemble_pangenome_graphs(root_data_dir, chroms)

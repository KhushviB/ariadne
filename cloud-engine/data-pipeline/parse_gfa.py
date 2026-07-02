import os
import json

# ---------------------------------------------------------------------------
# Path-Agnostic Superbubble Detection (Topology Only — No Reference Path Used)
# ---------------------------------------------------------------------------

def detect_superbubbles(adj, all_node_ids):
    """
    Detects superbubbles using ONLY graph topology (node degree).
    
    Does NOT use reference path information — the detection is purely structural.
    A bubble is defined as two high-degree nodes (degree >= 3) connected by
    2+ parallel paths through degree-2 interior nodes.
    
    In a vg-construct graph, every VCF variant creates exactly one bubble:
        source(deg>=3) → ref_allele(deg=2) → sink(deg>=3)
                       → alt_allele(deg=2) →
    
    BOTH ref_allele and alt_allele are marked as interior — no label leakage.
    
    Returns:
        bubbles: list of dicts with keys:
            - source: int
            - sink: int
            - interior: set of int (ALL interior nodes on ALL paths)
            - n_paths: int (number of parallel paths)
    """
    # 1. Compute degree for each node (using unique neighbors)
    degrees = {}
    for nid in all_node_ids:
        neighbors = adj.get(nid, [])
        degrees[nid] = len(set(neighbors))
    
    # 2. Find branching points (degree >= 3) — potential bubble boundaries
    branch_nodes = set(nid for nid, deg in degrees.items() if deg >= 3)
    
    # 3. From each branching node, trace paths through degree-2 nodes
    #    to find connecting branching nodes. A bubble exists when 2+ paths
    #    connect the same pair.
    bubbles = []
    visited_pairs = set()
    
    for source in branch_nodes:
        paths_from_source = {}  # sink_id -> list of paths (each path = list of interior nodes)
        
        source_neighbors = set(adj.get(source, []))
        
        for neighbor in source_neighbors:
            if neighbor == source:
                continue
            
            if neighbor in branch_nodes:
                # Direct edge source → neighbor (both high-degree)
                # This is a "0-interior-node" path (edge-only connection)
                sink = neighbor
                if sink not in paths_from_source:
                    paths_from_source[sink] = []
                paths_from_source[sink].append([])  # Empty interior
                continue
            
            # Trace through degree-2 nodes until we hit another branching node
            path_interior = [neighbor]
            current = neighbor
            prev = source
            found_sink = False
            
            max_steps = 500  # Safety limit for very long paths
            steps = 0
            
            while degrees.get(current, 0) == 2 and steps < max_steps:
                steps += 1
                current_neighbors = set(adj.get(current, []))
                next_nodes = [n for n in current_neighbors if n != prev]
                if not next_nodes:
                    break
                prev = current
                current = next_nodes[0]
                
                if current == source:
                    # Cycle back to source — skip
                    break
                    
                if degrees.get(current, 0) >= 3:
                    # Reached another branching node — this is the sink
                    found_sink = True
                    break
                else:
                    path_interior.append(current)
            
            if found_sink:
                sink = current
                if sink not in paths_from_source:
                    paths_from_source[sink] = []
                paths_from_source[sink].append(path_interior)
        
        # A bubble exists when 2+ paths connect source to the same sink
        for sink, paths in paths_from_source.items():
            pair = (min(source, sink), max(source, sink))
            if len(paths) >= 2 and pair not in visited_pairs:
                visited_pairs.add(pair)
                all_interior = set()
                for p in paths:
                    for nid in p:
                        all_interior.add(nid)
                
                bubbles.append({
                    'source': source,
                    'sink': sink,
                    'interior': all_interior,
                    'n_paths': len(paths),
                    'paths': paths,
                })
    
    return bubbles


def compute_bubble_metadata(bubbles):
    """
    Computes per-node bubble metadata from detected superbubbles.
    
    ALL interior nodes (both ref-allele and alt-allele paths) receive
    the same structural encoding. The model cannot distinguish ref from
    alt using BAPE alone — it must learn from sequence features.
    
    Returns:
        node_meta: dict mapping node_id -> bubble metadata dict
    """
    node_meta = {}
    
    for bubble_id, bubble in enumerate(bubbles):
        source = bubble['source']
        sink = bubble['sink']
        n_paths = bubble['n_paths']
        
        # Mark source (topological role — always degree >= 3)
        if source not in node_meta:
            node_meta[source] = {
                'bubble_id': bubble_id,
                'is_source': True,
                'is_sink': False,
                'is_interior': False,
                'n_paths': n_paths,
                'path_position': 0.0,
            }
        
        # Mark sink (topological role — always degree >= 3)
        if sink not in node_meta:
            node_meta[sink] = {
                'bubble_id': bubble_id,
                'is_source': False,
                'is_sink': True,
                'is_interior': False,
                'n_paths': n_paths,
                'path_position': 1.0,
            }
        
        # Mark ALL interior nodes on ALL paths (both ref and alt alleles)
        for path in bubble['paths']:
            path_len = len(path)
            for i, nid in enumerate(path):
                if nid not in node_meta:
                    pos = (i + 1) / max(path_len + 1, 2)
                    node_meta[nid] = {
                        'bubble_id': bubble_id,
                        'is_source': False,
                        'is_sink': False,
                        'is_interior': True,
                        'n_paths': n_paths,
                        'path_position': pos,
                    }
    
    return node_meta


# ---------------------------------------------------------------------------
# GFA Parser — Main Entry Point
# ---------------------------------------------------------------------------

def parse_gfa(gfa_path):
    """
    Parses segment (S) and linkage (L) lines from a GFA format file,
    detects superbubbles using ONLY graph topology (no reference path for 
    feature computation), and labels nodes by reference-path membership.
    
    KEY DESIGN: Features are derived from topology + sequence content.
                Labels are derived from reference-path membership.
                There is NO information flow from labels to features.
    """
    filename = os.path.basename(gfa_path)
    chr_id = filename.replace("chr", "").replace(".gfa", "")
    print(f"\n[GFA PARSER] Starting parsing of: {gfa_path} (Chromosome: {chr_id})...", flush=True)
    
    raw_nodes = []
    edges_raw = []
    ref_path_nodes = []
    
    with open(gfa_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            
            if parts[0] == 'S':
                node_id = int(parts[1])
                seq = parts[2]
                raw_nodes.append((node_id, seq))
                
            elif parts[0] == 'L':
                source = int(parts[1])
                target = int(parts[3])
                edges_raw.append((source, target))
                
            elif parts[0] == 'P':
                path_name = parts[1].replace("chr", "").replace("GRCh38.", "")
                if path_name == str(chr_id):
                    steps = parts[2].split(',')
                    for step in steps:
                        node_id = int(step.strip('+-'))
                        ref_path_nodes.append(node_id)
                        
            elif parts[0] == 'W':
                seq_id = parts[3].replace("chr", "").replace("GRCh38.", "")
                if seq_id == str(chr_id):
                    nodes_list = parts[6].replace('<', '>').split('>')
                    for node_item in nodes_list:
                        if node_item:
                            node_id = int(node_item.strip('+-'))
                            ref_path_nodes.append(node_id)

    print(f"[GFA PARSER] Parsed {len(raw_nodes)} nodes, {len(edges_raw)} edges. Reference path steps count: {len(ref_path_nodes)}", flush=True)

    if not ref_path_nodes:
        raise ValueError(
            f"CRITICAL ERROR: No reference path (P or W lines) matching Chromosome {chr_id} "
            f"found in GFA file '{gfa_path}'. Real coordinate alignment is impossible."
        )

    # 1. Build bidirectional adjacency map (used for topology-only analysis)
    print("[GFA PARSER] Building adjacency map...", flush=True)
    adj = {}
    for src, tgt in edges_raw:
        if src not in adj:
            adj[src] = []
        adj[src].append(tgt)
        if tgt not in adj:
            adj[tgt] = []
        adj[tgt].append(src)
    print(f"[GFA PARSER] Adjacency map indexed with {len(adj)} nodes.", flush=True)

    # 2. Detect superbubbles using ONLY graph topology (degree-based)
    #    NO reference path information is used here — this is the key
    #    anti-leakage design decision.
    all_node_ids = [nid for nid, _ in raw_nodes]
    
    print("[GFA PARSER] Detecting superbubbles (topology-only, path-agnostic)...", flush=True)
    bubbles = detect_superbubbles(adj, all_node_ids)
    print(f"[GFA PARSER] Detected {len(bubbles)} superbubbles.", flush=True)
    
    # 3. Compute per-node bubble metadata (same encoding for ref AND alt alleles)
    bubble_meta = compute_bubble_metadata(bubbles)
    
    # Count how many interior nodes are ref vs alt for diagnostic purposes
    ref_path_set = set(ref_path_nodes)
    interior_ref = sum(1 for nid, m in bubble_meta.items() if m['is_interior'] and nid in ref_path_set)
    interior_alt = sum(1 for nid, m in bubble_meta.items() if m['is_interior'] and nid not in ref_path_set)
    print(f"[GFA PARSER] Bubble metadata: {len(bubble_meta)} nodes total, "
          f"{interior_ref} interior-ref, {interior_alt} interior-alt (both get is_interior=True)", flush=True)

    # 4. Build node list — label by reference-path membership (purely topological label)
    #    Features (BAPE) are derived from graph structure only, NOT from the label.
    nodes = []
    alt_count = 0
    ref_count = 0
    
    for nid, seq in raw_nodes:
        if nid in ref_path_set:
            node_type = "Reference"
            frequency = 1.0
            ref_count += 1
        else:
            node_type = "Structural Variant Slot"
            frequency = 0.5
            alt_count += 1
        
        meta = bubble_meta.get(nid, {
            'bubble_id': -1,
            'is_source': False,
            'is_sink': False,
            'is_interior': False,
            'n_paths': 0,
            'path_position': 0.0,
        })
        
        nodes.append({
            "id": nid,
            "sequence": seq,
            "type": node_type,
            "frequency": frequency,
            "bubble_id": meta['bubble_id'],
            "is_source": meta['is_source'],
            "is_sink": meta['is_sink'],
            "is_interior": meta['is_interior'],
            "n_paths": meta['n_paths'],
            "path_position": meta['path_position'],
        })

    print(f"[GFA PARSER] Labeling complete. Reference: {ref_count}, Alternative: {alt_count}", flush=True)

    # 5. Build edges with uniform weights + topological edge type
    #    Edge type is based on degree of endpoints, NOT on label
    edges = []
    degrees = {nid: len(set(adj.get(nid, []))) for nid in all_node_ids}
    
    for source, target in edges_raw:
        # Edge type: backbone if both endpoints have degree <= 2, branch otherwise
        is_backbone = (degrees.get(source, 0) <= 2 and degrees.get(target, 0) <= 2)
        
        edges.append({
            "source": source,
            "target": target,
            "frequency": 1.0,
            "edge_type": "backbone" if is_backbone else "branch",
            "attention": 0.0,
        })

    print(f"[GFA PARSER] Ingestion complete. {alt_count}/{len(nodes)} nodes labeled as alternative alleles.", flush=True)
    return nodes, edges


def save_parsed_graph(nodes, edges, output_json_path):
    """Saves parsed GFA topology to a JSON file."""
    data = {
        "nodes": nodes,
        "edges": edges
    }
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Graph schema saved to: {output_json_path}")

if __name__ == '__main__':
    pass

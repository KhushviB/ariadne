# PanGNN: A Topological Foundation Model for Structural Variant Imputation and Phenotypic Mapping on Heterogeneous Pangenome Graphs

Project Ariadne (**PanGNN**) introduces a novel Graph Neural Network architecture designed to operate directly on graphical pangenomes (Directed Acyclic Graphs in GFA formats) to impute structural variants ($\ge$50bp) and predict clinical phenotypes. By replacing traditional flat sequence coordinates with spatial attention pathway learning, PanGNN eliminates alignment reference biases that skew standard DNA sequencing pipelines.

---

## 🧬 Architectural Overview

The project is structured as a hybrid monorepo, separating lightweight visualization elements from heavy biological compute pipelines:
*   `/frontend`: An interactive researcher dashboard built on **Next.js**, **Vanilla CSS**, and **Three.js**. It maps GFA segment structures as a 3D DNA "Tube Map," highlighting GNN attention coefficients in real time.
*   `/cloud-engine`: The biological compute module containing the **PyTorch Geometric (PyG)** message passing layers, `vg toolkit` command-mapping utilities, a **FastAPI** coordinate query routing service, a **Qdrant Vector DB** index layer, and validation benchmarking scripts.

---

## 📊 Scientific Thesis & GNN Formalization

Unlike standard graph convolutional layers, PanGNN implements a custom **Path-Aware Graph Attention (P-GAT)** model that integrates directional sequence trajectories and population frequencies directly into edge parameters. The node state is updated at layer $l+1$ as follows:

$$h_v^{(l+1)} = \sigma \left( \mathbf{W}^{(l)} h_v^{(l)} + \sum_{u \in \mathcal{N}(v)} \alpha_{uv}^{(l)} \mathbf{M}^{(l)}(h_u^{(l)}, \mathbf{e}_{uv}) \right)$$

Where:
*   $h_v^{(l)}$ is the embedding vector of allele node $v$ at layer $l$.
*   $\mathcal{N}(v)$ represents the set of neighboring genomic variations.
*   $\mathbf{e}_{uv}$ is the topological edge feature representing empirical population transition frequency.
*   $\alpha_{uv}^{(l)}$ is a structural attention coefficient computed via softmax over hypervariable genomic corridors.

---

## 🚀 Quickstart: Local Workspace Validation

To test and visual-audit the interface on a local Windows machine, follow these steps (no GPU or external biological database required):

### 1. Launch the WebGL Interactive Dashboard
```bash
# Navigate to the frontend directory
cd frontend

# Install Node and Visualizer dependencies
npm install

# Start the local hot-reloading development server
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser to inspect the 3D chromosome model.

### 2. Run the Benchmarking & Telemetry Suite
Verify local statistics calculation and log generation. These scripts output JSON reports to a local `/results` directory:
```bash
# Navigate to the workspace root
cd ..

# Install standard Python requirements
pip install -r cloud-engine/requirements.txt

# Run the Truvari-mode competitor accuracy evaluator
python cloud-engine/benchmarking/evaluate.py

# Run the hardware telemetry resource profiler
python cloud-engine/benchmarking/telemetry.py

# Run the Friedman and Holm-corrected Wilcoxon significance checks
python cloud-engine/benchmarking/statistics.py
```

---

## ☁️ Cloud Handoff (Remote Linux GPU Node)

Once local simulation and code layout adjustments are validated:
1.  Push the monorepo workspace to GitHub.
2.  Clone the repository on a remote GPU container
3.  Deploy the environment using Docker:
    ```bash
    docker build -t pangnn-engine ./cloud-engine
    docker run -d -p 8000:8000 --gpus all pangnn-engine
    ```
4.  Execute the biological ingest loop to download and process Chromosomes 21 and 22 from the HPRC and GIAB reference databases:
    ```bash
    python cloud-engine/data-pipeline/ingest.py
    ```
5.  Train the GNN model layers:
    ```bash
    python cloud-engine/models/train.py
    ```
6.  Launch the FastAPI server and direct the local Next.js client to query the remote IP address.

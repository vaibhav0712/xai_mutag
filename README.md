# Mutagenicity XAI GNN Pipeline

An end-to-end Explainable AI (XAI) pipeline using Relational Graph Convolutional Networks (RGCN) and GNNExplainer on the Mutagenicity dataset.

---

## Folder & File Structure

* **`mutag-hetero/`**: Contains raw dataset, stripped triple files (`.nt`), and processed dataset tensors (`.pt`).
* **`config.py`**: Centralized configuration file holding path definitions for datasets, model checkpoints, and result outputs.
* **`RGCN_Train.py`**: Defines and trains a 2-layer Relational GNN (RGCN) to predict molecule mutagenicity.
* **`RGCN_Kfold.py`**: Implements a 10-fold Stratified Cross-Validation loop to evaluate training stability.
* **`GNNExplainer.py`**: Generates local explanations (node/edge importances) for a single molecule and outputs a payload.json.
* **`Evaluate_GNNExplainer.py`**: Evaluates GNNExplainer performance across the test set (fidelity, sparsity, unfaithfulness). (See [other/EVALUATION.md]for details).
* **`run_pipeline.sh`**: Orchestrator bash script that automates the entire pipeline sequentially.
* **`model/`**: Stores trained GNN model checkpoints.
* **`results/`**: Stores training performance plots, 10-fold CV performance boxplot, explanation metrics etc.
* **`data_preprocessing.py`**: Parses the RDF graph, encodes atoms and bonds, and serializes molecules into PyTorch Geometric (`PyG`) format. (See [other/DATA_PREPROCESSING.md] for details).
* **`other/data_exploration.py`**: Performs exploratory data analysis (EDA) on both the raw `.nt` RDF graph and the preprocessed PyG dataset.

---

## Environment Setup

The active environment uses **Python 3.12.13** under the environment name `xai`.

### Set up the Conda environment:
```bash
# Create the environment with the correct Python version
conda create -n xai python=3.12.13 -y
conda activate xai

# Install necessary dependencies
pip install -r requirements.txt
```

---

## Execution Guide

The orchestration script **`run_pipeline.sh`** automates the execution of the entire workflow sequentially:
1. Preprocesses the RDF data
2. Trains the RGCN model
3. Evaluates training using 10-Fold Cross Validation
4. Generates an explanation payload for the default molecule (`d50`)
5. Evaluates GNNExplainer metrics on the test set

### Running on Linux / macOS:
```bash
# Ensure execution permissions
chmod +x run_pipeline.sh

# Run the pipeline
./run_pipeline.sh
```

### Running on Windows:
Open a **Git Bash**, **WSL**, or **MSYS2** terminal and run:
```bash
./run_pipeline.sh
```
### Note 
**`If .sh does not execute properly, please run each step mentioned in the .sh file manually in the terminal.`**
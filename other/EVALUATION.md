# GNNExplainer Evaluation Documentation

This document explains the evaluation pipeline implemented in [Evaluate_GNNExplainer.py], focusing on the implementation of custom metrics (Fidelity and Sparsity) and explaining the technical reasons why built-in PyTorch Geometric (`PyG`) metrics could not be used.

---

## 1. Overview of Evaluation

The evaluation script loads the pre-trained Relational GCN model and runs the GNNExplainer on the test set of molecules. It measures three key metrics to quantify explanation quality:
* **Fidelity (+) & (-)**: Quantifies how necessary and sufficient the explanation subgraph is to the model's predictions.
* **Unfaithfulness**: Measures how well the explainer approximates the model's decision boundary locally.
* **Sparsity**: Measures how concise and focused the explanation is (higher sparsity means fewer nodes/edges are required to explain the prediction).

---

## 2. Why Built-In PyG Metrics Failed

During development, the built-in metrics from `torch_geometric.explain.metric` (such as `fidelity`) could not be used due to two major technical limitations:

### A. Shape Mismatches in Relational GNNs (RGCN)
* **The Issue**: Our model uses `FastRGCNConv` layer types, which require a relational `edge_type` tensor of the same length as the `edge_index` in every forward pass.
* **The Failure**: PyG's built-in `fidelity` metric internally masks and removes edges from the graph to compute subgraphs, but it only sub-selects the `edge_index` tensor. It does not sub-select the associated `edge_type` tensor. When the model tries to process the modified graph, it crashes with shape mismatch errors because the number of relation types (`edge_type`) no longer matches the number of edges (`edge_index`).
* **Our Solution**: The custom `compute_fidelity` function explicitly filters **both** `edge_index` and `edge_type` using the same mask (`keep = mask > 0.5`), keeping the relational dimensions aligned.

### B. Coarse Binary Fidelity vs. Continuous Probability Drops
* **The Issue**: PyG's built-in fidelity metric operates on a binary classification threshold (did the predicted class flip from `0` to `1` or vice-versa?).
* **The Failure**: If removing an explanation reduces the model's confidence from `95%` to `51%`, the predicted class remains the same (still class `1`). In this case, PyG's binary fidelity returns `0.0` (no change), completely failing to capture that the removed features carried `44%` of the model's prediction confidence.
* **Our Solution**: We implemented a continuous, probability-based fidelity function that measures the exact drop/gain in the softmax probability of the predicted class:
  * **Fidelity+ (Necessity)**: Measures the drop in confidence when the explanation subgraph is removed. High values indicate the model relies heavily on those features.
  * **Fidelity- (Sufficiency)**: Measures the confidence retained when *only* the explanation subgraph is kept. High values indicate the explanation alone is sufficient.

---

## 3. Custom Metric Formulations

### Continuous Fidelity
* **Fidelity+**:
  $$\text{Fidelity}^+ = \max(P_{\text{original}}(c) - P_{\text{complement}}(c), 0)$$
  *(where $c$ is the predicted class, and the complement is the graph with explanation features removed)*
* **Fidelity-**:
  $$\text{Fidelity}^- = \max(P_{\text{original}}(c) - P_{\text{subgraph}}(c), 0)$$
  *(where the subgraph is the graph containing only the explanation features)*

### Top-K Sparsity
* **The Issue**: Standard sparsity measures the fraction of edge weights below a threshold. However, since the visualization tool ([visualizer.html]) displays only the top $k=10$ most important edges, standard sparsity does not represent what the user actually sees.
* **Our Solution**: We implemented a custom top-k sparsity metric:
  1. Identifies the top $k=10$ edges by importance score.
  2. Finds the set of incident nodes connected to those top-k edges.
  3. Computes sparsity as the fraction of graph elements (nodes and edges) *omitted* from this sub-selection:
     $$\text{Sparsity} = 1 - \frac{\text{selected\_nodes} + \text{selected\_edges}}{\text{total\_nodes} + \text{total\_edges}}$$
  4. This provides a normalized metric showing how concise the visual explanation is relative to the size of the molecule.
